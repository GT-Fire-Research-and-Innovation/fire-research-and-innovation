"""Data-grab statistics: pixel counts and input normalization in window_config.json, read before training."""

import json
import sys
import tracemalloc
import zipfile
from pathlib import Path

import numpy as np
import pytest

import synth
from firepred.config.dataset import NORMALIZATION_MEAN, NORMALIZATION_STD
from firepred.config.training import PAPER_CE_WEIGHT
from firepred.data.files import zip_folders
from firepred.data.stats import (
    NORMALIZATION,
    PIXEL_COUNTS,
    ChannelMoments,
    MissingStatistics,
    WindowConfig,
    count_labels,
    fire_moments,
    keep_statistics,
    split_normalization,
)
from firepred.training.config import RunConfig
from firepred.training.experiment import (
    Experiment,
    FireData,
    checkpoint_normalization,
    with_data_statistics,
)


def reference(root):
    """The window_config.json synth wrote with plain numpy (the expected statistics)."""
    return json.loads((Path(root) / "window_config.json").read_text())


def view(root, target, config=None):
    """A data root at `target` that links root's splits and holds `config` as its window_config.json (None: none)."""
    target.mkdir(parents=True)
    for name in ("train", "val", "test", "tessera"):
        (target / name).symlink_to(Path(root) / name, target_is_directory=True)
    if config is not None:
        (target / "window_config.json").write_text(json.dumps(config))
    return target


def without(config, *keys):
    """A copy of a window config without the given blocks."""
    return {key: value for key, value in config.items() if key not in keys}


# ----- measuring ----- #


def test_moments_merge_exactly_and_survive_a_large_offset():
    """Window-by-window moments equal whole-array statistics, even for 300 K +- 10 K values."""
    rng = np.random.default_rng(0)
    windows = [(300.0 + 10.0 * rng.standard_normal((3, 2, 16, 16))).astype(np.float32) for _ in range(5)]
    windows[2][1] = 7.0  # one window where a channel never varies
    total = ChannelMoments.empty(3)
    for window in windows:
        total = total.merge(ChannelMoments.of_window(window))
    every = np.concatenate([w.reshape(3, -1).astype(np.float64) for w in windows], axis=1)
    assert total.count == every.shape[1]
    np.testing.assert_allclose(total.mean, every.mean(axis=1), rtol=1e-13)
    np.testing.assert_allclose(total.std, every.std(axis=1), rtol=1e-11)


def test_normalization_matches_numpy_and_ignores_threads_and_unzipping(data_root, tmp_path):
    """The lazy, threaded measurement equals numpy over every value; thread count and unzipping change nothing."""
    from firepred.data.archive import unzip_data_root

    root, _ = data_root
    expected = reference(root)[NORMALIZATION]
    stats = split_normalization(root / "train", workers=1)
    assert list(stats.fires) == expected["fires"] and stats.windows == expected["windows"]
    assert stats.values_per_channel == expected["values_per_channel"]
    np.testing.assert_allclose(stats.mean, expected["mean"], rtol=1e-12)
    np.testing.assert_allclose(stats.std, expected["std"], rtol=1e-12)
    assert split_normalization(root / "train", workers=4).mean == stats.mean
    unzipped = unzip_data_root(root, tmp_path / "npy", splits=("train",))
    assert split_normalization(unzipped / "train").std == stats.std


def test_a_constant_channel_gets_std_one(tmp_path):
    """A channel that never varies is only shifted, never divided by zero."""
    split = tmp_path / "train"
    split.mkdir()
    synth.fire_archive(split / "p1.npz", np.random.default_rng(1), ts_length=1, n_windows=2)
    with np.load(split / "p1.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key in ("data_000", "data_001"):
        arrays[key][5] = 3.0
    synth._write_npz(split / "p1.npz", arrays, compressed=False)
    stats = split_normalization(split)
    assert stats.mean[5] == 3.0 and stats.std[5] == 1.0


def test_normalization_holds_one_window_at_a_time(tmp_path):
    """Peak memory while measuring a fire stays under two windows, however many windows the fire has."""
    path = tmp_path / "p1.npz"
    synth.fire_archive(path, np.random.default_rng(2), ts_length=2, n_windows=6)
    window_bytes = 27 * 2 * 256 * 256 * 4
    tracemalloc.start()
    try:
        windows, _ = fire_moments(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert windows == 6
    assert peak < 2 * window_bytes, f"peak {peak / 2**20:.1f} MiB for {window_bytes / 2**20:.1f} MiB windows"


# ----- the data-grab commands ----- #


def test_data_grab_commands_write_both_blocks(data_root, tmp_path):
    """count-pixels and compute-normalization add their blocks to the data root's file and keep the settings."""
    from firepred.data import windows_master

    root, _ = data_root
    expected = reference(root)
    target = view(root, tmp_path / "data", without(expected, PIXEL_COUNTS, NORMALIZATION))
    report = windows_master.count_pixels(target, splits=("train", "val"), workers=2)
    assert set(report.counts) == {"train", "val"}
    written = reference(target)
    assert written[PIXEL_COUNTS] == {split: expected[PIXEL_COUNTS][split] for split in ("train", "val")}
    windows_master.count_pixels(target, splits=("test",))
    assert reference(target)[PIXEL_COUNTS] == expected[PIXEL_COUNTS]  # earlier splits kept

    windows_master.compute_normalization(target)
    written = reference(target)
    np.testing.assert_allclose(written[NORMALIZATION]["mean"], expected[NORMALIZATION]["mean"], rtol=1e-12)
    assert written[NORMALIZATION]["channels"][0] == "VIIRS I1"
    assert {key: written[key] for key in ("ts_length", "ts_interval")} == {"ts_length": 6, "ts_interval": 1}
    assert not list(target.glob(".*partial"))


def test_download_keeps_local_statistics_the_repo_lacks():
    """A fresh copy of the repo's file keeps statistics computed locally; the repo's own win."""
    local = {"ts_length": 6, PIXEL_COUNTS: {"train": "local"}, NORMALIZATION: "local"}
    assert keep_statistics({"ts_length": 6}, local) == local
    assert keep_statistics({"ts_length": 6, NORMALIZATION: "repo"}, local)[NORMALIZATION] == "repo"
    assert keep_statistics({"ts_length": 6}, None) == {"ts_length": 6}


def test_data_zip_carries_window_config_at_its_root(tmp_path):
    """upload-zip's zip holds window_config.json beside train/, so unpacking puts it in the data root."""
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "p1.npz").write_bytes(b"x")
    (tmp_path / "window_config.json").write_text("{}")
    members = zip_folders([tmp_path / "train"], tmp_path / "data.zip", extra_files=[tmp_path / "window_config.json"])
    assert members == ["train/p1.npz", "window_config.json"]
    with zipfile.ZipFile(tmp_path / "data.zip") as bundle:
        assert bundle.read("window_config.json") == b"{}"


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """A fake Hugging Face holding one window repo whose window_config.json records 6-day windows."""
    pytest.importorskip("huggingface_hub")
    import fake_hub as hub_module

    monkeypatch.setattr(hub_module, "ROOT", str(tmp_path / "hub"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    api = hub_module.HfApi()
    api.create_repo("owner/windows-6-1")
    api.upload_file(json.dumps({"ts_length": 6, "ts_interval": 1}).encode(), "window_config.json",
                    "owner/windows-6-1")
    return hub_module


def test_missing_local_file_comes_from_the_repo_and_uploads_check_settings(data_root, tmp_path, fake_hub):
    """Without a local copy the repo's file is used; --upload refuses a repo that records other settings."""
    from firepred.data import windows_master

    root, _ = data_root
    target = view(root, tmp_path / "data")
    report = windows_master.count_pixels(target, splits=("train",), window_repo="owner/windows-6-1", upload=True)
    assert report.uploaded_to == "owner/windows-6-1"
    remote = json.loads((fake_hub._repo("owner/windows-6-1") / "window_config.json").read_text())
    assert remote[PIXEL_COUNTS]["train"] == reference(root)[PIXEL_COUNTS]["train"]

    other = view(root, tmp_path / "other", {"ts_length": 10, "ts_interval": 1})
    with pytest.raises(ValueError, match="other window settings"):
        windows_master.count_pixels(other, splits=("train",), window_repo="owner/windows-6-1", upload=True)


# ----- training reads the file first ----- #


def test_training_refuses_other_window_settings(data_root):
    """-ts must match the window settings the file records."""
    root, _ = data_root
    with pytest.raises(ValueError, match="-ts 10"):
        Experiment(RunConfig("tiny3d", ts_length=10, data_root=root)).run()


def test_missing_statistics_fall_back_to_official_values(data_root, tmp_path):
    """No file, no blocks, or a stale block: the official TS-SatFire values, each recorded with its reason."""
    from firepred.training.config import LossConfig

    root, _ = data_root
    stale = reference(root)
    stale[NORMALIZATION] = dict(stale[NORMALIZATION], fires=stale[NORMALIZATION]["fires"][:1])
    stale[PIXEL_COUNTS] = {}
    cases = {"none": (None, "No window_config.json"),
             "bare": (without(reference(root), NORMALIZATION, PIXEL_COUNTS), "compute-normalization"),
             "stale": (stale, "without 1 fires")}
    for name, (contents, reason) in cases.items():
        target = view(root, tmp_path / name, contents)
        window_config = WindowConfig.read(target)
        assert window_config.found == (contents is not None)
        config = with_data_statistics(RunConfig("tiny3d", data_root=target, loss=LossConfig("dicece", "auto")),
                                      window_config, target / "train")
        assert config.normalization == "paper"
        assert config.normalization_stats == (tuple(NORMALIZATION_MEAN), tuple(NORMALIZATION_STD))
        assert config.loss.ce_weight == PAPER_CE_WEIGHT and config.loss.ce_weight_counts is None
        assert [what for what, _ in config.defaults_used] == ["normalization", "ce_weight"]
        logged = config.wandb_config()
        assert reason in logged["defaults_used"]["normalization"]
        assert logged["normalization"] == "paper" and logged["ce_weight"] == list(PAPER_CE_WEIGHT)
        assert config.checkpoint_settings(None)["normalization"]["source"] == "paper"
    assert "defaults_used" not in RunConfig("tiny3d").wandb_config()


def test_statistics_that_miss_a_training_fire_are_refused(data_root, tmp_path):
    """Statistics computed before a fire was added do not describe the data; extra fires are only noted."""
    root, _ = data_root
    stale = reference(root)
    stale[NORMALIZATION] = dict(stale[NORMALIZATION], fires=stale[NORMALIZATION]["fires"][:1])
    target = view(root, tmp_path / "stale", stale)
    with pytest.raises(MissingStatistics, match="without 1 fires"):
        WindowConfig.read(target).normalization(target / "train")
    extra = reference(root)
    extra[NORMALIZATION] = dict(extra[NORMALIZATION], fires=extra[NORMALIZATION]["fires"] + ["p99999999"])
    target = view(root, tmp_path / "extra", extra)
    _, unused = WindowConfig.read(target).normalization(target / "train")
    assert unused == ("p99999999",)


def test_runs_take_their_numbers_from_window_config(data_root):
    """window-config normalization and --ce-weight auto use the file; paper normalization the released numbers."""
    from firepred.training.config import LossConfig

    root, _ = data_root
    expected = reference(root)
    window_config = WindowConfig.read(root)
    config = with_data_statistics(RunConfig("tiny3d", data_root=root, loss=LossConfig("dicece", "auto")),
                                  window_config, root / "train")
    assert list(config.normalization_stats[0]) == expected[NORMALIZATION]["mean"]
    counts = count_labels(root / "train")
    assert config.loss.ce_weight == (1.0, counts.background / counts.fire)
    assert config.checkpoint_settings(None)["normalization"]["std"] == expected[NORMALIZATION]["std"]
    dataset = FireData(config, root).dataset("val", augment=False)
    assert dataset.mean == config.normalization_stats[0]

    paper = with_data_statistics(RunConfig("tiny3d", data_root=root, normalization="paper"), window_config,
                                 root / "train")
    assert paper.normalization_stats == (tuple(NORMALIZATION_MEAN), tuple(NORMALIZATION_STD))
    assert paper.wandb_config()["normalization"] == "paper"
    tested = RunConfig("tiny3d", data_root=root, test_only=True)
    assert with_data_statistics(tested, window_config, root / "train") == tested  # -test uses the checkpoint's


def test_default_class_weights_come_from_window_config(data_root, tmp_path):
    """With no --ce-weight, Dice + CE counts its weights from the file; counts without fire fall back to the paper's."""
    root, _ = data_root
    default = RunConfig("tiny3d", data_root=root)
    assert default.loss.name == "dicece" and default.needs_label_counts  # "auto" is the default
    counts = count_labels(root / "train")
    filled = with_data_statistics(default, WindowConfig.read(root), root / "train")
    assert filled.loss.ce_weight == (1.0, counts.background / counts.fire)
    assert "ce_weight" not in dict(filled.defaults_used)

    no_fire = reference(root)
    no_fire[PIXEL_COUNTS] = {"train": dict(no_fire[PIXEL_COUNTS]["train"], fire=0, background_per_fire=None)}
    target = view(root, tmp_path / "no_fire", no_fire)
    fallback = with_data_statistics(RunConfig("tiny3d", data_root=target), WindowConfig.read(target), target / "train")
    assert fallback.loss.ce_weight == PAPER_CE_WEIGHT
    assert "no fire pixels" in dict(fallback.defaults_used)["ce_weight"]


def test_test_runs_use_the_checkpoint_normalization():
    """A checkpoint's recorded numbers normalize its test data; older checkpoints fall back to the paper's."""
    assert checkpoint_normalization({"normalization": {"mean": [1.0], "std": [2.0]}}) == ([1.0], [2.0])
    assert checkpoint_normalization({}) is None


def test_rebuild_zip_measures_statistics_once_every_archive_is_there(monkeypatch, tmp_path):
    """rebuild-zip: download all, then pixel counts and normalization (also to the repo), then zip and upload."""
    from firepred.data import windows_master

    calls = []

    def record(name):
        """A stand-in master that only records its call."""
        return lambda *args, **kwargs: calls.append((name, kwargs.get("upload"), kwargs.get("data_root")))

    for name in ("download_windows", "count_pixels", "compute_normalization", "upload_data_zip"):
        monkeypatch.setattr(windows_master, name, record(name))
    windows_master.rebuild_data_zip("token", window_repo="owner/windows-6-1", work_dir=tmp_path)
    assert [name for name, _, _ in calls] == ["download_windows", "count_pixels", "compute_normalization",
                                              "upload_data_zip"]
    assert calls[1][1] is True and calls[2][1] is True  # the repo's window_config.json gets them too
    calls.clear()
    windows_master.rebuild_data_zip("token", window_repo="owner/windows-6-1", work_dir=tmp_path, statistics=False)
    assert [name for name, _, _ in calls] == ["download_windows", "upload_data_zip"]
