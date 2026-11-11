"""Daily archives (format version 3): each day stored once, windows sliced when read."""

import dataclasses
import json
import sys
import zipfile

import numpy as np
import pytest

from firepred.data.archive import (
    convert_to_daily,
    days_from_windows,
    fire_name,
    find_fire_sources,
    index_fires,
    is_daily_archive,
    open_fire,
    read_daily_fire,
    unzip_data_root,
    write_fire_archive,
)

STATIC = (17, 18, 19, 21)


def daily_fire(rng, num_days, size=8, channels=27):
    """Input days (channels, num_days, size, size) with static terrain channels, and a target map per day."""
    days = rng.normal(20.0, 10.0, size=(channels, num_days, size, size)).astype(np.float32)
    for channel in STATIC:
        days[channel] = days[channel, :1]
    days[6] = np.maximum.accumulate(days[6], axis=0)  # a running maximum, like night I4
    targets = rng.random((num_days + 1, size, size)) < 0.05  # targets[d]: newly burned on day d
    return days, targets


def windows_of(days, targets, length, interval):
    """Cut windows the way windows.build_fire_windows does: one every interval days while the target day exists."""
    n = days.shape[1] + 1  # days 0 .. n-2 are inputs, day n-1 is the last target
    starts = [start for start in range(0, n, interval) if start + length < n]
    shape = days.shape[2:]
    data = np.stack([days[:, s:s + length] for s in starts]) if starts else np.zeros((0, 27, length, *shape), np.float32)
    labels = (np.stack([targets[s + length] for s in starts]).astype(np.float32) if starts
              else np.zeros((0, *shape), np.float32))
    return data, labels


def bits(array):
    """An array's dtype, shape and bytes, so equal means bit-identical."""
    array = np.asarray(array)
    return array.dtype.str, array.shape, array.tobytes()


def v2_archive(path, days, targets, length=6, interval=1, label_sel=0):
    """Write a format-2 archive of the windows cut from days and targets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data, labels = windows_of(days, targets, length, interval)
    return write_fire_archive(path, data, labels, label_sel=label_sel)


def test_every_window_setting_reads_back_bit_identical(tmp_path):
    """One daily archive gives the exact windows of its own settings and of any other whose targets it holds."""
    days, targets = daily_fire(np.random.default_rng(0), 14)
    source = v2_archive(tmp_path / "v2" / "p1.npz", days, targets)
    daily = tmp_path / "p1.npz"
    assert convert_to_daily(source, daily, 6, 1, block_days=4, compresslevel=1) == 9
    assert is_daily_archive(daily) and not is_daily_archive(source)
    for length, interval in ((6, 1), (6, 2), (7, 1), (8, 3), (14, 1), (15, 1)):
        data, labels = windows_of(days, targets, length, interval)
        with open_fire(daily, length, interval) as archive:
            assert int(archive["num_windows"]) == len(data)
            for window in range(len(data)):
                assert bits(archive[f"data_{window:03d}"]) == bits(data[window])
                assert bits(archive[f"label_{window:03d}"]) == bits(labels[window])
    with open_fire(daily) as archive:  # no settings: the archive's own (6, 1)
        assert int(archive["format_version"]) == 3 and int(archive["label_sel"]) == 0
        assert int(archive["num_windows"]) == 9
        with pytest.raises(KeyError):
            archive["data_009"]


def test_window_settings_without_their_target_days_are_refused(tmp_path):
    """Windows shorter than the stored first target day, or on days the source skipped, cannot be read."""
    days, targets = daily_fire(np.random.default_rng(1), 14)
    daily = tmp_path / "p1.npz"
    convert_to_daily(v2_archive(tmp_path / "v2" / "p1.npz", days, targets), daily, 6, 1, compresslevel=1)
    with pytest.raises(ValueError, match="lacks target days"):
        open_fire(daily, 5, 1)
    every_other = tmp_path / "p2.npz"
    convert_to_daily(v2_archive(tmp_path / "v2" / "p2.npz", days, targets, 6, 2), every_other, 6, 2, compresslevel=1)
    with open_fire(every_other) as archive:
        assert int(archive["num_windows"]) == 5
    with pytest.raises(ValueError, match="lacks target days"):
        open_fire(every_other, 6, 1)
    with pytest.raises(ValueError, match="only daily archives"):
        open_fire(tmp_path / "v2" / "p1.npz", 6, 1)


def test_static_channels_are_stored_once_and_days_read_back(tmp_path):
    """Channels equal on every day get one member; the days and targets read back whole."""
    days, targets = daily_fire(np.random.default_rng(2), 10)
    daily = tmp_path / "p1.npz"
    convert_to_daily(v2_archive(tmp_path / "v2" / "p1.npz", days, targets), daily, 6, 1, block_days=4,
                     compresslevel=1)
    names = zipfile.ZipFile(daily).namelist()
    for channel in STATIC:
        assert f"static_c{channel}.npy" in names and not any(n.startswith(f"days_c{channel}_") for n in names)
    assert {"days_c00_000.npy", "days_c00_004.npy", "days_c00_008.npy"} <= set(names)
    fire = read_daily_fire(daily)
    assert bits(fire.days) == bits(days)
    assert fire.target_days.tolist() == list(range(6, 11))
    assert bits(fire.targets) == bits(targets[6:11])


def test_build_fire_windows_gives_the_same_windows_as_any_daily_view(tmp_path):
    """Windows built from GeoTIFFs for any length/stride equal the views of one converted archive."""
    pytest.importorskip("rasterio")
    import synth

    from firepred.data.windows import build_fire_windows

    local_map = synth.fire_geotiffs(tmp_path / "geotiffs", n_days=8)
    built = build_fire_windows("fire", local_map, length=3, interval=1, label_sel=0)
    source = write_fire_archive(tmp_path / "v2.npz", built.inputs, built.targets, label_sel=0)
    daily = tmp_path / "p1.npz"
    convert_to_daily(source, daily, 3, 1, compresslevel=1)
    for length, interval in ((3, 1), (3, 2), (4, 1), (6, 1), (6, 3), (8, 1)):
        reference = build_fire_windows("fire", local_map, length=length, interval=interval, label_sel=0)
        with open_fire(daily, length, interval) as archive:
            assert int(archive["num_windows"]) == reference.count
            for window in range(reference.count):
                assert bits(archive[f"data_{window:03d}"]) == bits(reference.inputs[window])
                assert bits(archive[f"label_{window:03d}"]) == bits(reference.targets[window])


def test_windows_that_are_not_one_frame_per_day_are_refused():
    """A window whose copy of a day differs, a non-0/1 target, or an interval that skips days stops the conversion."""
    days, targets = daily_fire(np.random.default_rng(3), 10)
    data, labels = windows_of(days, targets, 6, 1)
    changed = data.copy()
    changed[1, 4, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="not cut from one frame per day"):
        days_from_windows(len(changed), lambda k: (changed[k], labels[k]), 6, 1)
    half = labels.copy()
    half[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="0/1"):
        days_from_windows(len(data), lambda k: (data[k], half[k]), 6, 1)
    with pytest.raises(ValueError, match="leaves days in no window"):
        days_from_windows(2, lambda k: (data[k], labels[k]), 2, 3)


def test_fire_without_windows_converts(tmp_path):
    """A fire too short for one window becomes a daily archive with no windows, which indexing skips."""
    source = write_fire_archive(tmp_path / "v2.npz", np.zeros((0, 27, 6, 8, 8), np.float32),
                                np.zeros((0, 8, 8), np.float32))
    daily = tmp_path / "p1.npz"
    assert convert_to_daily(source, daily, 6, 1, compresslevel=1) == 0
    assert index_fires([daily]) == []


def make_roots(tmp_path, fires):
    """A format-2 data root of full-size (256 x 256) fires and the daily root converted from it."""
    from firepred.data.windows_master import convert_daily

    rng = np.random.default_rng(4)
    v2 = tmp_path / "v2"
    for split, names in fires.items():
        for name, num_days in names:
            days, targets = daily_fire(rng, num_days, size=256)
            days[21] = rng.integers(0, 18, size=(256, 256)).astype(np.float32)  # land cover classes
            v2_archive(v2 / split / f"{name}.npz", days, targets)
    (v2 / "window_config.json").write_text(json.dumps({"ts_length": 6, "ts_interval": 1, "archive_format_version": 2,
                                                       "pixel_counts": {"train": {"fires": ["kept"]}}}))
    report = convert_daily(data_root=v2, dest_root=tmp_path / "v3", splits=tuple(fires), compresslevel=1,
                           work_dir=tmp_path)
    return v2, tmp_path / "v3", report


def test_daily_data_root_trains_like_the_window_archives(tmp_path):
    """Indexing, pixel counts, normalization and FireDataset items are identical on both roots."""
    torch = pytest.importorskip("torch")
    from firepred.data.fire_dataset import FireDataset
    from firepred.data.stats import count_labels, split_normalization

    v2, v3, report = make_roots(tmp_path, {"train": [("p1", 9), ("p2", 7)]})
    assert len(report.converted) == 2 and sum(row[1] for row in report.converted) == 6
    config = json.loads((v3 / "window_config.json").read_text())
    assert config["archive_format_version"] == 3 and config["pixel_counts"] == {"train": {"fires": ["kept"]}}

    names = lambda index: [(fire_name(path), window) for path, window in index]  # noqa: E731
    assert names(index_fires(find_fire_sources(v2 / "train"))) == names(index_fires(find_fire_sources(v3 / "train")))
    same_root = lambda a, b: dataclasses.replace(a, split_dir=b.split_dir) == b  # noqa: E731
    assert same_root(count_labels(v2 / "train"), count_labels(v3 / "train"))
    assert same_root(split_normalization(v2 / "train"), split_normalization(v3 / "train"))
    old = FireDataset(v2 / "train", ts_length=6, target_is_single_day=True)
    new = FireDataset(v3 / "train", ts_length=6, target_is_single_day=True)
    assert len(old) == len(new) == 6
    for item in range(len(old)):
        a, b = old[item], new[item]
        assert a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)
    assert unzip_data_root(v3, tmp_path / "npy") == v3  # nothing to unzip


def test_daily_root_fixes_no_window(tmp_path):
    """A daily root's window_config.json pins no -ts/-it; a format-2 root's still does."""
    from firepred.data.stats import WindowConfig

    v2, v3, _ = make_roots(tmp_path, {"train": [("p1", 9)]})
    config = json.loads((v3 / "window_config.json").read_text())
    assert "ts_length" not in config and "ts_interval" not in config
    assert config["notes"]["converted_from_windows"] == {"ts_length": 6, "ts_interval": 1}
    assert WindowConfig.read(v3).check_windows(10, 3).is_daily
    with pytest.raises(ValueError, match="-ts 10 does not match"):
        WindowConfig.read(v2).check_windows(10, 3)


def test_training_cuts_daily_archives_at_the_runs_windows(tmp_path):
    """FireDataset cuts a daily archive at its -ts/-it, the same windows open_fire gives; missing days stop it."""
    torch = pytest.importorskip("torch")
    from firepred.data.fire_dataset import FireDataset

    _, v3, _ = make_roots(tmp_path, {"train": [("p1", 12), ("p2", 7)]})
    daily = v3 / "train" / "p1.npz"
    for length, interval in ((7, 1), (6, 2), (8, 3)):
        dataset = FireDataset(v3 / "train", ts_length=length, ts_interval=interval, target_is_single_day=True)
        with open_fire(daily, length, interval) as archive:
            windows = int(archive["num_windows"])
            first = torch.from_numpy(archive["data_000"])
        assert [w for path, w in dataset.index if path == daily] == list(range(windows))
        assert dataset[0]["data"].shape[1] == length and first.shape[1] == length
    # p2 holds 7 input days and targets for days 6 and 7: one 7-day window (target day 7), two 6-day ones.
    sevens = FireDataset(v3 / "train", ts_length=7, target_is_single_day=True).index
    assert [w for path, w in sevens if path.name == "p2.npz"] == [0]
    with pytest.raises(ValueError, match="cannot give -ts 5"):
        FireDataset(v3 / "train", ts_length=5, target_is_single_day=True)  # converted from 6-day windows


def test_convert_daily_resumes_and_refuses_daily_sources(tmp_path):
    """A second run skips converted fires; a daily root cannot be converted again."""
    from firepred.data.windows_master import convert_daily

    v2, v3, _ = make_roots(tmp_path, {"train": [("p1", 7)]})
    again = convert_daily(data_root=v2, dest_root=v3, splits=("train",), compresslevel=1, work_dir=tmp_path)
    assert again.converted == () and again.present == ("train/p1",)
    with pytest.raises(ValueError, match="already holds daily archives"):
        convert_daily(data_root=v3, dest_root=tmp_path / "v4", splits=("train",), work_dir=tmp_path)
    with pytest.raises(ValueError, match="records ts_length = 6"):
        convert_daily(data_root=v2, dest_root=tmp_path / "v5", length=8, splits=("train",), work_dir=tmp_path)


def test_daily_repo_converts_uploads_and_stages_without_a_zip(tmp_path, monkeypatch):
    """convert_daily streams a window repo into a daily repo; stage downloads its archives one by one."""
    import fake_hub

    from firepred.data import windows_master

    fake_hub.ROOT = str(tmp_path / "hub")
    fake_hub.CALLS.clear()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    source = fake_hub._repo("u/windows")
    fires = {"train": "p900001", "val": "p20568194", "test": "p900002"}
    rng = np.random.default_rng(5)
    for split, name in fires.items():
        days, targets = daily_fire(rng, 8)
        v2_archive(source / split / f"{name}.npz", days, targets)
    (source / "window_config.json").write_text(json.dumps({"ts_length": 6, "ts_interval": 1}))
    raw = fake_hub._repo("u/raw")
    raw.mkdir(parents=True)
    header = "Id,area,lat,lon,start_date,end_date"
    (raw / "us_fire_2020_out_new.csv").write_text(f"{header}\n900001,1,37,-122,2020-08-01,2020-08-20\n")
    (raw / "us_fire_2017_out_new.csv").write_text(f"{header}\n20568194,1,37,-122,2017-08-01,2017-08-20\n")
    (raw / "us_fire_2021_out_new.csv").write_text(f"{header},label_sel\n900002,1,37,-122,2021-08-01,2021-08-20,0\n")

    report = windows_master.convert_daily(window_repo="u/windows", dest_root=tmp_path / "daily", dest_repo="u/daily",
                                          compresslevel=1, batch_size=2, work_dir=tmp_path)
    assert len(report.converted) == 3
    assert sorted(report.uploaded) == sorted([f"{s}/{n}.npz" for s, n in fires.items()] + ["window_config.json"])
    daily_repo = fake_hub._repo("u/daily")
    assert json.loads((daily_repo / "window_config.json").read_text())["dataset"] == "u/daily"
    assert not (daily_repo / "data.zip").exists()

    work = windows_master.workspace(tmp_path / "work", window_repo="u/daily")
    staged = windows_master.stage_training_data(work, window_repo="u/daily", src_repo="u/raw")
    assert staged.data_root == work.extract_dir.resolve()
    for split, name in fires.items():
        assert is_daily_archive(work.extract_dir / split / f"{name}.npz")
    assert not any(call[0] == "hf_hub_download" and "data.zip" in call[1] for call in fake_hub.CALLS)
