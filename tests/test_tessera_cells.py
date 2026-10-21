"""k x k TESSERA cells per VIIRS pixel: build, archive, lazy dataset, stem model, run settings."""

import numpy as np
import pytest
import torch

from firepred.config.dataset import BASE_MODEL_CHANNELS, TESSERA_CHANNELS
from firepred.data.archive import (
    cells_per_pixel_of,
    compute_tessera_stats,
    find_fire_sources,
    iter_tessera_maps,
    load_tessera_maps,
    open_npz,
    read_tessera,
    save_tessera_npz,
)
from firepred.data.fire_dataset import FireDataset, batch_inputs, make_dataloader, validate_dataset_output
from firepred.models.registry import ModelSpec, build_model, forward_model
from firepred.training.config import RunConfig

K = 2  # small enough for quick tests; the code is the same for 5


@pytest.fixture(scope="module")
def fine_root(tmp_path_factory):
    """A synthetic data root whose TESSERA files have K x K cells per VIIRS pixel."""
    import synth

    return synth.data_root(tmp_path_factory.mktemp("fine") / "data", synth.QUICK_FIRES, cells_per_pixel=K)


def test_crop_grid_subdivides_the_same_square():
    """k cells per pixel shrink the cell size by k and keep the crop's corner; k = 1 is unchanged."""
    from rasterio.transform import Affine

    from firepred.data.tessera import crop_grid

    profile = {"crs": "EPSG:32610", "transform": Affine(375.0, 0, 500000.0, 0, -375.0, 4200000.0),
               "height": 512, "width": 512}
    _, plain = crop_grid(profile)
    _, same = crop_grid(profile, cells_per_pixel=1)
    _, fine = crop_grid(profile, cells_per_pixel=5)
    assert plain == same
    assert (fine.a, fine.e) == (75.0, -75.0) and (fine.c, fine.f) == (plain.c, plain.f)
    assert fine * (5 * 256, 5 * 256) == plain * (256, 256)  # the same far corner
    with pytest.raises(ValueError):
        crop_grid(profile, cells_per_pixel=0)


def test_build_fire_tessera_writes_a_finer_map(tmp_path, monkeypatch):
    """build_fire_tessera(cells_per_pixel=K) writes a (128, 256K, 256K) map and records K; k = 1 records nothing."""
    import importlib.metadata

    import synth

    real_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "0.0-fake" if name == "geotessera" else real_version(name))

    from firepred.data.tessera import build_fire_tessera

    local_map = synth.fire_geotiffs(tmp_path / "geotiffs", n_days=2)
    days = {name: path for name, path in local_map.items() if "/VIIRS_Day/" in name}
    fine = build_fire_tessera(synth.FakeGeoTessera(), "1", days, 2020, tmp_path / "fine.npz", cells_per_pixel=K)
    plain = build_fire_tessera(synth.FakeGeoTessera(), "1", days, 2020, tmp_path / "plain.npz")
    assert fine.coverage.shape == (256 * K, 256 * K) and plain.coverage.shape == (256, 256)
    with open_npz(fine.path) as archive:
        assert archive["embedding"].shape == (128, 256 * K, 256 * K) and int(archive["cells_per_pixel"]) == K
    with open_npz(plain.path) as archive:
        assert "cells_per_pixel" not in archive.files


def test_archive_checks_the_cell_count(tmp_path):
    """Maps must be a square multiple of 256; a reader asking for another k is refused."""
    assert cells_per_pixel_of((128, 256, 256)) == 1 and cells_per_pixel_of((512, 512)) == 2
    with pytest.raises(ValueError):
        cells_per_pixel_of((128, 300, 300))
    path = tmp_path / "p1.npz"
    save_tessera_npz(path, np.zeros((128, 512, 512), np.float32), np.ones((512, 512), np.float32), {})
    assert read_tessera(path, cells_per_pixel=2)[0].shape == (128, 512, 512)
    with pytest.raises(ValueError, match="cells per VIIRS pixel"):
        read_tessera(path, cells_per_pixel=1)


def test_streamed_statistics_match_the_kept_ones(fine_root):
    """Statistics from maps read one at a time equal those from maps kept in memory."""
    root, _ = fine_root
    train = find_fire_sources(root / "train")
    kept = compute_tessera_stats(load_tessera_maps(root / "tessera", train, K).values())
    streamed = compute_tessera_stats(iter_tessera_maps(root / "tessera", train, K))
    assert np.array_equal(kept["mean"], streamed["mean"]) and np.array_equal(kept["std"], streamed["std"])


def test_lazy_dataset_returns_the_fine_map_beside_the_data(fine_root):
    """With K cells the dataset reads each map per window (float16) and batch_inputs returns (data, static)."""
    root, _ = fine_root
    stats = compute_tessera_stats(iter_tessera_maps(root / "tessera", find_fire_sources(root / "train"), K))
    lazy = FireDataset(root / "val", ts_length=6, target_is_single_day=True, tessera_dir=root / "tessera",
                       tessera_stats=stats, cells_per_pixel=K)
    kept = FireDataset(root / "val", ts_length=6, target_is_single_day=True, tessera_dir=root / "tessera",
                       tessera_stats=stats, cells_per_pixel=K, lazy_tessera=False)
    assert lazy.lazy_tessera and not lazy.tessera_maps
    static = lazy[0]["static"]
    assert static.dtype == torch.float16 and tuple(static.shape) == (TESSERA_CHANNELS, 256 * K, 256 * K)
    assert torch.allclose(static.float(), kept[0]["static"], atol=1e-2)
    validate_dataset_output(lazy, BASE_MODEL_CHANNELS + TESSERA_CHANNELS, "val")
    batch = next(iter(make_dataloader(lazy, 1, shuffle=False, num_workers=0, pin_memory=False)))
    data, static = batch_inputs(batch, "cpu")
    assert tuple(data.shape) == (1, BASE_MODEL_CHANNELS, 6, 256, 256)
    assert tuple(static.shape) == (1, TESSERA_CHANNELS, 256 * K, 256 * K)


def test_stem_model_compresses_the_map_and_learns():
    """A spec with K cells wraps the backbone; the pair input gives (B, 2, H, W) and trains the stem."""
    import tiny_model  # noqa: F401  registers "tiny3d"

    spec = ModelSpec(BASE_MODEL_CHANNELS + TESSERA_CHANNELS, 2, 3, 3, 12, image_size=(16, 16),
                     tessera_cells_per_pixel=K)
    model = build_model("tiny3d", spec)
    assert type(model).__name__ == "TesseraStemModel"
    assert type(build_model("tiny3d", ModelSpec(BASE_MODEL_CHANNELS, 2, 3, 3, 12))).__name__ == "Tiny3d"
    data = torch.randn(2, BASE_MODEL_CHANNELS, 3, 16, 16)
    static = torch.randn(2, TESSERA_CHANNELS, 16 * K, 16 * K).half()
    logits = forward_model(model, (data, static))
    assert tuple(logits.shape) == (2, 2, 16, 16)
    logits.sum().backward()
    assert model.stem.embed.weight.grad is not None and torch.isfinite(model.stem.embed.weight.grad).all()


def test_mask_is_pooled_to_the_fraction_of_valid_cells():
    """The stem's last channel is the fraction of valid cells inside each VIIRS pixel."""
    from firepred.models.blocks.tessera_stem import TesseraStem

    static = torch.zeros(1, TESSERA_CHANNELS, 4, 4)
    static[0, -1, :2, :2] = 1.0
    static[0, -1, 2, 2] = 1.0
    mask = TesseraStem(2)(static)[0, -1]
    assert torch.equal(mask, torch.tensor([[1.0, 0.0], [0.0, 0.25]]))


def test_run_settings_follow_the_cell_count(tmp_path):
    """k reaches the model spec, the names and the checkpoint, and only applies with --tessera."""
    from firepred.training.checkpoints import CheckpointLoader

    fine = RunConfig("swinunetr3d", tessera=True, tessera_cells_per_pixel=5)
    plain = RunConfig("swinunetr3d", tessera=True)
    assert fine.model_spec().tessera_cells_per_pixel == 5 and plain.model_spec().tessera_cells_per_pixel == 1
    assert "_tessera5x5" in fine.run_name and "_tessera5x5" in fine.checkpoint_template
    assert plain.run_tag.startswith("_tessera_")
    assert RunConfig("swinunetr3d", tessera_cells_per_pixel=5).model_spec().tessera_cells_per_pixel == 1
    assert fine.checkpoint_settings(None)["tessera_cells_per_pixel"] == 5
    with pytest.raises(ValueError):
        RunConfig("swinunetr3d", tessera_cells_per_pixel=0)
    path = tmp_path / "c.pth"
    torch.save({"use_tessera": True, "model_input_channels": 172, "swin_temporal_downsample": False,
                "tessera_cells_per_pixel": 5}, path)
    assert CheckpointLoader("cpu", fine).load(path)["tessera_cells_per_pixel"] == 5
    with pytest.raises(ValueError, match="--tessera-cells-per-pixel 5"):
        CheckpointLoader("cpu", plain).load(path)


def test_cli_flag_reaches_the_config():
    """--tessera-cells-per-pixel is a training flag with default 1."""
    from firepred.cli.train import config_from_args, parse_args

    base = ["-m", "swinunetr3d", "-mode", "pred", "-b", "1", "-r", "1", "-lr", "0.001",
            "-nh", "3", "-ed", "36", "-ts", "6", "-it", "1", "--tessera"]
    assert config_from_args(parse_args(base)).tessera_cells_per_pixel == 1
    assert config_from_args(parse_args(base + ["--tessera-cells-per-pixel", "5"])).tessera_cells == 5


def test_training_run_with_fine_tessera(fine_root, tmp_path, monkeypatch):
    """A one-epoch tiny3d run with K x K TESSERA trains, saves a stem checkpoint and tests it."""
    import fake_wandb
    import tiny_model  # noqa: F401  registers "tiny3d"

    from firepred.training.checkpoints import CheckpointLoader
    from firepred.training.experiment import Experiment

    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setenv("FAKE_WANDB_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    root, csv_path = fine_root
    config = RunConfig("tiny3d", batch_size=1, epochs=1, num_workers=0, data_root=root, test_csv=csv_path,
                       plot_dir=tmp_path / "plots", tessera=True, tessera_cells_per_pixel=K, device="cpu")
    result = Experiment(config).run()
    checkpoint = CheckpointLoader("cpu").read(result.checkpoint, trusted=True)  # this run wrote it
    assert checkpoint["tessera_cells_per_pixel"] == K
    assert any(key.startswith("module.stem.embed.") for key in checkpoint["model_state_dict"])
    assert result.scores.fires == 2
