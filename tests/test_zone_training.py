"""Training on fire split zones: one sample per fire of each window, from one forward pass per window."""

import json

import numpy as np
import pytest
import torch

from test_daily import v2_archive
from test_daily_zones import growing_fire

from firepred.config.dataset import ZONE_FORMAT_VERSION
from firepred.data.archive import write_zone_file
from firepred.data.fire_records import FireEvent, ZoneFile
from firepred.data.fire_tracks import split_settings
from firepred.data.fire_zones import fire_sample
from firepred.training.config import LossConfig, RunConfig
from firepred.training.objectives import TrainingLoss, fire_samples


def two_fire_window(size=8):
    """Zones and target owners of one window: fires 1 (left) and 3 (right), one new ignition, plus a fire-less window."""
    zones = np.zeros((2, 2, size, size), dtype=np.int32)
    zones[0, 0, :, :3], zones[0, 0, :, 5:] = 1, 3
    zones[0, 1, 0, 1], zones[0, 1, 2, 6], zones[0, 1, 7, 4] = 1, 3, -1
    zones[1, 1, 4, 4] = -1  # window 1: no fire, one new ignition nothing reaches
    return zones


def label_zone_files(split_dir, zones_dir, ts_length=6):
    """Write a zone file per archive that agrees with its labels: left half fire 1, right half fire 2.

    Window 0 has no fire, so its new pixels are ignitions (-1). Labels of any
    shape work, so the synthetic (random-label) data roots can train on zones.
    """
    zones_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(split_dir.glob("p*.npz")):
        with np.load(path) as archive:
            labels = [archive[f"label_{index:03d}"] > 0 for index in range(int(archive["num_windows"]))]
        halves = np.where(np.arange(labels[0].shape[1]) < labels[0].shape[1] // 2, 1, 2)
        area = np.broadcast_to(halves, labels[0].shape).astype(np.int16)
        zones = np.stack([np.zeros_like(area) if index == 0 else area for index in range(len(labels))])
        owners = np.stack([np.where(label, -1 if index == 0 else area, 0).astype(np.int16)
                           for index, label in enumerate(labels)])
        events = tuple(FireEvent(fire, "ignition", 0, ts_length, ts_length + len(labels), 0) for fire in (1, 2))
        write_zone_file(zones_dir / path.name, ZoneFile(zones, owners, events, split_settings(2.25), -1, ts_length,
                                                        ZONE_FORMAT_VERSION))
    return zones_dir


def zone_root(tmp_path, names=("p1", "p2"), num_days=12):
    """A format-2 training root of full-size growing fires (each with a far second fire) and its zone files."""
    pytest.importorskip("scipy")
    from firepred.data.zones_master import build_zones

    rng = np.random.default_rng(3)
    root = tmp_path / "v2"
    for position, name in enumerate(names):
        days, targets = growing_fire(rng, num_days, size=256)
        # Each fire burns elsewhere in its crop, so no two fires share a target.
        days, targets = np.roll(days, 40 * position, axis=-1), np.roll(targets, 40 * position, axis=-1)
        days[21] = rng.integers(0, 18, size=(256, 256)).astype(np.float32)  # land cover classes
        v2_archive(root / "train" / f"{name}.npz", days, targets)
    (root / "window_config.json").write_text(json.dumps({"ts_length": 6, "ts_interval": 1}))
    build_zones(root, splits=("train",))
    return root


# ----- The loss ----- #


def test_fire_samples_are_fire_sample_for_a_whole_batch():
    """The batch version gives every fire of every window the target and mask of data/fire_zones.fire_sample."""
    zones = two_fire_window()
    rows, targets, masks = fire_samples(torch.from_numpy(zones))
    assert rows.tolist() == [0, 0, 1]
    for sample, fire in ((0, 1), (1, 3)):
        target, mask = fire_sample(zones[0, 0], zones[0, 1], fire)
        assert torch.equal(targets[sample, 1].bool(), torch.from_numpy(target))
        assert torch.equal(targets[sample, 0].bool(), ~torch.from_numpy(target))
        assert torch.equal(masks[sample, 0].bool(), torch.from_numpy(mask))
    # The fire-less window: no target, and only its new ignition is left out.
    assert not targets[2, 1].any() and masks[2, 0].sum() == 63 and masks[2, 0, 4, 4] == 0


@pytest.mark.parametrize("name", ["dice", "dicece"])
def test_one_fire_over_the_whole_crop_gives_the_plain_loss(name):
    """With one fire owning every pixel and every new pixel, the per-fire loss is MONAI's loss and parts."""
    torch.manual_seed(0)
    outputs = torch.randn(3, 2, 16, 16)
    fire = torch.rand(3, 16, 16) > 0.8
    labels = torch.stack((~fire, fire), dim=1).float()
    zones = torch.stack((torch.ones(3, 16, 16, dtype=torch.int32), fire.int()), dim=1)
    criterion = TrainingLoss(LossConfig(name, ce_weight=(1.0, 50.0)))
    plain, plain_parts = criterion(outputs, labels)
    zoned, zoned_parts = criterion(outputs, labels, zones)
    assert torch.allclose(plain, zoned, rtol=1e-6, atol=1e-6)
    assert plain_parts.keys() == zoned_parts.keys()
    for part in plain_parts:
        assert torch.allclose(plain_parts[part], zoned_parts[part], rtol=1e-6, atol=1e-6)


def test_pixels_no_sample_keeps_get_no_gradient():
    """A new ignition no fire reaches teaches nothing; a pixel inside a fire's zone does."""
    zones = torch.from_numpy(two_fire_window())
    outputs = torch.randn(2, 2, 8, 8, requires_grad=True)
    labels = torch.stack((zones[:, 1] == 0, zones[:, 1] != 0), dim=1).float()
    loss, parts = TrainingLoss(LossConfig("dicece", ce_weight=(1.0, 10.0)))(outputs, labels, zones)
    loss.backward()
    assert torch.isfinite(loss) and set(parts) == {"dice_loss", "ce_loss"}
    assert outputs.grad[0, :, 7, 4].abs().sum() == 0 and outputs.grad[1, :, 4, 4].abs().sum() == 0
    assert outputs.grad[0, :, 0, 0].abs().sum() > 0 and outputs.grad[1, :, 0, 0].abs().sum() > 0


def test_each_fire_counts_as_its_own_sample():
    """Two fires in one window weigh like two windows of one fire each: Dice is averaged per sample."""
    zones = torch.from_numpy(two_fire_window())[:1]
    outputs = torch.randn(1, 2, 8, 8)
    criterion = TrainingLoss(LossConfig("dice"))
    rows, targets, masks = fire_samples(zones)
    together, _ = criterion(outputs, None, zones)
    apart = [criterion.masked(outputs[rows[i:i + 1]], targets[i:i + 1], masks[i:i + 1])[0] for i in range(2)]
    assert torch.allclose(together, sum(apart) / 2)


# ----- The data ----- #


def test_items_carry_zone_maps_that_move_with_the_target(tmp_path):
    """Zone maps change nothing else in an item and follow its flips and rotations exactly."""
    from firepred.data.fire_dataset import FireDataset

    root = zone_root(tmp_path)
    plain = FireDataset(root / "train", ts_length=6, augment=True, target_is_single_day=True)
    zoned = FireDataset(root / "train", ts_length=6, augment=True, target_is_single_day=True,
                        zones_dir=root / "zones")
    assert len(plain) == len(zoned) == 14 and zoned.zones.settings.scale_km == 2.25  # 7 windows per fire
    fires_seen = set()
    for item in range(len(zoned)):
        np.random.seed(item)
        before = plain[item]
        np.random.seed(item)
        after = zoned[item]
        assert torch.equal(before["data"], after["data"]) and torch.equal(before["labels"], after["labels"])
        assert "zones" not in before and after["zones"].dtype == torch.int32
        assert tuple(after["zones"].shape) == (2, 256, 256)
        assert torch.equal(after["zones"][1] != 0, after["labels"][1] > 0)  # owners moved with the target
        fires_seen.add(len(torch.unique(after["zones"][0][after["zones"][0] > 0])))
    assert max(fires_seen) >= 2  # the far ignition becomes a second fire in later windows


def test_zone_files_that_do_not_fit_are_refused(tmp_path):
    """A missing file, another window length, mixed distances or another fire's file stop training."""
    from firepred.data.fire_dataset import FireDataset
    from firepred.data.zones_master import build_zones

    root = zone_root(tmp_path)
    zones = root / "zones"
    with pytest.raises(ValueError, match="-ts 7"):
        FireDataset(root / "train", ts_length=7, target_is_single_day=True, zones_dir=zones)
    (zones / "p1.npz").rename(tmp_path / "p1.npz")
    with pytest.raises(RuntimeError, match="no zone file"):
        FireDataset(root / "train", ts_length=6, target_is_single_day=True, zones_dir=zones)
    build_zones(root, zones_root=tmp_path / "zones-3km", splits=("train",), scale_km=3.0)
    (tmp_path / "zones-3km" / "p1.npz").replace(zones / "p1.npz")  # p1 at another distance than p2
    with pytest.raises(ValueError, match=r"scale_km 2.25 \(others 3.0\).*one zones folder per setting"):
        FireDataset(root / "train", ts_length=6, target_is_single_day=True, zones_dir=zones)
    (tmp_path / "p1.npz").replace(zones / "p1.npz")
    (zones / "p1.npz").rename(tmp_path / "swap.npz")
    (zones / "p2.npz").rename(zones / "p1.npz")
    (tmp_path / "swap.npz").rename(zones / "p2.npz")  # each fire now has the other's zones
    dataset = FireDataset(root / "train", ts_length=6, target_is_single_day=True, zones_dir=zones)
    with pytest.raises(ValueError, match="does not match the target"):
        for item in range(len(dataset)):
            dataset[item]


def test_daily_archives_read_every_other_day_use_every_other_zone_window(tmp_path):
    """A daily archive recorded at -it 2 trains on windows 0, 2, 4, ...; its window w is zone window 2 w."""
    pytest.importorskip("scipy")
    from firepred.data.archive import DailyFire, write_daily_archive
    from firepred.data.fire_dataset import FireDataset
    from firepred.data.zones_master import build_zones

    rng = np.random.default_rng(6)
    days, targets = growing_fire(rng, 12, size=256)
    days[21] = rng.integers(0, 18, size=(256, 256)).astype(np.float32)
    root = tmp_path / "v3"
    (root / "train").mkdir(parents=True)
    fire = DailyFire(days=days, target_days=np.arange(6, 13), targets=targets[6:13])
    write_daily_archive(root / "train" / "p1.npz", fire, length=6, interval=2, label_sel=0, compresslevel=1)
    (root / "window_config.json").write_text(json.dumps({"ts_length": 6, "ts_interval": 2,
                                                         "archive_format_version": 3}))
    build_zones(root, splits=("train",))  # daily archives give a zone window every day

    dataset = FireDataset(root / "train", ts_length=6, target_is_single_day=True, zones_dir=root / "zones")
    assert len(dataset) == 4 and dataset.zones.strides == {"p1": 2}
    for item in range(len(dataset)):
        assert torch.equal(dataset[item]["zones"][1] != 0, dataset[item]["labels"][1] > 0)


# ----- The run ----- #


def test_zone_runs_are_named_recorded_and_round_trip():
    """--zones-dir survives argv_for, adds _zones to every name and is recorded; TPUs are refused."""
    from firepred.cli.train import argv_for, config_from_args, parse_args
    from firepred.data.fire_tracks import split_settings as settings_for

    plain = RunConfig(model="swinunetr3d", loss=LossConfig("dicece", ce_weight=(1.0, 2.0)))
    zoned = RunConfig(model="swinunetr3d", loss=LossConfig("dicece", ce_weight=(1.0, 2.0)), zones_dir="zones")
    assert config_from_args(parse_args(argv_for(zoned))).zones_dir.name == "zones"
    assert "--zones-dir" not in argv_for(plain)
    assert zoned.run_name == plain.run_name.replace("_dicece", "_zones_dicece")
    assert "_zones_dicece" in zoned.checkpoint_template and "_zones" not in plain.checkpoint_template
    recorded = zoned.with_zone_settings(settings_for(2.25))
    assert recorded.wandb_config()["zone_settings"]["scale_km"] == 2.25
    assert recorded.checkpoint_settings(None)["zones"]["zones_dir"] == "zones"
    assert "zones_dir" not in plain.wandb_config() and "zones" not in plain.checkpoint_settings(None)
    with pytest.raises(ValueError, match="GPU or the CPU"):
        RunConfig(model="swinunetr3d", zones_dir="zones", device="tpu")
