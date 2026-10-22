"""Label counts and --ce-weight auto: exact, repeatable, and what the loss actually sees."""

import numpy as np
import pytest

from firepred.cli import train as train_cli
from firepred.data.archive import unzip_data_root, write_fire_archive
from firepred.data.fire_dataset import FireDataset
from firepred.data.stats import count_labels
from firepred.training.config import AUTO_CE_WEIGHT, LossConfig, RunConfig
from firepred.training.flags import argv_for
from firepred.training.objectives import TrainingLoss

REQUIRED = ["-m", "swinunetr3d", "-mode", "pred", "-b", "8", "-r", "1", "-lr", "0.001",
            "-nh", "3", "-ed", "36", "-ts", "6", "-it", "1"]


def totals(counts):
    """The counts without the folder they came from."""
    return counts.fires, counts.windows, counts.background, counts.fire, counts.other


def test_counts_match_the_targets_the_dataset_serves(data_root):
    """Background and fire counts equal the two label channels FireDataset gives the loss, summed."""
    root, _ = data_root
    counts = count_labels(root / "train")
    dataset = FireDataset(root / "train", ts_length=6, target_is_single_day=True, augment=False)
    labels = sum(dataset[index]["labels"].sum(dim=(1, 2)) for index in range(len(dataset)))
    assert (counts.windows, counts.background, counts.fire, counts.other) == (
        len(dataset), int(labels[0]), int(labels[1]), 0)
    assert counts.fires == 2 and counts.fire > 0


def test_counts_are_repeatable_and_ignore_threads_and_unzipping(data_root, tmp_path):
    """Any thread count, a second run and the unzipped copy all give the same whole numbers."""
    root, _ = data_root
    first = count_labels(root / "train", workers=1)
    assert totals(count_labels(root / "train", workers=4)) == totals(first)
    assert count_labels(root / "train", workers=1) == first
    unzipped = unzip_data_root(root, tmp_path / "npy", splits=("train",))
    assert unzipped == tmp_path / "npy"
    assert totals(count_labels(unzipped / "train")) == totals(first)
    assert first.background_per_fire == first.background / first.fire


def test_pixels_of_neither_class_are_counted_apart(tmp_path):
    """NaN and negative targets are neither background nor fire, as in read_window."""
    split = tmp_path / "train"
    split.mkdir()
    label = np.array([[0, 1, 2.5], [np.nan, -1, 0]], dtype=np.float32)
    write_fire_archive(split / "p1.npz", np.zeros((2, 1)), np.stack([label, np.zeros_like(label)]))
    counts = count_labels(split)
    assert totals(counts) == (1, 2, 8, 2, 2)
    assert counts.fire_fraction == 0.2 and counts.background_per_fire == 4.0
    assert "2 neither" in counts.summary()


def test_no_fire_and_no_fires_fail_clearly(tmp_path):
    """A split without fire pixels has no ratio; an empty split is an error."""
    split = tmp_path / "train"
    split.mkdir()
    write_fire_archive(split / "p1.npz", np.zeros((1, 1)), np.zeros((1, 4, 4), dtype=np.float32))
    counts = count_labels(split)
    with pytest.raises(ValueError, match="no fire pixels"):
        counts.background_per_fire
    with pytest.raises(ValueError, match="no fire pixels"):
        LossConfig("dicece", AUTO_CE_WEIGHT).with_label_counts(counts)
    with pytest.raises(RuntimeError, match="No .npz archives"):
        count_labels(tmp_path / "missing")


def test_auto_weights_are_filled_in_and_recorded(data_root):
    """'auto' becomes (1, background / fire) with its counts recorded; given weights stay as they are."""
    root, _ = data_root
    counts = count_labels(root / "train")
    config = RunConfig("swinunetr3d", loss=LossConfig("dicece", AUTO_CE_WEIGHT))
    assert config.needs_label_counts and config.loss.record()["ce_weight"] == AUTO_CE_WEIGHT
    resolved = config.with_label_counts(counts)
    assert not resolved.needs_label_counts
    assert resolved.loss.ce_weight == (1.0, counts.background / counts.fire)
    assert resolved.loss.record()["ce_weight_counts"] == [counts.background, counts.fire]
    assert resolved.run_name == config.run_name and resolved.checkpoint_template == config.checkpoint_template
    given = RunConfig("swinunetr3d", loss=LossConfig("dicece", (1.0, 300.0)))
    assert given.with_label_counts(counts) == given and "ce_weight_counts" not in given.loss.record()
    # Losses without a CE term and -test runs never count.
    assert not RunConfig("swinunetr3d", loss=LossConfig("dice", AUTO_CE_WEIGHT)).needs_label_counts
    assert not RunConfig("swinunetr3d", test_only=True, loss=LossConfig("dicece", AUTO_CE_WEIGHT)).needs_label_counts


def test_unresolved_or_malformed_weights_are_refused():
    """The loss refuses 'auto' that was never counted; LossConfig refuses other words and lengths."""
    with pytest.raises(ValueError, match="counted from the training labels"):
        TrainingLoss(LossConfig("dicece", AUTO_CE_WEIGHT))
    for bad in ("Auto", (1.0,), (1.0, 2.0, 3.0)):
        with pytest.raises(ValueError, match="ce_weight must be"):
            LossConfig("dicece", bad)


def test_ce_weight_flag_takes_two_numbers_or_auto(capsys):
    """--ce-weight auto and --ce-weight B F both parse and round-trip; anything else is a usage error."""
    auto = train_cli.config_from_args(train_cli.parse_args(REQUIRED + ["--ce-weight", "auto"]))
    assert auto.loss.ce_weight == AUTO_CE_WEIGHT
    assert argv_for(auto)[argv_for(auto).index("--ce-weight") + 1] == "auto"
    assert train_cli.config_from_args(train_cli.parse_args(argv_for(auto))) == auto
    given = train_cli.config_from_args(train_cli.parse_args(REQUIRED + ["--ce-weight", "1", "352.5"]))
    assert given.loss.ce_weight == (1.0, 352.5)
    assert train_cli.config_from_args(train_cli.parse_args(argv_for(given))) == given
    default = train_cli.config_from_args(train_cli.parse_args(REQUIRED))
    assert default.loss.ce_weight == (1.0, 446.7836)
    for bad in (["auto", "2"], ["1"], ["1", "x"], ["1", "2", "3"]):
        with pytest.raises(SystemExit):
            train_cli.parse_args(REQUIRED + ["--ce-weight", *bad])
        assert "--ce-weight takes BACKGROUND FIRE or auto" in capsys.readouterr().err
