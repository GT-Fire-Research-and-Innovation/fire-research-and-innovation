"""Losses, metrics, schedules, RunConfig names and checkpoints behave as documented."""

from pathlib import Path

import pytest
import torch
from torch import nn, optim

from firepred.cli.train import argv_for, config_from_args, parse_args
from firepred.training.checkpoints import CheckpointLoader, CheckpointStore
from firepred.training.config import LossConfig, RunConfig, ScheduleConfig
from firepred.training.objectives import Scorer, TrainingLoss
from firepred.training.tracker import Tracker
from firepred.training.trainer import LearningRateSchedule

scorer = Scorer()

BASE = ["-m", "swinunetr3d", "-mode", "pred", "-b", "8", "-r", "1", "-lr", "0.001", "-nh", "3", "-ed", "36",
        "-ts", "6", "-it", "1"]


def loss(name, **weights):
    """A TrainingLoss for `name` with optional LossConfig weights."""
    return TrainingLoss(LossConfig(name, **weights))


class HostilePayload:
    """Stands in for an arbitrary pickled object; restricted loading must refuse it."""


def masks(fire):
    """(1, 2, H, W) one-hot background/fire masks from a boolean fire map."""
    fire = torch.as_tensor(fire, dtype=torch.bool)[None]
    return torch.stack((~fire, fire), dim=1).float()


def test_empty_prediction_and_empty_target_score_one():
    """Empty prediction and empty target score a perfect 1, like the released sklearn scores."""
    empty = masks([[0, 0], [0, 0]])
    assert scorer.fire_f1(empty, empty).item() == 1.0
    assert scorer.fire_iou(empty, empty).item() == 1.0


def test_f1_and_iou_on_a_known_case():
    """F1 and IoU agree with hand-computed values."""
    prediction = masks([[1, 1], [0, 0]])
    target = masks([[1, 0], [1, 0]])
    assert scorer.fire_f1(prediction, target).item() == 0.5  # 2*1 / (2 + 2)
    assert abs(scorer.fire_iou(prediction, target).item() - 1 / 3) < 1e-7
    assert scorer.background_f1(prediction, target).item() == 0.5


def test_false_positives_and_negatives_on_a_known_case():
    """False alarms and missed fire pixels are counted per window, on the fire channel only."""
    prediction = torch.cat((masks([[1, 1], [0, 0]]), masks([[0, 0], [0, 0]])))
    target = torch.cat((masks([[1, 0], [1, 0]]), masks([[0, 0], [0, 0]])))
    # Window 0: (0, 1) predicted but not burned; (1, 0) burned but not predicted. Window 1: empty.
    assert scorer.fire_false_positives(prediction, target).tolist() == [1.0, 0.0]
    assert scorer.fire_false_negatives(prediction, target).tolist() == [1.0, 0.0]
    # With the one true positive they make IoU's denominator: 1 / (1 + 1 + 1).
    assert abs(scorer.fire_iou(prediction, target)[0].item() - 1 / 3) < 1e-7


def test_dicece_reports_its_two_parts():
    """Dice+CE returns its total and both parts; plain Dice has no parts."""
    logits = torch.randn(2, 2, 8, 8)
    labels = masks(torch.rand(8, 8) > 0.9).expand(2, -1, -1, -1)
    total, parts = loss("dicece")(logits, labels)
    assert set(parts) == {"dice_loss", "ce_loss"}
    assert torch.isclose(total, parts["dice_loss"] + parts["ce_loss"])
    assert loss("dice")(logits, labels)[1] == {}


def test_dicece_puts_the_class_weight_on_cross_entropy_only():
    """Dice+CE is lambda_dice x unweighted Dice + lambda_ce x class-weighted CE (the paper's 1:1 mixture)."""
    torch.manual_seed(3)
    logits = torch.randn(2, 2, 8, 8) * 3
    labels = masks(torch.rand(8, 8) > 0.9).expand(2, -1, -1, -1)
    total, parts = loss("dicece", ce_weight=(1.0, 400.0), lambda_dice=0.5, lambda_ce=2.0)(logits, labels)
    dice = loss("dice")(logits, labels)[0]
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 400.0]))(logits, labels)
    assert torch.isclose(parts["dice_loss"], 0.5 * dice)
    assert torch.isclose(parts["ce_loss"], 2.0 * ce)
    assert torch.isclose(total, 0.5 * dice + 2.0 * ce)
    # With the weight off the Dice term, Dice stays between 0 and lambda_dice.
    assert 0.0 <= parts["dice_loss"].item() <= 0.5


def test_dicefocal_parts_are_lambda_weighted():
    """Dice+focal charts lambda x each term, so its parts add up to the total."""
    logits = torch.randn(2, 2, 8, 8)
    labels = masks(torch.rand(8, 8) > 0.9).expand(2, -1, -1, -1)
    total, parts = loss("dicefocal", lambda_dice=0.5, lambda_focal=2.5)(logits, labels)
    assert torch.isclose(parts["dice_loss"], 0.5 * loss("dice")(logits, labels)[0])
    assert torch.isclose(total, parts["dice_loss"] + parts["focal_loss"])


def test_background_iou_scores_the_background_channel():
    """background_iou is TP / (TP + FP + FN) on the background channel."""
    target = masks([[1, 0], [0, 0]])
    prediction = masks([[1, 1], [0, 0]])
    # Background: predicted {(1,0), (1,1)}, true {(0,1), (1,0), (1,1)} -> 2 / 3.
    assert abs(scorer.background_iou(prediction, target)[0].item() - 2 / 3) < 1e-7


def test_plateau_schedule_halves_after_patience():
    """plateau multiplies the rate by its factor after `patience` epochs without improvement."""
    optimizer = optim.Adam([nn.Parameter(torch.zeros(1))], lr=1e-3)
    schedule = LearningRateSchedule(optimizer, ScheduleConfig("plateau", 0.5, 0, 1e-6), 10)
    for val_loss in (1.0, 2.0):
        schedule.step(val_loss)
    assert optimizer.param_groups[0]["lr"] == 5e-4
    assert LearningRateSchedule(optimizer, ScheduleConfig("none"), 10).scheduler is None


def test_names_never_change():
    """Run, checkpoint and artifact names stay exactly as old runs recorded them."""
    config = RunConfig("swinunetr3d", tessera=True, loss=LossConfig("dicece"), schedule=ScheduleConfig("plateau"))
    assert config.run_tag == "_tessera_dicece_lr-plateau"
    assert config.run_name == \
        "train_pred_swinunetr3d_num_heads_3_hidden_size_36_batchsize_8_tessera_dicece_lr-plateau"
    assert config.checkpoint_template.format(epoch=12) == (
        "model_swinunetr3d_mode_pred_num_heads_3_hidden_size_36_batchsize_8_"
        "checkpoint_epoch_12_nc_172_ts_6_tessera_dicece_lr-plateau.pth"
    )
    assert config.test_checkpoint == Path("saved_models") / config.checkpoint_template.format(epoch=199)
    assert config.artifact_name("abc") == "swinunetr3d-tessera-dicece-lr-plateau-checkpoints-abc"


def test_command_line_round_trips_through_run_config():
    """argv_for(config) parses back to the same RunConfig, so notebooks can launch exactly a config."""
    for extra in ([], ["--tessera", "--loss", "dicefocal", "--focal-alpha", "0.25", "--lr-schedule", "cosine"],
                  ["-test", "--checkpoint", "x.pth", "--model-option", "depth=2", "-nc", "43", "--no-test-after-train"]):
        config = config_from_args(parse_args(BASE + extra))
        assert config_from_args(parse_args(argv_for(config))) == config


def test_run_config_fills_and_checks_settings():
    """The paper's loss per model, loss flags kept without --loss, and -nc checked against the data."""
    assert RunConfig("attunet").loss.name == "dice" and RunConfig("swinunetr3d").loss.name == "dicece"
    config = config_from_args(parse_args(BASE + ["--ce-weight", "1", "300"]))
    assert config.loss.name == "dicece" and config.loss.ce_weight == (1.0, 300.0)
    with pytest.raises(ValueError, match="-nc 43"):
        RunConfig("swinunetr3d", tessera=True, expected_channels=43)


def test_wandb_chart_names_are_stable():
    """Dashboards are built on the chart names, so they change only when what they chart changes."""
    labels = Tracker.chart_labels("dicece")
    assert labels["val_fire_f1"] == "Validation F1 — Next-Day Fire Growth"
    assert labels["train_ce_loss"] == "Training Weighted Cross-Entropy Loss (x lambda_ce) — Next-Day Fire Growth"
    assert labels["val_background_iou"] == "Validation IoU — Background"
    assert labels["periodic_test_background_iou"] == "Test IoU — Background (periodic, current weights)"
    assert "miou" not in labels and "mdice" not in labels
    assert "train_ce_loss" not in Tracker.chart_labels("dice")
    assert labels["val_fire_false_positives"] == "Validation False Positives — Next-Day Fire Growth (pixels per window)"
    assert labels["train_fire_false_negatives"] == "Training False Negatives — Next-Day Fire Growth (pixels per window)"
    for loss_name in ("dice", "dicefocal"):
        assert {"periodic_test_false_positives", "periodic_test_false_negatives"} <= set(Tracker.chart_labels(loss_name))


def test_checkpoints_load_without_running_pickled_code(tmp_path):
    """Restricted loading reads tensors and TESSERA statistics but refuses arbitrary pickled objects."""
    import numpy as np

    read_checkpoint = CheckpointLoader("cpu").read

    plain = tmp_path / "plain.pth"
    torch.save({"epoch": 0, "model_state_dict": {"w": torch.ones(2)}, "tessera_stats": None}, plain)
    assert read_checkpoint(plain)["epoch"] == 0  # weights_only=True suffices

    with_stats = tmp_path / "tessera.pth"
    stats = {"mean": np.arange(3, dtype=np.float32), "std": np.ones(3, dtype=np.float32), "min_coverage": 0.5}
    torch.save({"epoch": 1, "tessera_stats": stats}, with_stats)
    if hasattr(torch.serialization, "safe_globals"):  # torch >= 2.4: numpy arrays allowed
        loaded = read_checkpoint(with_stats)
    else:
        with pytest.raises(RuntimeError, match="--trust-checkpoint"):
            read_checkpoint(with_stats)
        loaded = read_checkpoint(with_stats, trusted=True)
    assert (loaded["tessera_stats"]["mean"] == stats["mean"]).all()

    hostile = tmp_path / "hostile.pth"
    torch.save({"x": HostilePayload()}, hostile)
    with pytest.raises(RuntimeError, match="--trust-checkpoint"):
        read_checkpoint(hostile)


def test_dicefocal_is_weighted_dice_plus_weighted_focal():
    """dicefocal = lambda_dice x Dice + lambda_focal x focal on the fire channel, no cross entropy."""
    from monai.losses import FocalLoss

    torch.manual_seed(3)
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    labels = masks(torch.rand(8, 8) > 0.9).expand(2, -1, -1, -1)
    criterion = loss("dicefocal", lambda_dice=0.5, lambda_focal=2.5, focal_gamma=2.0, focal_alpha=0.25)
    total, parts = criterion(logits, labels)
    assert set(parts) == {"dice_loss", "focal_loss"}  # no cross entropy
    dice, _ = loss("dice")(logits, labels)
    focal = FocalLoss(include_background=False, gamma=2.0, alpha=0.25)(logits, labels)  # fire channel only
    assert torch.isclose(total, 0.5 * dice + 2.5 * focal)
    total.backward()
    assert torch.isfinite(logits.grad).all()
    assert [loss(n).name for n in ("dice", "dicece", "dicefocal")] == ["dice", "dicece", "dicefocal"]


def test_dicefocal_names_and_charts():
    """dicefocal has its own tag, loss record and charts (its Dice chart is lambda x Dice)."""
    assert RunConfig("swinunetr3d", loss=LossConfig("dicefocal")).run_tag == "_dicefocal"
    record = LossConfig("dicefocal", (1.0, 446.7836), 0.5, 1.0, 2.5, 2.0, None).record()
    assert record == {"loss": "dicefocal", "lambda_dice": 0.5, "lambda_focal": 2.5, "focal_gamma": 2.0, "focal_alpha": None}
    assert "lambda_focal" not in LossConfig("dicece").record()
    labels = Tracker.chart_labels("dicefocal")
    assert labels["train_loss"] == "Training Dice + Focal Loss — Next-Day Fire Growth"
    assert labels["val_dice_loss"] == "Validation Dice Loss (x lambda_dice) — Next-Day Fire Growth"
    assert labels["val_focal_loss"] == "Validation Focal Loss (x lambda_focal) — Next-Day Fire Growth"
    assert "train_ce_loss" not in labels


def test_checkpoint_store_keeps_the_best(tmp_path):
    """The store keeps the `keep` lowest validation losses on disk and deletes the ones they displace."""
    model = nn.DataParallel(nn.Conv3d(2, 2, 1))
    optimizer = optim.Adam(model.parameters())
    store = CheckpointStore(tmp_path, "ckpt_{epoch}.pth", {"use_tessera": False}, keep=2)
    saved = [store.offer(epoch, val_loss, model, optimizer) for epoch, val_loss in enumerate((0.9, 0.7, 0.8, 0.95, 0.6))]
    assert [path is not None for path in saved] == [True, True, True, False, True]
    assert sorted(p.name for p in tmp_path.glob("*.pth")) == ["ckpt_2.pth", "ckpt_5.pth"]
    assert store.best.name == "ckpt_5.pth" and [loss for loss, _ in store.ranking()] == [0.6, 0.7]
    assert torch.load(store.best, weights_only=True)["use_tessera"] is False


def test_load_checkpoint_refuses_other_inputs(tmp_path):
    """A checkpoint trained with other inputs (TESSERA, width, temporal merging) is refused."""
    path = tmp_path / "c.pth"
    torch.save({"use_tessera": False, "model_input_channels": 43, "swin_temporal_downsample": False}, path)
    assert CheckpointLoader("cpu", RunConfig("swinunetr3d")).load(path)["model_input_channels"] == 43
    with pytest.raises(ValueError, match="--tessera"):
        CheckpointLoader("cpu", RunConfig("swinunetr3d", tessera=True)).load(path)
    with pytest.raises(ValueError, match="temporal"):
        CheckpointLoader("cpu", RunConfig("swinunetr3d", swin_temporal_downsample=True)).load(path)


def test_test_mode_builds_no_trainer(data_root, tmp_path, monkeypatch):
    """-test loads a checkpoint and scores it; no Trainer, loss or optimizer is ever built."""
    import fake_wandb
    import tiny_model  # noqa: F401  registers "tiny3d"

    from firepred.training import experiment

    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setenv("FAKE_WANDB_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    fake_wandb.CALLS.clear()  # calls from earlier runs in this process are not this run's

    def refuse(*args, **kwargs):
        """Stands in for Trainer: building one in test mode is a bug."""
        raise AssertionError("Trainer built in -test mode")

    monkeypatch.setattr(experiment, "Trainer", refuse)
    root, csv_path = data_root
    config = RunConfig("tiny3d", batch_size=1, data_root=root, test_csv=csv_path, plot_dir=tmp_path / "plots")
    model = experiment.Experiment(config).prepare_model(torch.device("cpu"))
    store = CheckpointStore(tmp_path, config.checkpoint_template, config.checkpoint_settings(None))
    checkpoint = store.offer(0, 1.0, model, optim.Adam(model.parameters()))
    result = experiment.Experiment(
        RunConfig("tiny3d", batch_size=1, data_root=root, test_csv=csv_path, plot_dir=tmp_path / "plots",
                  test_only=True, checkpoint=checkpoint)
    ).run()
    assert result.checkpoint == checkpoint and result.scores.fires == 2
    assert not any(call[0] == "define_metric" for call in fake_wandb.CALLS)


def test_swinunetr3d_variants_share_its_checks(tmp_path):
    """Every swinunetr3d_* variant gets the temporal-downsampling refusal and W&B record."""
    path = tmp_path / "c.pth"
    torch.save({"use_tessera": False, "model_input_channels": 43, "swin_temporal_downsample": True}, path)
    for model in ("swinunetr3d", "swinunetr3d_tpool"):
        config = RunConfig(model)
        assert config.is_swinunetr3d and config.wandb_config()["swin_temporal_downsample"] is False
        with pytest.raises(ValueError, match="temporal"):
            CheckpointLoader("cpu", config).load(path)
    assert not RunConfig("unet3d").is_swinunetr3d and "swin_temporal_downsample" not in RunConfig("unet3d").wandb_config()
    assert CheckpointLoader("cpu", RunConfig("unet3d")).load(path)["swin_temporal_downsample"] is True


def test_loader_without_config_reads_but_refuses_to_check(tmp_path):
    """read() needs no config; load() without one fails with a clear message."""
    path = tmp_path / "c.pth"
    torch.save({"use_tessera": False}, path)
    assert CheckpointLoader("cpu").read(path) == {"use_tessera": False}
    with pytest.raises(ValueError, match="with a config"):
        CheckpointLoader("cpu").load(path)
