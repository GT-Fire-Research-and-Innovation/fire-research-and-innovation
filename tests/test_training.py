"""Losses, metrics, schedules and names behave as documented."""

from pathlib import Path

import torch
from torch import nn, optim

from firepred.training import naming
from firepred.training.losses import build_criterion, compute_loss
from firepred.training.metrics import background_f1, fire_only_f1, fire_only_iou
from firepred.training.optim import build_lr_scheduler, step_lr_scheduler
from firepred.training.tracking import build_wandb_metric_labels


class HostilePayload:
    """Stands in for an arbitrary pickled object; restricted loading must refuse it."""


def masks(fire):
    fire = torch.as_tensor(fire, dtype=torch.bool)[None]
    return torch.stack((~fire, fire), dim=1).float()


def test_empty_prediction_and_empty_target_score_one():
    empty = masks([[0, 0], [0, 0]])
    assert fire_only_f1(empty, empty).item() == 1.0
    assert fire_only_iou(empty, empty).item() == 1.0


def test_f1_and_iou_on_a_known_case():
    prediction = masks([[1, 1], [0, 0]])
    target = masks([[1, 0], [1, 0]])
    assert fire_only_f1(prediction, target).item() == 0.5  # 2*1 / (2 + 2)
    assert abs(fire_only_iou(prediction, target).item() - 1 / 3) < 1e-7
    assert background_f1(prediction, target).item() == 0.5


def test_dicece_reports_its_two_parts():
    logits = torch.randn(2, 2, 8, 8)
    labels = masks(torch.rand(8, 8) > 0.9).expand(2, -1, -1, -1)
    loss, parts = compute_loss(build_criterion("dicece"), logits, labels)
    assert set(parts) == {"dice_loss", "ce_loss"}
    assert torch.isclose(loss, parts["dice_loss"] + parts["ce_loss"])
    assert compute_loss(build_criterion("dice"), logits, labels)[1] == {}


def test_plateau_schedule_halves_after_patience():
    optimizer = optim.Adam([nn.Parameter(torch.zeros(1))], lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, "plateau", 10, 0.5, 0, 1e-6)
    for val_loss in (1.0, 2.0):
        step_lr_scheduler(scheduler, val_loss)
    assert optimizer.param_groups[0]["lr"] == 5e-4
    assert build_lr_scheduler(optimizer, "none", 10, 0.5, 0, 1e-6) is None


def test_names_never_change():
    tag = naming.run_tag(True, "dicece", "plateau")
    assert tag == "_tessera_dicece_lr-plateau"
    assert naming.run_name(False, "pred", "swinunetr3d", 3, 36, 8, True, "dicece") == \
        "train_pred_swinunetr3d_num_heads_3_hidden_size_36_batchsize_8_tessera_dicece"
    template = naming.checkpoint_template("swinunetr3d", "pred", 3, 36, 8, 172, 6, tag)
    assert template.format(epoch=12) == (
        "model_swinunetr3d_mode_pred_num_heads_3_hidden_size_36_batchsize_8_"
        "checkpoint_epoch_12_nc_172_ts_6_tessera_dicece_lr-plateau.pth"
    )
    assert naming.default_test_checkpoint(template, 199) == Path("saved_models") / template.format(epoch=199)
    assert naming.artifact_name("swinunetr3d", tag, "abc") == "swinunetr3d-tessera-dicece-lr-plateau-checkpoints-abc"


def test_wandb_chart_names_are_stable():
    labels = build_wandb_metric_labels("dicece")
    assert labels["val_fire_f1"] == "Validation F1 — Next-Day Fire Growth"
    assert labels["train_ce_loss"] == "Training Weighted Cross-Entropy Loss — Next-Day Fire Growth"
    assert "train_ce_loss" not in build_wandb_metric_labels("dice")


def test_checkpoints_load_without_running_pickled_code(tmp_path):
    import numpy as np
    import pytest

    from firepred.training.checkpoints import read_checkpoint

    plain = tmp_path / "plain.pth"
    torch.save({"epoch": 0, "model_state_dict": {"w": torch.ones(2)}, "tessera_stats": None}, plain)
    assert read_checkpoint(plain, "cpu")["epoch"] == 0  # weights_only=True suffices

    with_stats = tmp_path / "tessera.pth"
    stats = {"mean": np.arange(3, dtype=np.float32), "std": np.ones(3, dtype=np.float32), "min_coverage": 0.5}
    torch.save({"epoch": 1, "tessera_stats": stats}, with_stats)
    if hasattr(torch.serialization, "safe_globals"):  # torch >= 2.4: numpy arrays allowed
        loaded = read_checkpoint(with_stats, "cpu")
    else:
        with pytest.raises(RuntimeError, match="--trust-checkpoint"):
            read_checkpoint(with_stats, "cpu")
        loaded = read_checkpoint(with_stats, "cpu", trusted=True)
    assert (loaded["tessera_stats"]["mean"] == stats["mean"]).all()

    hostile = tmp_path / "hostile.pth"
    torch.save({"x": HostilePayload()}, hostile)
    with pytest.raises(RuntimeError, match="--trust-checkpoint"):
        read_checkpoint(hostile, "cpu")


def test_dicefocal_is_weighted_dice_plus_weighted_focal():
    from monai.losses import FocalLoss

    from firepred.training.losses import loss_name_of

    torch.manual_seed(3)
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    labels = masks(torch.rand(8, 8) > 0.9).expand(2, -1, -1, -1)
    criterion = build_criterion("dicefocal", lambda_dice=0.5, lambda_focal=2.5, focal_gamma=2.0, focal_alpha=0.25)
    loss, parts = compute_loss(criterion, logits, labels)
    assert set(parts) == {"dice_loss", "focal_loss"}  # no cross entropy
    dice, _ = compute_loss(build_criterion("dice"), logits, labels)
    focal = FocalLoss(include_background=False, gamma=2.0, alpha=0.25)(logits, labels)  # fire channel only
    assert torch.isclose(loss, 0.5 * dice + 2.5 * focal)
    assert torch.isclose(criterion(logits, labels), loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert [loss_name_of(build_criterion(n)) for n in ("dice", "dicece", "dicefocal")] == ["dice", "dicece", "dicefocal"]


def test_dicefocal_names_and_charts():
    assert naming.run_tag(False, "dicefocal", "none") == "_dicefocal"
    config = naming.loss_config("dicefocal", (1.0, 446.7836), 0.5, 1.0, 2.5, 2.0, None)
    assert config == {"loss": "dicefocal", "lambda_dice": 0.5, "lambda_focal": 2.5, "focal_gamma": 2.0, "focal_alpha": None}
    assert "lambda_focal" not in naming.loss_config("dicece", (1.0, 446.7836), 1.0, 1.0)
    labels = build_wandb_metric_labels("dicefocal")
    assert labels["train_loss"] == "Training Dice + Focal Loss — Next-Day Fire Growth"
    assert labels["val_dice_loss"] == "Validation Dice Loss — Next-Day Fire Growth"  # same chart as a dice run
    assert labels["val_focal_loss"] == "Validation Focal Loss — Next-Day Fire Growth"
    assert "train_ce_loss" not in labels
