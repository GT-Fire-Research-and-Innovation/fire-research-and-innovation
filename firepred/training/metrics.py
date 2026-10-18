"""Scores on thresholded predictions: per-sample F1 and IoU, plus MONAI's averaged metrics.

Level 3 (training). Used by training/loop.py and training/evaluation.py.

Predictions are logits -> sigmoid -> threshold 0.5 (build_post_transform).
All per-sample scores treat "empty prediction and empty target" as a perfect
1, matching the released code's sklearn scores (zero_division=1).
"""

import torch
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose

from firepred.config.training import PREDICTION_THRESHOLD


def build_post_transform():
    """Logits -> binary masks: sigmoid, then threshold."""
    return Compose(
        [Activations(sigmoid=True), AsDiscrete(threshold=PREDICTION_THRESHOLD)]
    )


def build_validation_metrics():
    """MONAI MeanIoU and DiceMetric over both classes (background included).

    These average the background class in, so they are not the paper's
    numbers; the paper's fire-class scores are fire_only_f1/fire_only_iou.
    """
    mean_iou = MeanIoU(
        include_background=True,
        reduction="mean",
        ignore_empty=False,
    )
    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean",
        ignore_empty=False,
    )
    return mean_iou, dice_metric


def _channel_f1(predictions, labels, channel):
    """Return per-sample F1 for one channel of binary (B,2,H,W) masks.

    Empty prediction and empty target score 1, matching existing test F1
    and DiceMetric(ignore_empty=False). Channels are evaluated independently.
    """
    predicted = predictions[:, channel].bool().flatten(start_dim=1)
    target = (labels[:, channel] > 0).flatten(start_dim=1)
    true_positive = (predicted & target).sum(dim=1).float()
    denominator = predicted.sum(dim=1).float() + target.sum(dim=1).float()
    return torch.where(
        denominator > 0,
        2.0 * true_positive / denominator.clamp_min(1.0),
        torch.ones_like(denominator),
    )


def fire_only_f1(predictions, labels):
    """Per-sample next-day fire-growth F1 from thresholded channel 1."""
    return _channel_f1(predictions, labels, channel=1)


def background_f1(predictions, labels):
    """Per-sample background F1 from thresholded channel 0."""
    return _channel_f1(predictions, labels, channel=0)


def fire_only_iou(predictions, labels):
    """Per-sample next-day fire-growth IoU, the paper's Equation 2.

    IoU = TP / (TP + FP + FN) on thresholded channel 1. Empty prediction and
    empty target score 1, like the official code's test, which used
    sklearn's jaccard_score(zero_division=1).
    Logged only; never part of the training loss.
    """
    predicted = predictions[:, 1].bool().flatten(start_dim=1)
    target = (labels[:, 1] > 0).flatten(start_dim=1)
    intersection = (predicted & target).sum(dim=1).float()
    union = (predicted | target).sum(dim=1).float()
    return torch.where(
        union > 0,
        intersection / union.clamp_min(1.0),
        torch.ones_like(union),
    )
