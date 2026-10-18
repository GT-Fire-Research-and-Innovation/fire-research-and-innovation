"""What training optimizes and how predictions are scored: losses and metrics.

Level 3 (training). Used by training/trainer.py, training/evaluator.py and
cli/check.py.

Loss: build_criterion(LossConfig) returns a TrainingLoss, whose forward gives
(total, {part name: value}) so the parts can be logged.

Scores: predictions are logits -> sigmoid -> threshold 0.5 (build_post_transform).
All per-sample scores treat "empty prediction and empty target" as a perfect
1, matching the released code's sklearn scores (zero_division=1).
"""

from dataclasses import dataclass, field

import torch
from monai.losses import FocalLoss
from monai.losses.dice import DiceCELoss, DiceLoss
from monai.metrics import compute_dice, compute_iou
from monai.transforms import Activations, AsDiscrete, Compose
from torch import nn

from firepred.config.training import PREDICTION_THRESHOLD


# ----- Loss ----- #


class TrainingLoss(nn.Module):
    """One of the three losses, returning its total and its logged parts.

    dice:      sigmoid Dice loss, as in the released TS-SatFire code. [released]
               No parts.
    dicece:    the paper's prediction loss, MONAI DiceCELoss: lambda_dice x Dice
               + lambda_ce x cross entropy (1:1 by default). [paper] MONAI applies
               the class weight to both terms; the Dice term uses sigmoid
               probabilities while cross entropy uses softmax over the two
               channels (docs/KNOWN_ISSUES.md). Parts: dice_loss, ce_loss.
    dicefocal: lambda_dice x the `dice` loss + lambda_focal x focal loss on the
               fire channel with sigmoid probabilities (Lin et al., ICCV 2017;
               not MONAI's DiceFocalLoss, which applies focal to every
               channel). No cross entropy, no class weight. [ours]
               Parts: dice_loss, focal_loss.
    """

    def __init__(self, config):
        """Build the MONAI terms for a LossConfig."""
        super().__init__()
        self.name = config.name
        if config.name == "dice":
            self.dice = DiceLoss(include_background=True, reduction="mean", sigmoid=True)
        elif config.name == "dicece":
            # The terms live inside one DiceCELoss (as MONAI builds them), so
            # its weights and reductions are exactly MONAI's.
            self.dicece = DiceCELoss(
                include_background=True,
                sigmoid=True,
                reduction="mean",
                weight=torch.tensor(config.ce_weight, dtype=torch.float32),
                lambda_dice=config.lambda_dice,
                lambda_ce=config.lambda_ce,
            )
        elif config.name == "dicefocal":
            self.dice = DiceLoss(include_background=True, reduction="mean", sigmoid=True)
            self.focal = FocalLoss(
                include_background=False, gamma=config.focal_gamma, alpha=config.focal_alpha, reduction="mean"
            )
            self.lambda_dice = config.lambda_dice
            self.lambda_focal = config.lambda_focal
        else:
            raise ValueError(f"Unsupported loss: {config.name}")

    def forward(self, outputs, labels):
        """(total loss, {part: detached value}) for (B, 2, H, W) logits and one-hot labels."""
        if self.name == "dice":
            return self.dice(outputs, labels), {}
        if self.name == "dicece":
            dice = self.dicece.dice(outputs, labels)
            ce = self.dicece.ce(outputs, labels)
            total = self.dicece.lambda_dice * dice + self.dicece.lambda_ce * ce
            return total, {"dice_loss": dice.detach(), "ce_loss": ce.detach()}
        dice = self.dice(outputs, labels)
        focal = self.focal(outputs, labels)
        total = self.lambda_dice * dice + self.lambda_focal * focal
        return total, {"dice_loss": dice.detach(), "focal_loss": focal.detach()}


def build_criterion(config):
    """The TrainingLoss for a LossConfig (training/config.py)."""
    return TrainingLoss(config)


# ----- Scores ----- #


@dataclass(frozen=True)
class EpochScores:
    """One epoch's per-window means on one split ("train" or "val").

    components holds the loss parts with the split prefix (e.g. "train_dice_loss");
    miou / mdice (MONAI mean IoU / Dice over both classes) exist for validation only.
    """

    split: str
    loss: float
    components: dict = field(default_factory=dict)
    fire_f1: float = 0.0
    background_f1: float = 0.0
    fire_iou: float = 0.0
    miou: float = None
    mdice: float = None

    def as_dict(self):
        """The scores as a plain dict (validation adds miou and mdice after the components)."""
        scores = {"loss": self.loss, "components": dict(self.components)}
        if self.split == "val":
            scores.update(miou=self.miou, mdice=self.mdice)
        scores.update(fire_f1=self.fire_f1, background_f1=self.background_f1, fire_iou=self.fire_iou)
        return scores


def build_post_transform():
    """Logits -> binary masks: sigmoid, then threshold."""
    return Compose(
        [Activations(sigmoid=True), AsDiscrete(threshold=PREDICTION_THRESHOLD)]
    )


def validation_iou(predictions, labels):
    """MONAI mean IoU over both classes (background included), per window: (B, 2) scores.

    Stateless: nothing accumulates between calls. Averaging the background
    class in makes it not the paper's number; the paper's fire-class scores
    are fire_only_f1 / fire_only_iou.
    """
    return compute_iou(predictions, labels, include_background=True, ignore_empty=False)


def validation_dice(predictions, labels):
    """MONAI Dice over both classes (background included), per window: (B, 2) scores."""
    return compute_dice(predictions, labels, include_background=True, ignore_empty=False)


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
