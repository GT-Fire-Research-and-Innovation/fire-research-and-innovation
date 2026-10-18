"""What training optimizes and how predictions are scored: losses and metrics.

Level 3 (training). Used by training/trainer.py, training/evaluator.py and
cli/check.py.

Loss: TrainingLoss(LossConfig), whose forward gives (total, {part name: value})
so the parts can be logged.

Scores: a Scorer turns logits into masks (sigmoid, then threshold 0.5) and
scores them. All per-sample scores treat "empty prediction and empty target"
as a perfect 1, matching the released code's sklearn scores (zero_division=1).
"""

from dataclasses import dataclass, field

import torch
from monai.data import decollate_batch
from monai.losses import FocalLoss
from monai.losses.dice import DiceCELoss, DiceLoss
from monai.metrics import compute_dice, compute_iou
from monai.transforms import Activations, AsDiscrete, Compose
from torch import nn

from firepred.config.training import PREDICTION_THRESHOLD


# ----- Loss ----- #


class TrainingLoss(nn.Module):
    """The training loss for two-channel (background, fire) logits, with its logged parts.

    Three losses are available: Dice [released], Dice plus cross entropy
    [paper], and Dice plus focal loss on the fire channel [ours]. Calling it
    gives the total to minimize and its parts, which are charted separately.
    """

    def __init__(self, config):
        """Build the MONAI loss terms for a LossConfig.

        Dice+CE keeps MONAI's DiceCELoss intact, so its weights and reductions are
        exactly MONAI's. Dice+focal builds its two terms itself.

        Args:
            config (LossConfig): which loss, with its weights.

        Raises:
            ValueError: on an unknown loss name.
        """
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
        """Compute the loss and its parts for one batch.

        The total is what training minimizes. The parts are detached copies for
        the charts, e.g. {"dice_loss": ..., "ce_loss": ...}.

        Args:
            outputs (torch.Tensor): logits of shape (B, 2, H, W).
            labels (torch.Tensor): one-hot targets of shape (B, 2, H, W).

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: (total loss, parts).
        """
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


# ----- Scores ----- #


@dataclass(frozen=True)
class EpochScores:
    """Hold one epoch's per-window means on one split.

    The trainer returns one for training and one for validation, and the
    tracker charts them. Typed fields replace string-keyed dicts, so a typo
    fails at once.

    Attributes:
        split (str): "train" or "val".
        loss (float): mean loss per window.
        components (dict[str, float]): loss parts with the split prefix, e.g. {"val_dice_loss": ...}.
        fire_f1 (float): mean fire-class F1.
        background_f1 (float): mean background F1.
        fire_iou (float): mean fire-class IoU.
        miou (float | None): validation only: MONAI mean IoU over both classes.
        mdice (float | None): validation only: MONAI mean Dice over both classes.
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
        """Give the scores as a plain dict, in the order the old code returned them.

        The golden tests compare this. Validation adds miou and mdice after the
        components.

        Returns:
            dict: the scores.
        """
        scores = {"loss": self.loss, "components": dict(self.components)}
        if self.split == "val":
            scores.update(miou=self.miou, mdice=self.mdice)
        scores.update(fire_f1=self.fire_f1, background_f1=self.background_f1, fire_iou=self.fire_iou)
        return scores


class Scorer:
    """Turn logits into true/false masks and score them against the targets.

    The trainer and the evaluator both score with one Scorer, so they agree on
    the threshold and the formulas. Every score is computed per window, and
    nothing accumulates between calls.

    Attributes:
        post_transform (monai.transforms.Compose): sigmoid, then the threshold.
    """

    def __init__(self, threshold=PREDICTION_THRESHOLD):
        """Build the logits-to-mask transform: a sigmoid, then the threshold.

        A pixel is predicted burned when its probability is above the
        threshold. The paper uses 0.5.

        Args:
            threshold (float): the probability threshold.
        """
        self.post_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=threshold)])

    def predictions(self, outputs):
        """Turn a batch of logits into true/false masks.

        Each window is passed through the sigmoid and the threshold on its own.
        Scores are computed on the result.

        Args:
            outputs (torch.Tensor): logits of shape (B, 2, H, W).

        Returns:
            torch.Tensor: binary masks of shape (B, 2, H, W).
        """
        return torch.stack([self.post_transform(item) for item in decollate_batch(outputs)])

    @staticmethod
    def mean_iou(predictions, labels):
        """Compute MONAI's IoU over both classes for every window of a batch.

        Because the background class is averaged in, this is not the paper's
        number. The paper's fire-class scores are fire_f1 and fire_iou.

        Args:
            predictions (torch.Tensor): binary masks of shape (B, 2, H, W).
            labels (torch.Tensor): one-hot targets of shape (B, 2, H, W).

        Returns:
            torch.Tensor: shape (B, 2), the IoU per window and class.
        """
        return compute_iou(predictions, labels, include_background=True, ignore_empty=False)

    @staticmethod
    def mean_dice(predictions, labels):
        """Compute MONAI's Dice over both classes for every window of a batch.

        Like mean_iou, it averages in the background class. It is logged during
        validation only.

        Args:
            predictions (torch.Tensor): binary masks of shape (B, 2, H, W).
            labels (torch.Tensor): one-hot targets of shape (B, 2, H, W).

        Returns:
            torch.Tensor: shape (B, 2), the Dice per window and class.
        """
        return compute_dice(predictions, labels, include_background=True, ignore_empty=False)

    @staticmethod
    def channel_f1(predictions, labels, channel):
        """Compute the F1 score of one class for every window of a batch.

        F1 = 2 TP / (predicted + true). An empty prediction with an empty target
        scores 1, matching the released code's scores.

        Args:
            predictions (torch.Tensor): binary masks of shape (B, 2, H, W).
            labels (torch.Tensor): one-hot targets of shape (B, 2, H, W).
            channel (int): 0 for background, 1 for fire.

        Returns:
            torch.Tensor: shape (B,), the F1 per window.
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

    def fire_f1(self, predictions, labels):
        """Compute the fire-class F1 for every window of a batch.

        This is the paper's F1: how well the newly burned pixels are found. An
        empty prediction with an empty target scores 1.

        Args:
            predictions (torch.Tensor): binary masks of shape (B, 2, H, W).
            labels (torch.Tensor): one-hot targets of shape (B, 2, H, W).

        Returns:
            torch.Tensor: shape (B,).
        """
        return self.channel_f1(predictions, labels, channel=1)

    def background_f1(self, predictions, labels):
        """Compute the background-class F1 for every window of a batch.

        It shows how well unburned pixels are kept unburned. It is logged, not
        optimized.

        Args:
            predictions (torch.Tensor): binary masks of shape (B, 2, H, W).
            labels (torch.Tensor): one-hot targets of shape (B, 2, H, W).

        Returns:
            torch.Tensor: shape (B,).
        """
        return self.channel_f1(predictions, labels, channel=0)

    @staticmethod
    def fire_iou(predictions, labels):
        """Compute the fire-class IoU for every window of a batch (the paper's Equation 2).

        IoU = TP / (TP + FP + FN). An empty prediction with an empty target scores
        1, like the released code. It is logged, never trained on.

        Args:
            predictions (torch.Tensor): binary masks of shape (B, 2, H, W).
            labels (torch.Tensor): one-hot targets of shape (B, 2, H, W).

        Returns:
            torch.Tensor: shape (B,).
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
