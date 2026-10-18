"""The training loss for 2-channel (background, fire) logits.

Level 3 (training). Used by training/run.py and training/loop.py.
"""

import torch
from monai.losses import FocalLoss
from monai.losses.dice import DiceCELoss, DiceLoss
from torch import nn

from firepred.config.training import (
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    LAMBDA_CE,
    LAMBDA_DICE,
    LAMBDA_FOCAL,
    PAPER_CE_WEIGHT,
)


class DiceFocalLoss(nn.Module):
    """`lambda_dice` x sigmoid Dice + `lambda_focal` x focal loss on the fire channel. [ours]

    The Dice part is exactly `--loss dice` (both channels, no class weight).
    Focal loss (Lin et al., ICCV 2017) scales each pixel's binary cross entropy
    by (1 - p_t)^gamma, so confidently right pixels (most of the background)
    contribute almost nothing and hard pixels near the fire front dominate.
    It runs on channel 1 only with sigmoid probabilities, the same
    probabilities the 0.5 threshold uses. (Not MONAI's DiceFocalLoss, which
    applies focal to every channel.)
    """

    def __init__(self, dice, focal, lambda_dice, lambda_focal):
        super().__init__()
        self.dice = dice
        self.focal = focal
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal

    def forward(self, outputs, labels):
        return compute_loss(self, outputs, labels)[0]


def build_criterion(
    loss_name,
    ce_weight=PAPER_CE_WEIGHT,
    lambda_dice=LAMBDA_DICE,
    lambda_ce=LAMBDA_CE,
    lambda_focal=LAMBDA_FOCAL,
    focal_gamma=FOCAL_GAMMA,
    focal_alpha=FOCAL_ALPHA,
):
    """Training loss for 2-channel (background, fire) logits.

    dice:   sigmoid Dice loss, as in the released TS-SatFire code. [released]
    dicece: the paper's prediction loss, MONAI DiceCELoss with `lambda_dice`
            Dice + `lambda_ce` cross entropy (1:1 by default). [paper] MONAI
            applies the `weight` parameter to both terms: per-class Dice
            weights and CrossEntropyLoss class weights (the one-hot target is
            used as class probabilities), the same in MONAI 1.3.2 and later.
            The Dice term uses sigmoid probabilities while cross entropy uses
            softmax over the two channels (docs/KNOWN_ISSUES.md).
    dicefocal: `lambda_dice` x the `dice` loss + `lambda_focal` x focal loss
            with `focal_gamma` and `focal_alpha` on the fire channel; no
            cross entropy and no class weight. [ours]
    """
    if loss_name == "dice":
        return DiceLoss(include_background=True, reduction="mean", sigmoid=True)
    if loss_name == "dicece":
        return DiceCELoss(
            include_background=True,
            sigmoid=True,
            reduction="mean",
            weight=torch.tensor(ce_weight, dtype=torch.float32),
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    if loss_name == "dicefocal":
        return DiceFocalLoss(
            build_criterion("dice"),
            FocalLoss(include_background=False, gamma=focal_gamma, alpha=focal_alpha, reduction="mean"),
            lambda_dice,
            lambda_focal,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


def loss_name_of(criterion):
    """The --loss name a criterion was built from."""
    if isinstance(criterion, DiceFocalLoss):
        return "dicefocal"
    return "dicece" if isinstance(criterion, DiceCELoss) else "dice"


def compute_loss(criterion, outputs, labels):
    """Return (loss, {component name: value}) so components can be logged."""
    if isinstance(criterion, DiceFocalLoss):
        dice = criterion.dice(outputs, labels)
        focal = criterion.focal(outputs, labels)
        loss = criterion.lambda_dice * dice + criterion.lambda_focal * focal
        return loss, {"dice_loss": dice.detach(), "focal_loss": focal.detach()}
    if isinstance(criterion, DiceCELoss):
        dice = criterion.dice(outputs, labels)
        ce = criterion.ce(outputs, labels)
        loss = criterion.lambda_dice * dice + criterion.lambda_ce * ce
        return loss, {"dice_loss": dice.detach(), "ce_loss": ce.detach()}
    return criterion(outputs, labels), {}
