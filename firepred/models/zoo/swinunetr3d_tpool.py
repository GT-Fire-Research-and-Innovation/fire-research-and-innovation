"""swinunetr3d_tpool: SwinUNETR-3D with learned weights for its six day-slices.

Level 2 (models/zoo). [ours] Pathway 2a in docs/swinunetr3d/10-pathways.md.

SwinUNETR-3D outputs one logit map per input day, (B, K, T, H, W), and the
trainer averages them with equal weights (registry.forward_model). This model
learns the T weights instead: softmax(day_logits), with day_logits starting at
0, so training starts from the plain average and moves away only if that
lowers the loss. The output is already (B, K, H, W).

Train it with `-m swinunetr3d_tpool`; every SwinUNETR-3D flag applies. Its
checkpoint keys are SwinUNETR-3D's with a "backbone." prefix plus "day_logits".
"""

import torch
from torch import nn

from firepred.models.registry import register_model
from firepred.models.zoo.swinunetr3d import build as build_swinunetr3d


class TemporalPooledSwinUNETR(nn.Module):
    """SwinUNETR-3D whose per-day maps are averaged with learned day weights (softmax over T)."""

    def __init__(self, backbone, ts_length):
        """Wrap `backbone` and add one learned logit per input day (all zero: a plain mean at start)."""
        super().__init__()
        self.backbone = backbone
        self.day_logits = nn.Parameter(torch.zeros(ts_length))

    def forward(self, x):                                         # (B, C, T, H, W)
        """(B, C, T, H, W) input -> (B, K, H, W): the day-weighted average of the backbone's per-day logits."""
        per_day = self.backbone(x)                                # (B, K, T, H, W) one map per input day
        weights = torch.softmax(self.day_logits, dim=0)           # (T,) positive, sums to 1
        return torch.einsum("bkthw,t->bkhw", per_day, weights)    # (B, K, H, W) weighted average


@register_model("swinunetr3d_tpool")
def build(spec):
    """Build the temporally pooled SwinUNETR-3D for a ModelSpec."""
    return TemporalPooledSwinUNETR(build_swinunetr3d(spec), spec.ts_length)
