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
    """SwinUNETR-3D whose per-day maps are combined with learned day weights.

    The plain model averages its per-day maps equally. This variant learns one
    weight per input day (a softmax, so they stay positive and sum to 1), so it
    can rely more on recent days.
    """

    def __init__(self, backbone, ts_length):
        """Wrap a SwinUNETR-3D and add one learned weight per input day.

        The weights start equal (all logits 0). Training therefore begins exactly
        like the plain average.

        Args:
            backbone (nn.Module): a SwinUNETR-3D.
            ts_length (int): input days per window.
        """
        super().__init__()
        self.backbone = backbone
        self.day_logits = nn.Parameter(torch.zeros(ts_length))

    def forward(self, x):                                         # (B, C, T, H, W)
        """Run the backbone and combine its per-day maps with the learned weights.

        The result is one map, not one per day. The weights are a softmax over
        the days.

        Args:
            x (torch.Tensor): input of shape (B, C, T, H, W).

        Returns:
            torch.Tensor: logits of shape (B, K, H, W).
        """
        per_day = self.backbone(x)                                # (B, K, T, H, W) one map per input day
        weights = torch.softmax(self.day_logits, dim=0)           # (T,) positive, sums to 1
        return torch.einsum("bkthw,t->bkhw", per_day, weights)    # (B, K, H, W) weighted average


@register_model("swinunetr3d_tpool")
def build(spec):
    """Build the temporally pooled SwinUNETR-3D for a ModelSpec.

    The backbone is exactly swinunetr3d's. Only the day weights are added.

    Args:
        spec (ModelSpec): what the model is built for.

    Returns:
        TemporalPooledSwinUNETR: the new model.
    """
    return TemporalPooledSwinUNETR(build_swinunetr3d(spec), spec.ts_length)
