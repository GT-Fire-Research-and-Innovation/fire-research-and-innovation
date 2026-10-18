"""A one-layer model registered as "tiny3d", so end-to-end runs take seconds.

Importing this module registers it; the training CLI then accepts `-m tiny3d`.
"""

from torch import nn

from firepred.models.registry import register_model


class Tiny3d(nn.Module):
    """1x1x1 convolution: (B, C, T, H, W) -> per-day logits (B, K, T, H, W)."""

    def __init__(self, in_channels, num_classes):
        """One pointwise Conv3d from the input channels to the classes."""
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        """Per-day logits; the trainer averages them over the days."""
        return self.conv(x)


@register_model("tiny3d")
def build(spec):
    """Build Tiny3d for a ModelSpec."""
    return Tiny3d(spec.in_channels, spec.num_classes)
