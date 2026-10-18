"""TESSERA context: a small CNN that learns features from a fine TESSERA map while shrinking it. [ours]

Level 2 (models/blocks). Used by models/zoo/swinunetr3d_tessera.py.

    static (B, 129, 256 k, 256 k)        normalized embedding (128) + validity mask (1)
      patchify   k = f1 f2 ... (prime factors): Conv f1 x f1 stride f1, then one
                 conv stage per further factor, each learning features while shrinking
      refine     residual 3 x 3 block on the VIIRS grid          -> (B, C, 256, 256)
      pyramid    stride-2 convs                                   -> 128, 64, 32, 16, 8
    -> one context map per scale of SwinUNETR-3D

ContextFiLM then conditions a feature map on its context map: features x (1 +
scale) + shift, with scale and shift predicted per pixel and channel and the
same for every day. Its projection starts at zero, so a model starts exactly as
if it had no TESSERA and learns how much context to use.
"""

import torch
from torch import nn

from firepred.config.dataset import TESSERA_CHANNELS


def prime_factors(number):
    """Split a whole number into its prime factors, smallest first.

    The encoder shrinks the map by each factor in turn, so k = 4 becomes two
    halvings and k = 5 one five-fold step. 1 has no factors.

    Args:
        number (int): k, 1 or more.

    Returns:
        list[int]: the factors, e.g. [2, 2] for 4 and [5] for 5.
    """
    factors, divisor = [], 2
    while number > 1:
        while number % divisor == 0:
            factors.append(divisor)
            number //= divisor
        divisor += 1
    return factors


def group_count(channels):
    """Pick a GroupNorm group count that divides the channels.

    GroupNorm normalizes each sample on its own, so it behaves the same when the
    encoder runs one large map at a time. Up to 8 groups are used.

    Args:
        channels (int): the feature channels.

    Returns:
        int: the number of groups.
    """
    return next(groups for groups in (8, 4, 2, 1) if channels % groups == 0)


class ConvNormAct(nn.Sequential):
    """One 2-D convolution, GroupNorm and GELU.

    This is the encoder's basic layer. With stride s it also shrinks the map
    s times.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """Build the convolution, its normalization and the activation.

        The convolution has no bias, because GroupNorm adds its own. Padding keeps
        3 x 3 layers at the same size.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            kernel_size (int): convolution size.
            stride (int): how many times the map shrinks.
            padding (int): zero padding on each side.
        """
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.GELU(),
        )


class ResidualBlock(nn.Module):
    """Two 3 x 3 layers added back onto their input.

    It refines features without changing their size. The skip keeps the
    input's information when the layers learn little.
    """

    def __init__(self, channels):
        """Build the two 3 x 3 layers.

        Both keep the channel count. The second one ends without an activation
        of its own before the sum.

        Args:
            channels (int): feature channels.
        """
        super().__init__()
        self.first = ConvNormAct(channels, channels, 3, padding=1)
        self.second = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        """Refine a feature map and add it to its input.

        The output has the input's shape. The activation comes after the sum.

        Args:
            x (torch.Tensor): (B, C, H, W).

        Returns:
            torch.Tensor: (B, C, H, W).
        """
        return self.activation(x + self.second(self.first(x)))


class TesseraContextEncoder(nn.Module):
    """Learn features from a fine TESSERA map while shrinking it to the VIIRS grid, at several scales.

    Each shrinking step is a learned convolution, so the encoder can pick out
    patterns inside a VIIRS pixel (a road, a field edge, a patch of forest)
    instead of averaging them away. The result is one context map per scale
    the main model works at.

    Attributes:
        patchify (ConvNormAct): the first shrinking step, by the smallest factor of k.
        reduce (torch.nn.ModuleList): one shrinking step per further factor of k.
        refine (ResidualBlock): refines the features on the VIIRS grid.
        pyramid (torch.nn.ModuleList): stride-2 steps for the coarser scales.
    """

    def __init__(self, cells_per_pixel, levels, channels=32, in_channels=TESSERA_CHANNELS):
        """Build the shrinking steps for k cells per pixel and the pyramid.

        k is split into prime factors and the map shrinks by one factor per step;
        k = 1 starts with a 1 x 1 layer. The pyramid adds levels - 1 halvings.

        Args:
            cells_per_pixel (int): k, TESSERA cells per VIIRS pixel side.
            levels (int): context maps to return, the first on the VIIRS grid.
            channels (int): context channels.
            in_channels (int): TESSERA input channels (129).
        """
        super().__init__()
        factors = prime_factors(cells_per_pixel) or [1]
        self.patchify = ConvNormAct(in_channels, channels, factors[0], stride=factors[0])
        self.reduce = nn.ModuleList(ConvNormAct(channels, channels, factor, stride=factor) for factor in factors[1:])
        self.refine = ResidualBlock(channels)
        self.pyramid = nn.ModuleList(ConvNormAct(channels, channels, 3, stride=2, padding=1)
                                     for _ in range(levels - 1))

    def forward(self, static):
        """Turn a batch of TESSERA maps into context maps at every scale.

        The first step runs one sample at a time: a fine map may arrive as
        float16 to save host memory, and only one float32 copy of it is made at
        once. Everything after it is small and runs on the whole batch.

        Args:
            static (torch.Tensor): (B, 129, k H, k W) TESSERA map.

        Returns:
            list[torch.Tensor]: (B, C, H / 2^i, W / 2^i) for i = 0 .. levels - 1.
        """
        dtype = self.patchify[0].weight.dtype
        x = torch.cat([self.patchify(sample.unsqueeze(0).to(dtype)) for sample in static], dim=0)
        for step in self.reduce:
            x = step(x)
        x = self.refine(x)
        maps = [x]
        for step in self.pyramid:
            x = step(x)
            maps.append(x)
        return maps


class ContextFiLM(nn.Module):
    """Condition a feature map on a context map: features x (1 + scale) + shift.

    Scale and shift are predicted from the context for every pixel and channel,
    and are the same for every day, because TESSERA does not change during a
    fire. The projection starts at zero, so the features pass through unchanged
    until training finds the context useful.

    Attributes:
        project (torch.nn.Conv2d): context channels -> scale and shift per feature channel.
    """

    def __init__(self, context_channels, feature_channels):
        """Build the zero-initialized 1 x 1 projection.

        It predicts two values per feature channel. Both start at zero.

        Args:
            context_channels (int): channels of the context map.
            feature_channels (int): channels of the feature map it conditions.
        """
        super().__init__()
        self.project = nn.Conv2d(context_channels, 2 * feature_channels, kernel_size=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, features, context):
        """Scale and shift a feature map by its context.

        The context's height and width must match the features'. The days of the
        features may be any number.

        Args:
            features (torch.Tensor): (B, C, T, H, W).
            context (torch.Tensor): (B, C_context, H, W).

        Returns:
            torch.Tensor: (B, C, T, H, W).
        """
        scale, shift = self.project(context).unsqueeze(2).chunk(2, dim=1)
        return features * (1 + scale) + shift
