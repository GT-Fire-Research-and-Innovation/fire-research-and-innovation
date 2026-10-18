"""U-Net-3D, the paper's U-Net baseline (`-m unet3d`); not used by SwinUNETR-3D."""

# ============================================================================
# NOT USED BY SWINUNETR-3D (our current work: next-day fire prediction).
#
# Registered as "unet3d" (firepred/models/registry.py); built only for
# `-m unet3d`. SwinUNETR-3D runs never build it. It must still import
# cleanly, because the registry imports every file in models/zoo/.
#
# What it does: U-Net, a MONAI copy whose strides can leave the time axis
# alone. It is the paper's U-Net-3D baseline, built only for `-m unet3d`.
#
# From the original TS-SatFire code (https://github.com/zhaoyutim/TS-SatFire).
# ============================================================================

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Important changes made: Adjusted convolutional stride to be non-uniform across dimensions,
# but identical across blocks. 

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection

__all__ = ["UNet", "Unet"]


from firepred.models.registry import register_model

# Architecture constants of the U-Net-3D baseline. [released]
CHANNELS = (64, 128, 256, 512, 1024)
STRIDES = (1, 2, 2)  # never downsample the T days


class UNet(nn.Module):
    """U-Net in 3-D with residual units: the paper's U-Net-3D baseline.

    The encoder halves the image level by level with strided convolutions, the
    decoder doubles it back, and skip connections join matching levels.
    Strides never touch the days, so the output keeps one map per day. Copied
    from MONAI's UNet so the strides can leave time alone.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        """Build the encoder-decoder from channel widths and strides.

        The arguments are MONAI UNet's. The levels are created recursively, top first.

        Args:
            spatial_dims (int): 3 for this model.
            in_channels (int): input channels.
            out_channels (int): output channels.
            channels (Sequence[int]): channels per level, top first (at least 2).
            strides (Sequence[int]): downsampling factor between levels (len(channels) - 1).
            kernel_size (Sequence[int] | int): convolution kernel size (odd).
            up_kernel_size (Sequence[int] | int): upsampling kernel size (odd).
            num_res_units (int): residual units per level (0 = plain convolutions).
            act (tuple | str): activation type and arguments.
            norm (tuple | str): normalization type and arguments.
            dropout (float): dropout rate.
            bias (bool): bias terms in the convolutions.
            adn_ordering (str): order of activation (A), normalization (N), dropout (D).

        Raises:
            ValueError: on inconsistent channels and strides.
        """
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        #delta = len(strides) - (len(channels) - 1)
        #if delta < 0:
        #    raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        #if delta > 0:
        #    warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        # Build the U-Net recursively: each level is down layer -> (deeper levels) ->
        # up layer, wrapped so the level's input is concatenated with its output (skip).
        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """Build the U-Net from the bottom up, recursively.

            Each call builds one level: its down path, a skip connection around the
            deeper levels and its up path. The recursion stops at the bottom block.

            Args:
                inc (int): input channels of this level.
                outc (int): output channels of this level.
                channels (Sequence[int]): channels from this level down.
                strides (Sequence[int]): downsampling factors from this level down.
                is_top (bool): True for the top level.

            Returns:
                nn.Module: this level with all deeper ones inside it.
            """
            c = channels[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides, False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, strides, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, strides, is_top)  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        # channels (64, 128, 256, 512, 1024) with strides (1, 2, 2): space halves at each
        # level, the T days never do.
        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """Join one level's down path, the deeper levels and its up path.

        The skip connection concatenates the down path's output with the deeper
        levels' result. The up path then decodes the joined features.

        Args:
            down_path (nn.Module): this level's encoder half.
            up_path (nn.Module): this level's decoder half.
            subblock (nn.Module): the deeper levels.

        Returns:
            nn.Sequential: down_path, SkipConnection(subblock), up_path.
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """Build one level's encoder half, which may downsample.

        With residual units it is a ResidualUnit, otherwise one convolution. Its
        output goes down to the next level and across the skip connection.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            strides (int | Sequence[int]): downsampling factor.
            is_top (bool): True for the top level.

        Returns:
            nn.Module: the encoder half.
        """
        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build the bottleneck that links the encoder to the decoder.

        It is the deepest level and does not downsample. It is a down layer with stride 1.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.

        Returns:
            nn.Module: the bottleneck.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """Build one level's decoder half, which upsamples.

        A transposed convolution upsamples the joined features; with residual units
        a ResidualUnit follows. The top level outputs the logits.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            strides (int | Sequence[int]): upsampling factor.
            is_top (bool): True for the top level (no final activation).

        Returns:
            nn.Module: the decoder half.
        """
        conv: Convolution | nn.Sequential

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    # (B, C, T, H, W) -> logits (B, K, T, H, W); the trainer averages over T.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the U-Net on one batch.

        The days are kept throughout, so the output is one map per day.
        The trainer averages them.

        Args:
            x (torch.Tensor): input of shape (B, C, T, H, W).

        Returns:
            torch.Tensor: per-day logits of shape (B, K, T, H, W).
        """
        x = self.model(x)
        return x


Unet = UNet


@register_model("unet3d")
def build(spec):
    """Build the paper's U-Net-3D baseline for a ModelSpec.

    Channels and strides are the released code's, with strides that never
    downsample the days [released]. Heads, feature size and options are ignored.

    Args:
        spec (ModelSpec): what the model is built for.

    Returns:
        UNet: the new model.
    """
    return UNet(
        spatial_dims=3,
        in_channels=spec.in_channels,
        out_channels=spec.num_classes,
        channels=CHANNELS,
        strides=STRIDES,
    )
