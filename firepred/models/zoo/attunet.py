"""Attention-U-Net-3D, the paper's Attention U-Net baseline (`-m attunet`); not used by SwinUNETR-3D."""

# ============================================================================
# NOT USED BY SWINUNETR-3D (our current work: next-day fire prediction).
#
# Registered as "attunet" (firepred/models/registry.py); built only for
# `-m attunet`. SwinUNETR-3D runs never build it. It must still import
# cleanly, because the registry imports every file in models/zoo/.
#
# What it does: Attention U-Net, a MONAI copy whose strides can leave the time
# axis alone. It is the paper's Attention-U-Net-3D baseline, built only for
# `-m attunet`.
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

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm

__all__ = ["AttentionUnet"]

from firepred.models.registry import register_model

# Architecture constants of the Attention-U-Net-3D baseline. [released]
CHANNELS = (64, 128, 256, 512, 1024)
STRIDES = (1, 2, 2)  # never downsample the T days


class ConvBlock(nn.Module):
    """Two Conv-BatchNorm-ReLU layers; the first applies `strides`."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        strides: int = 1,
        dropout=0.0,
    ):
        """Build the two convolutions (`dropout` after each normalization)."""
        super().__init__()
        layers = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both convolutions."""
        x_c: torch.Tensor = self.conv(x)
        return x_c


class UpConv(nn.Module):
    """Transposed convolution that undoes one level's downsampling (BatchNorm, ReLU)."""

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size=3, strides=2, dropout=0.0):
        """Build the transposed convolution with `strides` and `kernel_size`."""
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act="relu",
            adn_ordering="NDA",
            norm=Norm.BATCH,
            dropout=dropout,
            is_transposed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample `x` by the level's stride."""
        x_u: torch.Tensor = self.up(x)
        return x_u


# Attention gate: the coarser decoder features (g) decide, per pixel, how much of
# the skip connection (x) to let through, suppressing irrelevant regions.
class AttentionBlock(nn.Module):
    """Attention gate: coarser decoder features `g` weight, per voxel, the skip features `x`."""

    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0):
        """1x1x1 projections of g and x to `f_int` channels and a sigmoid gate `psi`."""
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return x * sigmoid(psi(relu(W_g g + W_x x))): the skip, scaled by the gate."""
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# One U-Net level: down -> deeper levels -> up, then the gated skip is concatenated.
class AttentionLayer(nn.Module):
    """One U-Net level: run the deeper levels, upsample, gate the skip, merge both."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        submodule: nn.Module,
        up_kernel_size=3,
        strides=2,
        dropout=0.0,
    ):
        """Build the gate, the upsampling and the merge convolution around `submodule` (the deeper levels)."""
        super().__init__()
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims, f_g=in_channels, f_l=in_channels, f_int=in_channels // 2
        )
        self.upconv = UpConv(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=in_channels,
            strides=strides,
            kernel_size=up_kernel_size,
        )
        self.merge = Convolution(
            spatial_dims=spatial_dims, in_channels=2 * in_channels, out_channels=in_channels, dropout=dropout
        )
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deeper levels -> upsample -> gate the skip with them -> concatenate -> merge convolution."""
        fromlower = self.upconv(self.submodule(x))
        att = self.attention(g=fromlower, x=x)
        att_m: torch.Tensor = self.merge(torch.cat((att, fromlower), dim=1))
        return att_m


class AttentionUnet(nn.Module):
    """
    Attention Unet based on
    Otkay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        channels (Sequence[int]): sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides (Sequence[int]): stride to use for convolutions.
        kernel_size: convolution kernel size.
        up_kernel_size: convolution kernel size for transposed convolution layers.
        dropout: dropout ratio. Defaults to no dropout.
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
        dropout: float = 0.0,
    ):
        """Build the input block, the nested gated encoder-decoder and the 1x1x1 output convolution."""
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.dropout = dropout

        head = ConvBlock(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=channels[0], dropout=dropout)
        reduce_channels = Convolution(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        )
        self.up_kernel_size = up_kernel_size

        def _create_block(channels: Sequence[int], strides: Sequence[int]) -> nn.Module:
            """Build the nested levels for `channels` recursively, deepest last."""
            if len(channels) > 2:
                subblock = _create_block(channels[1:], strides)
                return AttentionLayer(
                    spatial_dims=spatial_dims,
                    in_channels=channels[0],
                    out_channels=channels[1],
                    submodule=nn.Sequential(
                        ConvBlock(
                            spatial_dims=spatial_dims,
                            in_channels=channels[0],
                            out_channels=channels[1],
                            strides=strides,
                            dropout=self.dropout,
                        ),
                        subblock,
                    ),
                    up_kernel_size=self.up_kernel_size,
                    strides=strides,
                    dropout=dropout,
                )
            else:
                # the next layer is the bottom so stop recursion,
                # create the bottom layer as the subblock for this layer
                return self._get_bottom_layer(channels[0], channels[1], strides)

        encdec = _create_block(self.channels, self.strides)
        # Input convolution, the nested encoder-decoder, and a final 1 x 1 x 1 convolution to the classes.
        self.model = nn.Sequential(head, encdec, reduce_channels)

    def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int) -> nn.Module:
        """The deepest level: one ConvBlock wrapped as an AttentionLayer."""
        return AttentionLayer(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            submodule=ConvBlock(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dropout=self.dropout,
            ),
            up_kernel_size=self.up_kernel_size,
            strides=strides,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) input -> (B, K, T, H, W) logits."""
        x_m: torch.Tensor = self.model(x)
        return x_m


@register_model("attunet")
def build(spec):
    """The paper's Attention-U-Net-3D baseline. [released]"""
    return AttentionUnet(
        spatial_dims=3,
        in_channels=spec.in_channels,
        out_channels=spec.num_classes,
        channels=CHANNELS,
        strides=STRIDES,
    )
