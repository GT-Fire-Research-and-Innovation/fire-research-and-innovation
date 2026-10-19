"""Attention-U-Net-3D, the paper's Attention U-Net baseline (`-m attunet`); not used by SwinUNETR-3D.

Level 2 (models/zoo). Registered as "attunet"; a baseline from the paper,
kept for comparison with SwinUNETR-3D.
"""

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
    """Two convolution layers, each followed by batch normalization and ReLU.

    This is one level's feature extractor. The first convolution applies the
    level's stride, so it can halve the image.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        strides: int = 1,
        dropout=0.0,
    ):
        """Build the two convolutions.

        Dropout comes after each normalization. Only the first one is strided.

        Args:
            spatial_dims (int): 3 for this model.
            in_channels (int): input channels.
            out_channels (int): output channels.
            kernel_size (int): convolution kernel size.
            strides (int | Sequence[int]): stride of the first convolution.
            dropout (float): dropout rate.
        """
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
        """Apply both convolutions.

        The output has out_channels channels. Its size is divided by the stride.

        Args:
            x (torch.Tensor): input features.

        Returns:
            torch.Tensor: the convolved features.
        """
        x_c: torch.Tensor = self.conv(x)
        return x_c


class UpConv(nn.Module):
    """A transposed convolution that undoes one level's downsampling.

    It doubles the image size again (never the days), followed by batch
    normalization and ReLU. It is the decoder's upsampling step.
    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size=3, strides=2, dropout=0.0):
        """Build the transposed convolution.

        The stride decides how much the image grows. Normalization and ReLU follow.

        Args:
            spatial_dims (int): 3 for this model.
            in_channels (int): input channels.
            out_channels (int): output channels.
            kernel_size (int): kernel size.
            strides (int | Sequence[int]): upsampling factor per axis.
            dropout (float): dropout rate.
        """
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
        """Upsample the features by the level's stride.

        This is the decoder going back up one level. The channels change to out_channels.

        Args:
            x (torch.Tensor): coarser features.

        Returns:
            torch.Tensor: upsampled features.
        """
        x_u: torch.Tensor = self.up(x)
        return x_u


# Attention gate: the coarser decoder features (g) decide, per pixel, how much of
# the skip connection (x) to let through, suppressing irrelevant regions.
class AttentionBlock(nn.Module):
    """An attention gate that decides how much of each skip-connection voxel to pass on.

    The coarser decoder features (g) and the skip features (x) are projected,
    added, and squeezed into one weight per voxel between 0 and 1. The skip
    features are multiplied by that weight, which suppresses irrelevant regions
    (Oktay et al., Attention U-Net, 2018).
    """

    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0):
        """Build the two projections and the gate.

        The projections are 1x1x1 convolutions to f_int channels. The gate ends in
        a sigmoid, giving weights between 0 and 1.

        Args:
            spatial_dims (int): 3 for this model.
            f_int (int): channels inside the gate.
            f_g (int): channels of the decoder features.
            f_l (int): channels of the skip features.
            dropout (float): dropout rate.
        """
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
        """Weight the skip features by the gate computed from both inputs.

        The gate is sigmoid(psi(relu(W_g g + W_x x))). Voxels the decoder finds
        irrelevant get weights near 0.

        Args:
            g (torch.Tensor): decoder features.
            x (torch.Tensor): skip features of the same size.

        Returns:
            torch.Tensor: x scaled by the gate.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# One U-Net level: down -> deeper levels -> up, then the gated skip is concatenated.
class AttentionLayer(nn.Module):
    """One level of the Attention U-Net: deeper levels, upsampling, a gated skip and a merge.

    The levels are nested: each holds the next deeper one as its submodule.
    Its output combines the upsampled deeper result with the gated skip.
    """

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
        """Build the gate, the upsampling and the merge convolution around the deeper levels.

        The submodule is everything below this level. The merge convolution mixes
        the gated skip with the upsampled result.

        Args:
            spatial_dims (int): 3 for this model.
            in_channels (int): channels at this level.
            out_channels (int): channels at the next deeper level.
            submodule (nn.Module): the deeper levels.
            up_kernel_size (int): upsampling kernel size.
            strides (int | Sequence[int]): downsampling factor between this level and the next.
            dropout (float): dropout rate.
        """
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
        """Run the deeper levels, upsample, gate the skip with the result and merge.

        The skip is this level's own input. The output has this level's channels.

        Args:
            x (torch.Tensor): this level's input features.

        Returns:
            torch.Tensor: the merged features.
        """
        fromlower = self.upconv(self.submodule(x))
        att = self.attention(g=fromlower, x=x)
        att_m: torch.Tensor = self.merge(torch.cat((att, fromlower), dim=1))
        return att_m


class AttentionUnet(nn.Module):
    """Attention U-Net in 3-D: a U-Net whose skip connections pass through attention gates.

    It is the paper's Attention-U-Net-3D baseline (Oktay et al., Attention U-Net,
    2018, via MONAI). Strides never downsample the days, so the output keeps one
    map per day.
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
        """Build the input block, the nested gated encoder-decoder and the output convolution.

        The levels are created recursively from the channel list, top first.
        Each level gets an attention gate on its skip connection.

        Args:
            spatial_dims (int): 3 for this model.
            in_channels (int): input channels.
            out_channels (int): output channels.
            channels (Sequence[int]): channels per level, top first.
            strides (Sequence[int]): downsampling factor between levels.
            kernel_size (int): convolution kernel size.
            up_kernel_size (int): upsampling kernel size.
            dropout (float): dropout rate.
        """
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
            """Build the nested levels for a channel list, recursively.

            Each call builds one level and recurses for the rest. The last two
            channels make the bottom level.

            Args:
                channels (Sequence[int]): channels from this level down.
                strides (Sequence[int]): downsampling factor between levels.

            Returns:
                nn.Module: this level with all deeper ones inside it.
            """
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
        """Build the deepest level: one ConvBlock wrapped as an AttentionLayer.

        It has no deeper levels below it. It still upsamples and gates like the others.

        Args:
            in_channels (int): channels at this level.
            out_channels (int): channels inside the bottom block.
            strides (int | Sequence[int]): downsampling factor.

        Returns:
            AttentionLayer: the bottom level.
        """
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
        """Run the whole Attention U-Net on one batch.

        The days are kept throughout, so the output is one map per day.
        The trainer averages them.

        Args:
            x (torch.Tensor): input of shape (B, C, T, H, W).

        Returns:
            torch.Tensor: per-day logits of shape (B, K, T, H, W).
        """
        x_m: torch.Tensor = self.model(x)
        return x_m


@register_model("attunet")
def build(spec):
    """Build the paper's Attention-U-Net-3D baseline for a ModelSpec.

    Channels are (64, 128, 256, 512, 1024) and strides (1, 2, 2), so the days are
    never downsampled [released]. Heads, feature size and options are ignored.

    Args:
        spec (ModelSpec): what the model is built for.

    Returns:
        AttentionUnet: the new model.
    """
    return AttentionUnet(
        spatial_dims=3,
        in_channels=spec.in_channels,
        out_channels=spec.num_classes,
        channels=CHANNELS,
        strides=STRIDES,
    )
