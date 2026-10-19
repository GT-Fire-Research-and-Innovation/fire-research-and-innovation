"""UNETR-3D, the paper's UNETR baseline (`-m unetr3d`); not used by SwinUNETR-3D.

Level 2 (models/zoo). Registered as "unetr3d"; a baseline from the paper,
kept for comparison with SwinUNETR-3D.
"""

# ============================================================================
# NOT USED BY SWINUNETR-3D (our current work: next-day fire prediction).
#
# Registered as "unetr3d" (firepred/models/registry.py); built only for
# `-m unetr3d`. SwinUNETR-3D runs never build it. It must still import
# cleanly, because the registry imports every file in models/zoo/.
#
# What it does: UNETR, a MONAI copy: a Vision Transformer encoder with a
# U-Net-style decoder. It is the paper's UNETR-3D baseline, built only for `-m
# unetr3d`.
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

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import deprecated_arg, ensure_tuple_rep


from firepred.models.registry import register_model

# Architecture constants of the UNETR-3D baseline. [released]
PATCH_SIZE = (1, 16, 16)
KERNEL_SIZE_UP_DOWN = (1, 2, 2)


class UNETR(nn.Module):
    """UNETR-3D: a plain vision transformer encoder with a U-Net-style convolutional decoder.

    The input is cut into patches (1 day x 16 x 16 pixels), so attention spans
    all days and the whole image. Features from transformer layers 3, 6 and 9
    and the last layer are reshaped into volumes and decoded with skips
    (Hatamizadeh et al., UNETR, WACV 2022, via MONAI).
    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        patch_size: tuple | int = (1,16,16),
        kernel_size_up_down: tuple | int = (1, 2, 2),
        pos_embed: str = "conv",
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """Build the transformer encoder and the decoder blocks.

        The paper's settings come from build(): hidden size 384, MLP size 1536,
        patch (1, 16, 16). The decoder's upsampling never touches the days.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            img_size (Sequence[int] | int): (T, H, W) of the input.
            feature_size (int): decoder base feature width.
            hidden_size (int): transformer width.
            mlp_dim (int): transformer MLP width.
            num_heads (int): attention heads.
            patch_size (tuple | int): ViT patch size; (1, 16, 16) splits only space.
            kernel_size_up_down (tuple | int): upsampling kernel and stride of every
                decoder step; (1, 2, 2) restores space and never touches the days.
            pos_embed (str): deprecated name of proj_type.
            proj_type (str): patch embedding type.
            norm_name (tuple | str): decoder normalization.
            conv_block (bool): convolutional blocks in the decoder.
            res_block (bool): residual blocks in the decoder.
            dropout_rate (float): dropout rate.
            spatial_dims (int): 3 for this model.
            qkv_bias (bool): bias on queries, keys and values.
            save_attn (bool): keep the attention maps for inspection.

        Raises:
            ValueError: on an invalid dropout rate or a hidden size not divisible by num_heads.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = patch_size
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        # Encoder: a plain Vision Transformer over (1 x 16 x 16) patches, with no
        # windows and no downsampling; features are taken from 4 depths of it.
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        # Turn the transformer features at depths 3, 6 and 9 (and the raw input) into
        # image-shaped skip connections at four resolutions.
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=(3,3,3),
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=(3,3,3),
            stride=1,
            upsample_kernel_size=kernel_size_up_down,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=(3,3,3),
            stride=1,
            upsample_kernel_size=kernel_size_up_down,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=(3,3,3),
            stride=1,
            upsample_kernel_size=kernel_size_up_down,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        # Decoder: upsample from the last transformer layer, adding each skip on the way up.
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=(3,3,3),
            upsample_kernel_size=kernel_size_up_down,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=(3,3,3),
            upsample_kernel_size=kernel_size_up_down,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=(1,3,3),
            upsample_kernel_size=kernel_size_up_down,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=(3,3,3),
            upsample_kernel_size=kernel_size_up_down,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    # Reshape a token sequence back into a (B, C, T, H/16, W/16) feature volume.
    def proj_feat(self, x):
        """Reshape transformer tokens back into a feature volume.

        The ViT outputs one token per patch. The decoder needs them laid out as a
        (T', H', W') grid with the features as channels.

        Args:
            x (torch.Tensor): tokens of shape (B, N, F).

        Returns:
            torch.Tensor: features of shape (B, F, T', H', W').
        """
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        """Run UNETR-3D on one batch.

        The transformer encodes the patches; features from layers 3, 6 and 9 and
        the last layer become skips, and the decoder upsamples back to full size.
        The output is one map per day.

        Args:
            x_in (torch.Tensor): input of shape (B, C, T, H, W).

        Returns:
            torch.Tensor: per-day logits of shape (B, K, T, H, W).
        """
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)


@register_model("unetr3d")
def build(spec):
    """Build the paper's UNETR-3D baseline for a ModelSpec.

    Patch (1, 16, 16), hidden size 384 and MLP size 1536 [paper]. Option
    unetr_version "v0" builds the ViT-Base sizes (768 / 3072) instead. The
    released code never built this model; the decoder's upsampling kernel is
    now a real parameter.

    Args:
        spec (ModelSpec): what the model is built for.

    Returns:
        UNETR: the new model.
    """
    if spec.options.get("unetr_version", "v1") == "v0":
        hidden_size, mlp_dim = 768, 3072
    else:
        hidden_size, mlp_dim = 384, 1536
    return UNETR(
        in_channels=spec.in_channels,
        out_channels=spec.num_classes,
        img_size=spec.input_size,
        spatial_dims=3,
        norm_name="batch",
        feature_size=16,
        patch_size=PATCH_SIZE,
        kernel_size_up_down=KERNEL_SIZE_UP_DOWN,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
    )
