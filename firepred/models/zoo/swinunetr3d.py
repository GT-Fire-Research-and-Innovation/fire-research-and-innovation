"""SwinUNETR-3D: a Swin Transformer encoder with a U-Net-style convolutional decoder.

Level 2 (models/zoo). Registered as "swinunetr3d", the main model of this repo.

Input (B, C, T, H, W) -> logits (B, 2, T, H, W). The encoder (blocks/swin_encoder.py)
gives five feature scales; the decoder upsamples back to full resolution,
adding the matching encoder scale at each step (skip connections), and a 1x1x1
convolution gives two logits per pixel per day. forward_model() averages them
over the T days into one next-day map. docs/swinunetr3d/ explains it from scratch.

Attribute names (swinViT, encoder1..10, decoder1..5, out) are checkpoint keys:
renaming any of them breaks loading old checkpoints.

Swin UNETR: Hatamizadeh et al. 2022, https://arxiv.org/abs/2201.01266.
Adapted from MONAI (https://github.com/Project-MONAI/MONAI, Apache-2.0) by the
TS-SatFire authors. [released]
"""

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch.nn as nn
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep
from typing import Sequence, Tuple, Union

from firepred.models.blocks.swin_encoder import SwinTransformer
from firepred.models.registry import register_model

__all__ = ["SwinUNETR"]

# Architecture constants of SwinUNETR-3D as trained on TS-SatFire. [paper]
PATCH_SIZE = (1, 2, 2)      # each token starts as 1 day x 2 x 2 pixels
WINDOW_HW = (4, 4)          # attention windows span all T days x 4 x 4 tokens
DEPTHS = (2, 2, 2, 2)       # Swin blocks per stage


class SwinUNETR(nn.Module):
    '''
    Swin UNETR based on: 'Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>'
    '''

    def __init__(
        self,
        image_size:     Union[Sequence[int], int],
        patch_size:     Union[Sequence[int], int],
        window_size:    Union[Sequence[int], int],
        in_channels:    int,
        out_channels:   int,
        depths:         Sequence[int] = (2, 2, 2, 2),
        num_heads:      Sequence[int] = (3, 6, 12, 24),
        feature_size:   int = 24,
        norm_name:      Union[Tuple, str] = 'batch',
        drop_rate:      float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_version:   str = 'v1',
        normalize:      bool = True,
        use_checkpoint: bool = False,
        spatial_dims:   int = 3,
        temporal_downsample: bool = True
    ) -> None:
        '''
        Args:
            image_size:     dimension of input image.
            patch_size:     dimension of patch.
            window_size:    dimension of window.
            in_channels:    dimension of input channels.
            out_channels:   dimension of output channels.
            feature_size:   dimension of network feature size.
            depths:         number of layers in each stage.
            num_heads:      number of attention heads.
            norm_name:      feature normalization type and arguments.
            drop_rate:      dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: drop path rate.
            attn_version:   version of attention in swin transformer.
            normalize:      normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims:   number of spatial dims.
            temporal_downsample: patch merging also halves the first (temporal) dim when it is even.
        '''

        super().__init__()

        image_size = ensure_tuple_rep(image_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError('spatial dimension should be 2 or 3.')

        # Input checks: the image must divide evenly through five halvings, and the
        # dropout rates must be valid probabilities.
        for m, p in zip(image_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError('input image size (image_size) should be divisible by stage-wise image resolution.')

        if not (0 <= drop_rate <= 1):
            raise ValueError('dropout rate should be between 0 and 1.')

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError('attention dropout rate should be between 0 and 1.')

        if not (0 <= drop_path_rate <= 1):
            raise ValueError('drop path rate should be between 0 and 1.')

        # if feature_size % 4 != 0:
        #     raise ValueError('feature_size should be divisible by 4.')

        if attn_version not in ['v1', 'v2']:
            raise ValueError('attn_version should be v1 or v2.')

        self.normalize = normalize

        # Encoder: the Swin transformer producing five feature scales.
        self.swinViT = SwinTransformer(
            image_size     = image_size,
            in_chans       = in_channels,
            embed_dim      = feature_size,
            window_size    = window_size,
            patch_size     = patch_size,
            depths         = depths,
            num_heads      = num_heads,
            mlp_ratio      = 4.0,
            qkv_bias       = True,
            drop_rate      = drop_rate,
            attn_drop_rate = attn_drop_rate,
            drop_path_rate = drop_path_rate,
            attn_version   = attn_version,
            norm_layer     = nn.LayerNorm,
            use_checkpoint = use_checkpoint,
            spatial_dims   = spatial_dims,
            temporal_downsample = temporal_downsample
        )
        resamples = self.swinViT.resamples

        # Convolutional blocks that refine the raw input (encoder1) and the first
        # three Swin scales (encoder2-4) before they are used as skip connections.
        self.encoder1 = UnetrBasicBlock(
            spatial_dims = spatial_dims,
            in_channels  = in_channels,
            out_channels = feature_size,
            kernel_size  = 3,
            stride       = 1,
            norm_name    = norm_name,
            res_block    = True
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims = spatial_dims,
            in_channels  = feature_size,
            out_channels = feature_size,
            kernel_size  = 3,
            stride       = 1,
            norm_name    = norm_name,
            res_block    = True
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims = spatial_dims,
            in_channels  = 2 * feature_size,
            out_channels = 2 * feature_size,
            kernel_size  = 3,
            stride       = 1,
            norm_name    = norm_name,
            res_block    = True
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims =spatial_dims,
            in_channels  = 4 * feature_size,
            out_channels = 4 * feature_size,
            kernel_size  = 3,
            stride       = 1,
            norm_name    = norm_name,
            res_block    = True
        )

        # Bottleneck: refines the coarsest Swin scale (16 x feature_size channels).
        # Holds about half of all parameters but runs on only 8 x 8 positions.
        self.encoder10 = UnetrBasicBlock(
            spatial_dims = spatial_dims,
            in_channels  = 16 * feature_size,
            out_channels = 16 * feature_size,
            kernel_size  = 3,
            stride       = 1,
            norm_name    = norm_name,
            res_block    = True
        )

        # Decoder: each block upsamples (transposed conv), concatenates the encoder
        # features of the same size (skip connection) and refines them, climbing from
        # 8 x 8 back to 256 x 256.
        self.decoder5 = UnetrUpBlock(
            spatial_dims         = spatial_dims,
            in_channels          = 16 * feature_size,
            out_channels         = 8 * feature_size,
            kernel_size          = 3,
            upsample_kernel_size = resamples[0],
            norm_name            = norm_name,
            res_block            = True
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims         = spatial_dims,
            in_channels          = feature_size * 8,
            out_channels         = feature_size * 4,
            kernel_size          = 3,
            upsample_kernel_size = resamples[1],
            norm_name            = norm_name,
            res_block            = True
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims         = spatial_dims,
            in_channels          = feature_size * 4,
            out_channels         = feature_size * 2,
            kernel_size          = 3,
            upsample_kernel_size = resamples[2],
            norm_name            = norm_name,
            res_block            = True
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims         = spatial_dims,
            in_channels          = feature_size * 2,
            out_channels         = feature_size,
            kernel_size          = 3,
            upsample_kernel_size = resamples[3],
            norm_name            = norm_name,
            res_block            = True
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims         = spatial_dims,
            in_channels          = feature_size,
            out_channels         = feature_size,
            kernel_size          = 3,
            upsample_kernel_size = patch_size,
            norm_name            = norm_name,
            res_block            = True
        )

        # Output head: a 1 x 1 x 1 convolution to 2 logits (background, fire) per pixel and day.
        self.out = nn.Sequential(
            UnetOutBlock(
                spatial_dims = spatial_dims,
                in_channels  = feature_size,
                out_channels = out_channels
            ),
            # nn.Sigmoid()
            # nn.Softmax()
        )


    def forward(self, x_in):
        """(B, C, T, H, W) input -> (B, K, T, H, W) per-day logits: Swin encoder, conv skips, U-Net decoder."""
        # Shapes for the paper setup: B windows, C=43, T=6, 256x256, F=feature_size=36,
        # no temporal downsampling. docs/swinunetr3d/shapes.md lists them all.
        hidden_states_out = self.swinViT(x_in, self.normalize)   # 5 scales: (B, F*2^i, T, 128/2^i, 128/2^i)

        enc0 = self.encoder1(x_in)                                # (B, F,   T, 256, 256) convs on the raw input
        enc1 = self.encoder2(hidden_states_out[0])                # (B, F,   T, 128, 128)
        enc2 = self.encoder3(hidden_states_out[1])                # (B, 2F,  T,  64,  64)
        enc3 = self.encoder4(hidden_states_out[2])                # (B, 4F,  T,  32,  32)

        dec4 = self.encoder10(hidden_states_out[4])               # (B, 16F, T,   8,   8) bottleneck
        dec3 = self.decoder5(dec4, hidden_states_out[3])          # (B, 8F,  T,  16,  16) upsample + skip
        dec2 = self.decoder4(dec3, enc3)                          # (B, 4F,  T,  32,  32)
        dec1 = self.decoder3(dec2, enc2)                          # (B, 2F,  T,  64,  64)
        dec0 = self.decoder2(dec1, enc1)                          # (B, F,   T, 128, 128)

        out = self.decoder1(dec0, enc0)                           # (B, F,   T, 256, 256)
        # out = torch.mean(out, dim=2)
        logits = self.out(out)                                    # (B, 2,   T, 256, 256); the trainer averages over T
        return logits


@register_model("swinunetr3d")
def build(spec):
    """SwinUNETR-3D for `spec`; option swin_temporal_downsample (default False).

    swin_temporal_downsample=False follows the TS-SatFire paper ("we do not
    use downsampling in the temporal dimension in any of our models"); True
    is the released code, whose Swin patch merging also halves an even
    number of days (6 -> 3).
    """
    return SwinUNETR(
        image_size=spec.input_size,
        patch_size=PATCH_SIZE,
        window_size=(spec.ts_length, *WINDOW_HW),
        in_channels=spec.in_channels,
        out_channels=spec.num_classes,
        depths=DEPTHS,
        num_heads=(spec.num_heads, spec.num_heads, spec.num_heads, spec.num_heads),
        feature_size=spec.feature_size,
        norm_name="batch",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        # Swin V1 attention as in the released code; --model-option attn_version=v2 for Swin V2.
        attn_version=spec.options.get("attn_version", "v1"),
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        temporal_downsample=spec.options.get("swin_temporal_downsample", False),
    )
