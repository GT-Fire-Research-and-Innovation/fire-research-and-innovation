"""The Swin encoder: patch embedding plus four Swin stages -> five feature scales.

Level 2 (models/blocks). Used by models/zoo/swinunetr3d.py.

Input (B, C, T, H, W). PatchEmbed turns each (1, 2, 2) patch into a
feature_size vector, then each stage runs shifted-window attention and merges
2x2 patches (doubling channels). forward() returns the five scales
[x0 .. x4], each LayerNorm-ed over channels when normalize=True; with
T=6, 256x256 and feature_size 36 they are
(B, 36, 6, 128, 128), (B, 72, 6, 64, 64), (B, 144, 6, 32, 32),
(B, 288, 6, 16, 16), (B, 576, 6, 8, 8). See docs/swinunetr3d/shapes.md.

From the Swin Transformer reference code (Liu et al. 2021) via MONAI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from einops import rearrange
from functools import partial
from monai.networks.blocks import PatchEmbed
from torch.nn import LayerNorm
from typing import Sequence, Type

from firepred.models.blocks.patch_merging import PatchMerging
from firepred.models.blocks.swin_block import BasicLayer


class SwinTransformer(nn.Module):
    """The Swin encoder: patch embedding then four stages, each at a coarser scale.

    The input is cut into patches (here only in space, so attention spans all
    days) and embedded. Four stages of Swin blocks follow, each halving H and
    W. The output of every scale feeds SwinUNETR's decoder (Liu et al., ICCV
    2021, via MONAI).
    """

    def __init__(
        self,
        image_size:     Sequence[int],
        in_chans:       int,
        embed_dim:      int,
        window_size:    Sequence[int],
        patch_size:     Sequence[int],
        depths:         Sequence[int],
        num_heads:      Sequence[int],
        mlp_ratio:      float = 4.0,
        qkv_bias:       bool = True,
        drop_rate:      float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_version:   str = 'v1',
        norm_layer:     Type[LayerNorm] = nn.LayerNorm,
        patch_norm:     bool = False,
        use_checkpoint: bool = False,
        spatial_dims:   int = 3,
        temporal_downsample: bool = True,
    ) -> None:
        """Build the patch embedding and the four stages.

        Each stage halves H and W and doubles the channels. The stochastic depth
        rate grows linearly over all blocks (0 for SwinUNETR-3D).

        Args:
            image_size (Sequence[int]): (T, H, W) of the input.
            in_chans (int): input channels.
            embed_dim (int): channels after patch embedding.
            window_size (Sequence[int]): window size per axis.
            patch_size (Sequence[int]): patch size per axis.
            depths (Sequence[int]): blocks per stage.
            num_heads (Sequence[int]): attention heads per stage.
            mlp_ratio (float): MLP hidden width as a multiple of the channels.
            qkv_bias (bool): learnable bias on queries, keys and values.
            drop_rate (float): dropout rate.
            attn_drop_rate (float): attention dropout rate.
            drop_path_rate (float): largest stochastic depth rate.
            attn_version (str): "v1" or "v2".
            norm_layer (type[nn.Module]): normalization layer class.
            patch_norm (bool): normalize after patch embedding.
            use_checkpoint (bool): recompute activations to save memory.
            spatial_dims (int): 2 or 3.
            temporal_downsample (bool): patch merging also halves an even number of days.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        # Patch embedding: a strided convolution turns each patch (1 day x 2 x 2 pixels)
        # into one token of embed_dim numbers.
        self.patch_embed = PatchEmbed(
            patch_size   = self.patch_size,
            in_chans     = in_chans,
            embed_dim    = embed_dim,
            norm_layer   = norm_layer if self.patch_norm else None,
            spatial_dims = spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        # Stochastic-depth rate per block, rising linearly with depth (all 0 here).
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        self.resamples = []

        input_size = deepcopy(image_size)
        # Four stages. Stage i works at embed_dim * 2^i channels on a grid halved i
        # times; each ends with patch merging. resamples records each stage's
        # downsampling so the decoder can upsample by the same factors.
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                input_resolution = input_size,
                dim              = int(embed_dim * 2**i_layer),
                depth            = depths[i_layer],
                num_heads        = num_heads[i_layer],
                window_size      = self.window_size,
                drop_path        = dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio        = mlp_ratio,
                qkv_bias         = qkv_bias,
                drop             = drop_rate,
                attn_drop        = attn_drop_rate,
                attn_version     = attn_version,
                norm_layer       = norm_layer,
                downsample       = partial(PatchMerging, merge_temporal=temporal_downsample),
                use_checkpoint   = use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)

            input_size = layer.downsample.output_resolution
            self.resamples.insert(0, layer.downsample.resample_scale)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        """Normalize one stage's output over its channels before it goes to the decoder.

        The normalization has no learned scale. Without normalize the features are
        returned unchanged.

        Args:
            x (torch.Tensor): features, channels-first (N, C, D, H, W) or (N, C, H, W).
            normalize (bool): apply the normalization.

        Returns:
            torch.Tensor: same shape as x.
        """
        # Optional channel-wise LayerNorm (no learned scale) on each stage's output
        # before it goes to the decoder.
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, 'n c d h w -> n d h w c')
                x = F.layer_norm(x, [ch])
                x = rearrange(x, 'n d h w c -> n c d h w')
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, 'n c h w -> n h w c')
                x = F.layer_norm(x, [ch])
                x = rearrange(x, 'n h w c -> n c h w')
        return x

    def forward(self, x, normalize=True):
        """Embed the input into patches, run the four stages and return every scale.

        The five outputs are taken after embedding and then after each stage. They
        are the decoder's skip connections, from fine to coarse.

        Args:
            x (torch.Tensor): input of shape (B, C, T, H, W).
            normalize (bool): normalize each output over its channels.

        Returns:
            list[torch.Tensor]: five feature volumes, each half the size of the previous one.
        """
        # Run the stages in order and return all five scales (after patch embedding,
        # then after each stage) for the decoder's skip connections.
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]