"""One Swin stage: shifted-window transformer blocks.

Level 2 (models/blocks). Used by swin_encoder.py.

SwinTransformerBlock  x + attention(window(LayerNorm(x))), then x + MLP(LayerNorm(x)).
                      Odd-numbered blocks shift the windows by half a window first,
                      so information crosses window borders.
BasicLayer            `depth` blocks (alternating unshifted/shifted), then an optional
                      downsample (PatchMerging). Input/output (B, C, D, H, W).

From the Swin Transformer reference code (Liu et al. 2021) via MONAI, with the
attention selectable by attn_version: "v1" (Swin V1, the released code and the
default) or "v2" (Swin V2 scaled cosine attention).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.layers import DropPath
from torch.nn import LayerNorm
from typing import Optional, Sequence, Type

from firepred.models.blocks.window_attention import WindowAttentionV1
from firepred.models.blocks.window_attention_v2 import WindowAttentionV2
from firepred.models.blocks.window_ops import compute_mask, get_window_size, window_partition, window_reverse


class SwinTransformerBlock(nn.Module):
    '''
    Swin Transformer block based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        dim:            int,
        num_heads:      int,
        window_size:    Sequence[int],
        shift_size:     Sequence[int],
        mlp_ratio:      float = 4.0,
        qkv_bias:       bool = True,
        drop:           float = 0.0,
        attn_drop:      float = 0.0,
        attn_version:   str = 'v1',
        drop_path:      float = 0.0,
        act_layer:      str = 'GELU',
        norm_layer:     Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        '''
        Args:
            dim: number     of feature channels.
            num_heads:      number of attention heads.
            window_size:    local window size.
            shift_size:     window shift size.
            mlp_ratio:      ratio of mlp hidden dim to embedding dim.
            qkv_bias:       add a learnable bias to query, key, value.
            drop:           dropout rate.
            attn_drop:      attention dropout rate.
            drop_path:      stochastic depth rate.
            act_layer:      activation layer.
            norm_layer:     normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        '''

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        # Layers of one block: LayerNorm -> window attention, then LayerNorm -> MLP.
        self.norm1 = norm_layer(dim)

        self.attn_version = attn_version
        if attn_version == 'v1':
            self.attn = WindowAttentionV1(
                dim,
                window_size = self.window_size,
                num_heads   = num_heads,
                qkv_bias    = qkv_bias,
                attn_drop   = attn_drop,
                proj_drop   = drop,
            )
        elif attn_version == 'v2':
            self.attn = WindowAttentionV2(
                dim,
                window_size = self.window_size,
                num_heads   = num_heads,
                qkv_bias    = qkv_bias,
                attn_drop   = attn_drop,
                proj_drop   = drop,
            )
        else:
            raise ValueError(f"unknown attn_version {attn_version!r}; choose 'v1' or 'v2'")

        # Stochastic depth (off for SwinUNETR-3D) and the MLP that processes each token
        # on its own after attention (dim -> 4 x dim -> dim).
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            hidden_size  = dim,
            mlp_dim      = mlp_hidden_dim,
            act          = act_layer,
            dropout_rate = drop,
            dropout_mode = 'swin'
        )

    def forward_part1(self, x, mask_matrix):
        """Attention half of the block: norm, pad, (shift), window attention, unshift, unpad."""
        x_shape = x.size()
        # LayerNorm, then pad the volume so each axis is a whole number of windows.
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        # Odd blocks roll the volume by half a window so the new windows straddle the
        # old window borders; the mask stops wrapped-around tokens from mixing.
        # For first SwinTransformer Block there is no shift and mask, but the second one has shift then has image mask
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # Shift_x is the x after cyclic shifting
        # Cut into windows, run attention inside each, and stitch the windows back.
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            # Remove the padding again.
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        """MLP half of the block: norm, MLP, drop path."""
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """One Swin block with residual connections: x + attention(x), then x + MLP(x)."""
        # Residual connections: each half adds a correction to its input,
        # x + attention(...) then x + MLP(...), which keeps deep stacks trainable.
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x

class BasicLayer(nn.Module):
    '''
    Basic Swin Transformer layer in one stage based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        input_resolution: Sequence[int],
        dim:              int,
        depth:            int,
        num_heads:        int,
        window_size:      Sequence[int],
        drop_path:        list,
        mlp_ratio:        float = 4.0,
        qkv_bias:         bool = False,
        drop:             float = 0.0,
        attn_drop:        float = 0.0,
        attn_version:     str = 'v1',
        norm_layer:       Type[LayerNorm] = nn.LayerNorm,
        downsample:       Optional[nn.Module] = None,
        use_checkpoint:   bool = False,
    ) -> None:
        '''
        Args:
            input_resolution: resolution of input feature maps.
            dim:              number of feature channels.
            depth:            number of layers in each stage.
            num_heads:        number of attention heads.
            window_size:      local window size.
            drop_path:        stochastic depth rate.
            mlp_ratio:        ratio of mlp hidden dim to embedding dim.
            qkv_bias:         add a learnable bias to query, key, value.
            drop:             dropout rate.
            attn_drop:        attention dropout rate.
            norm_layer:       normalization layer.
            downsample:       an optional downsampling layer at the end of the layer.
            use_checkpoint:   use gradient checkpointing for reduced memory usage.
        '''

        super().__init__()
        self.window_size = window_size
        # use shift_size if i==1 otherwise no_shift
        # Blocks alternate: even blocks use regular windows, odd blocks shifted ones.
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim            = dim,
                    num_heads      = num_heads,
                    window_size    = self.window_size,
                    shift_size     = self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio      = mlp_ratio,
                    qkv_bias       = qkv_bias,
                    drop           = drop,
                    attn_drop      = attn_drop,
                    attn_version   = attn_version,
                    drop_path      = drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer     = norm_layer,
                    use_checkpoint = use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        # Patch merging at the end of the stage halves H and W (see patch_merging.py).
        if callable(self.downsample):
            self.downsample = downsample(
                input_resolution = input_resolution,
                dim              = dim,
                norm_layer       = norm_layer,
                spatial_dims     = len(self.window_size)
            )

    def forward(self, x):
        """Run every block of the stage on (B, C, D, H, W) with the shifted-window mask, then merge patches."""
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            # If window_size == x_shape then the shift_size = 0 in that dimension.
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            # Channels last for attention; build the shifted-window mask once for the stage,
            # run every block, merge patches, then channels first again.
            x = rearrange(x, 'b c d h w -> b d h w c')
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, 'b d h w c -> b c d h w')

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, 'b c h w -> b h w c')
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, 'b h w c -> b c h w')
        return x
