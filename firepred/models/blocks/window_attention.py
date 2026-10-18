"""Window multi-head self-attention with a learned relative position bias (Swin V1).

Level 2 (models/blocks). Used by swin_block.py when attn_version="v1", the
setting SwinUNETR-3D trains with.

Input  (num_windows * B, N, C): the N = wd*wh*ww tokens of each window.
Output the same shape. Each head computes softmax(q k^T / sqrt(d) + bias) v,
where bias[i, j] is a learned number per head for the relative (dt, dy, dx)
offset between tokens i and j.

From the Swin Transformer reference code (Liu et al. 2021) via MONAI.
"""

from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.layers import trunc_normal_


class WindowAttentionV1(nn.Module):
    '''
    Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        window_size: Sequence[int],
        qkv_bias:    bool = False,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
    ) -> None:
        '''
        Args:
            dim: number  of feature channels.
            num_heads:   number of attention heads.
            window_size: local window size.
            qkv_bias:    add a learnable bias to query, key, value.
            attn_drop:   attention dropout rate.
            proj_drop:   dropout rate of output.
        '''

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        # Each head works on dim / num_heads channels; scores are divided by sqrt(head_dim).
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # Relative position bias: one learned number per head for every possible offset
        # (d_day, d_row, d_col) between two tokens of a window. relative_position_index
        # maps each token pair to its offset's row in that table.
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        elif len(self.window_size) == 2:

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        # One linear layer makes query, key and value for every token; proj mixes the
        # heads' outputs back together.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        # x: (windows, tokens, channels). Split into per-head queries, keys and values.
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        # Score every token pair in the window by how well its key matches the query.
        attn = q @ k.transpose(-2, -1)
        # Add the learned bias for each pair's relative offset: the only place attention
        # learns where (and when) a token is relative to another.
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        # Shifted windows: add the -100 mask so wrapped-around tokens are ignored, then
        # turn scores into weights that sum to 1 (softmax).
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        # Each token's output is the weighted average of the values, heads concatenated.
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

