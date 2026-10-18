"""Window bookkeeping for Swin attention: cut a volume into windows and back.

Level 2 (models/blocks). Used by swin_block.py.

A (B, D, H, W, C) feature volume is split into non-overlapping windows of
window_size = (wd, wh, ww) tokens; attention then runs inside each window
independently. compute_mask() builds the -100 additive mask that stops
tokens from attending across the seams created by the cyclic shift.

Adapted from MONAI / the Swin Transformer reference code (Liu et al. 2021).
[released] The released code also built a time mask for an "autoregressive"
attention variant on every forward pass; no model used it, and it is gone.
"""

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def window_partition(x, window_size):
    '''window partition operation based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        x:           input tensor.
        window_size: local window size.
    '''
    x_shape = x.size()
    if len(x_shape) == 5:
        # 3-D case: cut (B, D, H, W, C) into blocks of window_size tokens and list every
        # block as one row: (num_windows * B, tokens_per_window, C). Attention then runs
        # on each row independently.
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    '''window reverse operation based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        windows:     windows tensor.
        window_size: local window size.
        dims:        dimension values.
    '''
    # Inverse of window_partition: put the window rows back into a (B, D, H, W, C) volume.
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

def get_window_size(x_size, window_size, shift_size=None):
    '''Computing window size based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        x_size:      input size.
        window_size: local window size.
        shift_size:  window shifting size.
    '''

    # A window can't be larger than the volume: where an axis is smaller than the
    # window (e.g. all 6 days), the window shrinks to it and that axis is not shifted.
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def compute_mask(dims, window_size, shift_size, device):
    '''Computing region masks based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        dims:        dimension values.
        window_size: local window size.
        shift_size:  shift size.
        device:      device.
    '''

    # Shifted-window mask. After the cyclic shift, a window at the volume's edge
    # contains tokens that wrapped around from the far side. Label every token with
    # the region it came from; tokens from different regions must not attend to
    # each other.
    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    # Compare region labels pairwise inside each window: equal -> 0 (allowed),
    # different -> -100 (after softmax, effectively zero attention).
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask