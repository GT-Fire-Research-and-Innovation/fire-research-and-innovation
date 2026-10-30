"""ConvLSTM U-Net for next-day wildfire prediction.

Input:  (B, C, T, H, W)
Output: (B, K, H, W), raw logits.
"""

import torch
from torch import nn
from torch.nn import functional as F

from firepred.models.registry import register_model

NAME = "convlstm"
WIDTH = 16


class ConvBlock(nn.Module):
    """Extract spatial features without changing image dimensions."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ConvLSTMCell(nn.Module):
    """Update spatial hidden state and memory for one observation."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, state):
        h, c = state

        # Compute all four gates together, then separate them.
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)  # how much new information to store
        f = torch.sigmoid(f)  # how much old memory to retain
        o = torch.sigmoid(o)  # how much memory to expose
        g = torch.tanh(g)     # candidate new information

        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class Model(nn.Module):
    """Shared 2D encoder, temporal bottleneck, and 2D decoder."""

    def __init__(self, in_channels, num_classes, width=WIDTH):
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")

        self.encoder1 = ConvBlock(in_channels, width)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(width, 2 * width)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(2 * width, 4 * width)

        self.temporal = ConvLSTMCell(4 * width, 4 * width)

        self.decoder2 = ConvBlock(6 * width, 2 * width)
        self.decoder1 = ConvBlock(3 * width, width)
        self.head = nn.Conv2d(width, num_classes, kernel_size=1)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("Expected input shape (B, C, T, H, W)")
        if x.shape[2] < 1:
            raise ValueError("Input must contain at least one time step")
        if min(x.shape[-2:]) < 4:
            raise ValueError("Height and width must each be at least 4")

        state = None

        # Process observations in chronological order.
        for t in range(x.shape[2]):
            frame = x[:, :, t]

            skip1 = self.encoder1(frame)
            skip2 = self.encoder2(self.pool1(skip1))
            features = self.bottleneck(self.pool2(skip2))

            if state is None:
                shape = (
                    features.shape[0],
                    self.temporal.hidden_channels,
                    features.shape[2],
                    features.shape[3],
                )
                state = (
                    features.new_zeros(shape),
                    features.new_zeros(shape),
                )

            state = self.temporal(features, state)

        # These skip connections come from the final input day.
        h, _ = state
        decoded = F.interpolate(
            h, size=skip2.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        decoded = self.decoder2(torch.cat([decoded, skip2], dim=1))

        decoded = F.interpolate(
            decoded, size=skip1.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        decoded = self.decoder1(torch.cat([decoded, skip1], dim=1))

        return self.head(decoded)


@register_model(NAME)
def build(spec):
    """Build using the repository's data specification and model options."""
    width = int(spec.options.get("width", WIDTH))
    return Model(spec.in_channels, spec.num_classes, width)