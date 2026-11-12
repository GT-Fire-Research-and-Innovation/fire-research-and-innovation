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
    """Extract spatial features without changing image dimensions.

    Two 3x3 convolutions, each followed by group normalization and ReLU. The
    padding keeps the height and width of the input.

    Attributes:
        layers (torch.nn.Sequential): the two convolution, normalization and ReLU steps.
    """

    def __init__(self, in_channels, out_channels):
        """Build the two convolution steps.

        Group normalization with one group normalizes each sample on its own.
        The block therefore behaves the same at any batch size.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
        """
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
        """Apply the block to one batch of (B, in_channels, H, W) maps."""
        return self.layers(x)


class ConvLSTMCell(nn.Module):
    """Update spatial hidden state and memory for one observation.

    An LSTM cell whose gates are 3x3 convolutions, so the state keeps its
    spatial layout. One call advances the state by one day.

    Attributes:
        hidden_channels (int): channels of the hidden state and the memory.
        gates (torch.nn.Conv2d): computes the four gates from the input and hidden state.
    """

    def __init__(self, in_channels, hidden_channels):
        """Build the convolution that computes all four gates at once.

        The convolution reads the day's features and the hidden state side by
        side. Its output holds the four gates one after another.

        Args:
            in_channels (int): channels of each day's features.
            hidden_channels (int): channels of the hidden state and the memory.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, state):
        """Advance the hidden state and memory by one observation.

        The input, forget and output gates decide how much new information is
        stored, how much memory is kept and how much of it is exposed. The
        candidate information comes from the same convolution.

        Args:
            x (torch.Tensor): (B, in_channels, H, W) features of one day.
            state (tuple): (h, c), each (B, hidden_channels, H, W).

        Returns:
            tuple: the new (h, c).
        """
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
    """Shared 2D encoder, temporal bottleneck, and 2D decoder.

    Each day goes through the same 2D encoder. A ConvLSTM cell carries the
    bottleneck features from day to day, and the decoder upsamples its final
    state with the last day's skip connections.

    Attributes:
        encoder1, encoder2, bottleneck (ConvBlock): the shared per-day encoder.
        pool1, pool2 (torch.nn.MaxPool2d): 2x downsampling between encoder levels.
        temporal (ConvLSTMCell): the state carried across days.
        decoder2, decoder1 (ConvBlock): the decoder, deepest first.
        head (torch.nn.Conv2d): 1x1 conv to the class logits.
    """

    def __init__(self, in_channels, num_classes, width=WIDTH):
        """Build the encoder, the ConvLSTM cell and the decoder.

        The encoder has two 2x downsampling levels before the bottleneck. The
        decoder mirrors them and adds the matching skip connections.

        Args:
            in_channels (int): input channels per day.
            num_classes (int): output channels.
            width (int): channels of the first encoder level (doubled at each level).

        Raises:
            ValueError: if width is below 1.
        """
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
        """Predict next-day logits from a window of days.

        The days are encoded in chronological order and fed to the ConvLSTM
        cell. Its last hidden state is decoded with the final day's skips.

        Args:
            x (torch.Tensor): input of shape (B, C, T, H, W).

        Returns:
            torch.Tensor: logits of shape (B, K, H, W).

        Raises:
            ValueError: if x is not 5-D, has no days, or is smaller than 4x4.
        """
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