"""ResNet-18 U-Net: a ResNet-18 encoder and a U-Net decoder over the days stacked as channels.

Level 2 (models/zoo).

The T input days are folded into the channels, so one 2-D network sees the
whole window at once:

    (B, C, T, H, W) -> (B, C T, H, W)
      full-resolution conv block                    32 @ 1/1   skip for one-pixel fire edges
      ResNet-18 stem (7x7 conv, stride 2)            64 @ 1/2
      maxpool + layer1                               64 @ 1/4
      layer2, layer3, layer4                128, 256, 512 @ 1/8 .. 1/32
      five decoder steps (upsample, concat skip, convs) back to 1/1
    -> (B, K, H, W) logits, one next-day map

The encoder is torchvision's resnet18 built without pretrained weights: the
ImageNet stem expects 3 channels, so it is replaced by one for C x T channels.
T must therefore match spec.ts_length.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from firepred.models.registry import register_model

NAME = "res18unet"

# Architecture constants of this model.
FULL_WIDTH = 32  # channels of the full-resolution skip; override with --model-option full_width=16


def conv_block(in_channels, out_channels):
    """Two 3x3 convolutions with batch norm and ReLU, keeping the image size.

    Args:
        in_channels (int): input channels.
        out_channels (int): output channels.

    Returns:
        torch.nn.Sequential: the block.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    """Upsample to the skip's size, concatenate the skip, then convolve.

    Attributes:
        conv (torch.nn.Sequential): the conv block after the concatenation.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        """Build the conv block for the concatenated channels.

        Args:
            in_channels (int): channels coming up from the deeper level.
            skip_channels (int): channels of the skip connection.
            out_channels (int): output channels.
        """
        super().__init__()
        self.conv = conv_block(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        """Join one deeper feature map with its skip connection.

        Args:
            x (torch.Tensor): (B, in_channels, h, w).
            skip (torch.Tensor): (B, skip_channels, H, W), H and W at least h and w.

        Returns:
            torch.Tensor: (B, out_channels, H, W).
        """
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class Model(nn.Module):
    """ResNet-18 U-Net over the days stacked as channels.

    Attributes:
        full (torch.nn.Sequential): full-resolution skip block.
        stem (torch.nn.Sequential): 7x7 stride-2 conv for C x T channels, batch norm, ReLU.
        pool (torch.nn.MaxPool2d): ResNet's stride-2 max pool.
        layer1..layer4 (torch.nn.Sequential): the four ResNet-18 stages.
        up4..up0 (UpBlock): the decoder, deepest first.
        head (torch.nn.Conv2d): 1x1 conv to the class logits.
    """

    def __init__(self, in_channels, num_classes, ts_length, full_width):
        """Build the encoder from torchvision's resnet18 and the U-Net decoder.

        Args:
            in_channels (int): input channels per day.
            num_classes (int): output channels.
            ts_length (int): input days per window (T).
            full_width (int): channels of the full-resolution skip.
        """
        super().__init__()
        stacked = in_channels * ts_length
        self.ts_length = ts_length
        resnet = resnet18(weights=None)

        self.full = conv_block(stacked, full_width)
        self.stem = nn.Sequential(
            nn.Conv2d(stacked, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
        )
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = UpBlock(64, full_width, full_width)
        self.head = nn.Conv2d(full_width, num_classes, kernel_size=1)

    def forward(self, x):                                 # (B, C, T, H, W)
        """Run the ResNet-18 U-Net on one batch.

        Args:
            x (torch.Tensor): input of shape (B, C, T, H, W), T = ts_length.

        Returns:
            torch.Tensor: next-day logits of shape (B, K, H, W).

        Raises:
            ValueError: if T differs from the ts_length the model was built for.
        """
        b, c, t, h, w = x.shape
        if t != self.ts_length:
            raise ValueError(f"{NAME} was built for {self.ts_length} days, got {t}")
        x = x.reshape(b, c * t, h, w)                     # (B, C T, H, W)     days -> channels

        s0 = self.full(x)                                 # (B, 32,  H,    W)
        s1 = self.stem(x)                                 # (B, 64,  H/2,  W/2)
        s2 = self.layer1(self.pool(s1))                   # (B, 64,  H/4,  W/4)
        s3 = self.layer2(s2)                              # (B, 128, H/8,  W/8)
        s4 = self.layer3(s3)                              # (B, 256, H/16, W/16)
        x = self.layer4(s4)                               # (B, 512, H/32, W/32)

        x = self.up4(x, s4)                               # (B, 256, H/16, W/16)
        x = self.up3(x, s3)                               # (B, 128, H/8,  W/8)
        x = self.up2(x, s2)                               # (B, 64,  H/4,  W/4)
        x = self.up1(x, s1)                               # (B, 64,  H/2,  W/2)
        x = self.up0(x, s0)                               # (B, 32,  H,    W)
        return self.head(x)                               # (B, K,   H,    W)    logits


@register_model(NAME)
def build(spec):
    """Build the ResNet-18 U-Net for a ModelSpec.

    Uses in_channels, num_classes and ts_length. options["full_width"] sets
    the full-resolution skip width.

    Args:
        spec (ModelSpec): what the model is built for.

    Returns:
        Model: the new model.
    """
    return Model(spec.in_channels, spec.num_classes, spec.ts_length,
                 int(spec.options.get("full_width", FULL_WIDTH)))
