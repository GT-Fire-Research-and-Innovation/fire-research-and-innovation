"""TESSERA stem: compress a fine TESSERA map to the VIIRS grid before the backbone. [ours]

Level 2 (models/blocks). Used by models/registry.py when a run keeps k x k
TESSERA cells per VIIRS pixel (--tessera-cells-per-pixel k, k > 1).

    static (B, 129, 256 k, 256 k)   normalized embedding (128) + validity mask (1)
      embedding -> Conv2d(128, 128, kernel k, stride k)   learned, one vector per VIIRS pixel
      mask      -> AvgPool2d(k)                           fraction of valid cells
    -> (B, 129, 256, 256), repeated over the T days and appended to the 43 channels

The backbone therefore still receives 172 channels per pixel, exactly as with
one averaged TESSERA vector per pixel (k = 1, where no stem is built).
"""

import torch
from torch import nn

from firepred.config.dataset import TESSERA_BANDS


class TesseraStem(nn.Module):
    """Turn a k-times finer TESSERA map into one learned vector per VIIRS pixel.

    A k x k convolution with stride k sees exactly the k x k cells inside one
    VIIRS pixel and mixes them into 128 values, so the model can learn which
    parts of the pixel matter instead of taking a plain average. The validity
    mask is averaged, giving the fraction of valid cells.

    Attributes:
        cells_per_pixel (int): k, TESSERA cells along each side of one VIIRS pixel.
        embed (torch.nn.Conv2d): the learned k x k, stride-k convolution.
        mask_pool (torch.nn.AvgPool2d): averages the validity mask over each pixel.
    """

    def __init__(self, cells_per_pixel, bands=TESSERA_BANDS):
        """Build the convolution and the mask pooling for k cells per pixel.

        The convolution keeps the band count, so the backbone's input width does
        not change. Its starting weights come from reset_parameters.

        Args:
            cells_per_pixel (int): k (2 or more).
            bands (int): embedding bands (128).

        Raises:
            ValueError: if cells_per_pixel is below 2.
        """
        super().__init__()
        if cells_per_pixel < 2:
            raise ValueError(f"TesseraStem needs at least 2 cells per pixel, got {cells_per_pixel}")
        self.cells_per_pixel = cells_per_pixel
        self.embed = nn.Conv2d(bands, bands, kernel_size=cells_per_pixel, stride=cells_per_pixel)
        self.mask_pool = nn.AvgPool2d(cells_per_pixel)
        self.reset_parameters()

    def reset_parameters(self):
        """Start the convolution as a plain average of each pixel's cells.

        Band i of the output starts as the mean of band i over the k x k cells, so
        before training the stem gives exactly what one averaged vector per pixel
        would (up to normalizing before rather than after averaging). Training
        then learns where to depart from the average.

        Returns:
            TesseraStem: this stem.
        """
        cells = self.cells_per_pixel
        with torch.no_grad():
            identity = torch.eye(self.embed.out_channels, self.embed.in_channels)[:, :, None, None]
            self.embed.weight.copy_(identity.expand(-1, -1, cells, cells) / cells ** 2)
            self.embed.bias.zero_()
        return self

    def forward(self, static):
        """Compress a batch of fine TESSERA maps to the VIIRS grid.

        Maps arrive as float16 to halve host memory. Samples are widened and
        convolved one at a time, so only one float32 copy of a large map exists.

        Args:
            static (torch.Tensor): (B, 129, 256 k, 256 k), embedding bands then the mask.

        Returns:
            torch.Tensor: (B, 129, 256, 256) float32.
        """
        pieces = []
        for sample in static:
            sample = sample.unsqueeze(0).to(self.embed.weight.dtype)
            pieces.append(torch.cat((self.embed(sample[:, :-1]), self.mask_pool(sample[:, -1:])), dim=1))
        return torch.cat(pieces, dim=0)


class TesseraStemModel(nn.Module):
    """Run the TESSERA stem, then any registered backbone on the combined input.

    The backbone is built exactly as it would be for one TESSERA vector per
    pixel. This wrapper only adds the stem in front and takes the fine map as a
    second input.

    Attributes:
        stem (TesseraStem): compresses the fine map.
        backbone (torch.nn.Module): the registered model.
    """

    def __init__(self, backbone, cells_per_pixel):
        """Put a stem for k cells per pixel in front of a backbone.

        The backbone keeps its own input width (172 channels). Checkpoint keys
        start with "stem." and "backbone.".

        Args:
            backbone (torch.nn.Module): the registered model.
            cells_per_pixel (int): k.
        """
        super().__init__()
        self.stem = TesseraStem(cells_per_pixel)
        self.backbone = backbone

    def forward(self, data, static):
        """Compress the fine map, repeat it over the days, append it and run the backbone.

        This is what batch_inputs does for a one-vector-per-pixel map, after the
        stem. The days dimension is T of data.

        Args:
            data (torch.Tensor): (B, 43, T, H, W), the encoded VIIRS days.
            static (torch.Tensor): (B, 129, k H, k W), the fine TESSERA map.

        Returns:
            torch.Tensor: the backbone's logits.
        """
        static = self.stem(static)
        static = static.unsqueeze(2).expand(-1, -1, data.shape[2], -1, -1)
        return self.backbone(torch.cat((data, static), dim=1))
