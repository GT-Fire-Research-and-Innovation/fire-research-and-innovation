"""What happens to one window between disk and model: normalize, augment, encode.

Every function here returns new arrays and leaves its inputs unchanged.

Level 1 (data). Used by data/fire_dataset.py, in this order per window:

    normalize    standardize every channel except angles and land cover
    augment      (training only) random flips and 90-degree rotation
    preprocess   sin(angle) for the angle channels, one-hot land cover:
                 (27, T, H, W) -> (43, T, H, W)
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF

from firepred.config.dataset import DEGREE_CHANNELS, LANDCOVER_CHANNEL, N_LANDCOVER_CLASSES


class Normalize:
    """Normalize selected input channels using fixed dataset statistics."""

    def __init__(self, mean, std, dont_normalize_idc):
        """Store per-channel statistics; channels in `dont_normalize_idc` pass through."""
        # [released] the original per-channel normalization statistics.
        self.mean = mean
        self.std = std
        self.dont_normalize_idc = set(dont_normalize_idc)

        # [ours] fail early when the supplied statistics are inconsistent.
        if len(self.mean) != len(self.std):
            raise ValueError(
                "Normalization mean/std lengths differ: "
                f"{len(self.mean)} versus {len(self.std)}"
            )

    def __call__(self, sample):
        """A standardized copy of `sample` (C, T, H, W), tensor or array; `sample` is unchanged."""
        # [ours] validate the channel dimension first.
        if sample.ndim != 4:
            raise ValueError(
                "Expected input shaped (C, T, H, W), but found "
                f"{tuple(sample.shape)}"
            )

        if sample.shape[0] != len(self.mean):
            raise ValueError(
                f"Expected {len(self.mean)} channels for normalization, "
                f"but found {sample.shape[0]}"
            )

        # [released] normalize one channel at a time and skip angle/land-cover
        # channels, which preprocess() encodes instead. The per-channel
        # arithmetic is kept as released so results stay bit-identical.
        normalized = sample.copy() if isinstance(sample, np.ndarray) else sample.clone()
        for channel_idx in range(len(self.mean)):
            if channel_idx not in self.dont_normalize_idc:
                normalized[channel_idx] = (
                    sample[channel_idx] - self.mean[channel_idx]
                ) / self.std[channel_idx]

        return normalized


def landcover_one_hot_matrix(n_classes=N_LANDCOVER_CLASSES):
    """(n_classes + 1, n_classes) lookup table: row k is class k's encoding.

    [released] classes 1..n map to one-hot rows. [ours] row 0, the leading
    zero row, encodes missing land cover (0) as no class.
    """
    return torch.cat((torch.zeros(1, n_classes), torch.eye(n_classes)), dim=0)


def augment(x, y, static=None, degree_channels=DEGREE_CHANNELS):
    """Random horizontal flip, vertical flip and multiple-of-90-degree rotation.

    [released] three independent draws from np.random per sample, applied to
    the input x (C, T, H, W) and target y (2, H, W) alike; angle channels are
    rotated with the raster so wind still points the right way. [ours]
    `static` (the TESSERA map, or None) gets the same transforms.
    """
    hflip = bool(np.random.random() > 0.5)
    vflip = bool(np.random.random() > 0.5)
    rotate = int(np.floor(np.random.random() * 4))

    if hflip:
        x = TF.hflip(x)
        y = TF.hflip(y)
        if static is not None:
            static = TF.hflip(static)

        # Reflect directional angles across the vertical axis.
        x[degree_channels] = (
            360 - x[degree_channels]
        ) % 360

    if vflip:
        x = TF.vflip(x)
        y = TF.vflip(y)
        if static is not None:
            static = TF.vflip(static)

        # Reflect directional angles across the horizontal axis.
        x[degree_channels] = (
            180 - x[degree_channels]
        ) % 360

    if rotate:
        angle = rotate * 90

        # torchvision rotates the last two dimensions and supports both
        # (2,H,W) and (2,T,H,W); no squeeze/unsqueeze is needed.
        x = TF.rotate(x, angle)
        y = TF.rotate(y, angle)
        if static is not None:
            static = TF.rotate(static, angle)

        # Rotate directional values consistently with the raster.
        x[degree_channels] = (
            x[degree_channels] - angle
        ) % 360

    return x, y, static


def preprocess(x, one_hot_matrix, degree_channels=DEGREE_CHANNELS, landcover_channel=LANDCOVER_CHANNEL):
    """sin() of the angle channels and land cover -> one-hot planes, as a new tensor. [released]

    (27, T, H, W) -> (27 - 1 + 17, T, H, W) = (43, T, H, W); `x` is unchanged.
    """
    new_shape = (
        x.shape[1],
        x.shape[2],
        x.shape[3],
        one_hot_matrix.shape[1],
    )

    # Source land-cover classes use integer identifiers 1..17.
    landcover_classes = x[landcover_channel].long()

    # [paper] 0 is a missing value (preprocessing replaces NaN with 0), read
    # as "an absence of all land covers", so it gets an all-zero encoding;
    # the released code indexed it as -1, silently making it class 17.
    invalid_landcover = (landcover_classes < 0) | (
        landcover_classes > one_hot_matrix.shape[1]
    )
    if torch.any(invalid_landcover):
        invalid_values = torch.unique(
            landcover_classes[invalid_landcover]
        ).tolist()
        raise ValueError(
            "Land-cover channel contains values outside 0..17: "
            f"{invalid_values[:20]}"
        )

    # Row 0 of the padded matrix is the all-zero encoding for class 0.
    landcover_flat = landcover_classes.flatten()
    landcover_encoding = one_hot_matrix[landcover_flat].reshape(
        new_shape
    ).permute(3, 0, 1, 2)

    # Assemble the output channel by channel, so `x` itself is never written:
    # angles become sin(angle) (after augmentation has rotated them), land
    # cover becomes its one-hot planes, every other channel is kept.
    degree_channels = set(degree_channels)
    pieces = []
    for channel in range(x.shape[0]):
        if channel == landcover_channel:
            pieces.append(landcover_encoding)
        elif channel in degree_channels:
            pieces.append(torch.sin(torch.deg2rad(x[channel:channel + 1])))
        else:
            pieces.append(x[channel:channel + 1])
    return torch.cat(pieces, dim=0)
