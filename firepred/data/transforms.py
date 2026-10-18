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


def normalize(sample, mean, std, skip=()):
    """Return a standardized copy of one window with fixed dataset statistics.

    Each channel has its own mean and spread, computed by the released code over
    the training data. Standardizing ((x - mean) / std) puts every channel on a
    similar scale, which helps training. Skipped channels (angles and land
    cover) are copied unchanged, because preprocess encodes them differently.

    Args:
        sample (torch.Tensor | np.ndarray): one window, shape (C, T, H, W).
        mean (Sequence[float]): per-channel means (27 values).
        std (Sequence[float]): per-channel standard deviations (27 values).
        skip (Iterable[int]): channels left as they are.

    Returns:
        torch.Tensor | np.ndarray: the standardized copy, same type and shape.

    Raises:
        ValueError: if mean and std differ in length, or the window is not 4-D
            or has the wrong channel count.
    """
    # [ours] fail early when the statistics or the window are inconsistent.
    if len(mean) != len(std):
        raise ValueError(f"Normalization mean/std lengths differ: {len(mean)} versus {len(std)}")
    if sample.ndim != 4:
        raise ValueError(f"Expected input shaped (C, T, H, W), but found {tuple(sample.shape)}")
    if sample.shape[0] != len(mean):
        raise ValueError(f"Expected {len(mean)} channels for normalization, but found {sample.shape[0]}")

    # [released] normalize one channel at a time and skip angle/land-cover
    # channels, which preprocess() encodes instead. The per-channel arithmetic
    # is kept as released so results stay bit-identical.
    skip = set(skip)
    normalized = sample.copy() if isinstance(sample, np.ndarray) else sample.clone()
    for channel_idx in range(len(mean)):
        if channel_idx not in skip:
            normalized[channel_idx] = (sample[channel_idx] - mean[channel_idx]) / std[channel_idx]
    return normalized


def landcover_one_hot_matrix(n_classes=N_LANDCOVER_CLASSES):
    """Build the lookup table that turns a land-cover class into one-hot planes.

    Row k is the encoding of class k: classes 1..17 get a single 1 in their own
    column [released]. Row 0, for missing land cover, is all zeros, meaning "no
    class" [ours].

    Args:
        n_classes (int): number of land-cover classes.

    Returns:
        torch.Tensor: shape (n_classes + 1, n_classes), float.
    """
    return torch.cat((torch.zeros(1, n_classes), torch.eye(n_classes)), dim=0)


def augment(x, y, static=None, degree_channels=DEGREE_CHANNELS, rng=np.random):
    """Randomly flip and rotate a training window, its target and its TESSERA map together.

    Three random draws decide a horizontal flip, a vertical flip and a rotation
    by 0, 90, 180 or 270 degrees [released]. The same change is applied to the
    input, the target and the TESSERA map, so they stay aligned; angle channels
    are rotated too, so wind still points the right way. New tensors are
    returned and the inputs are left unchanged.

    Args:
        x (torch.Tensor): the input window, shape (C, T, H, W).
        y (torch.Tensor): the target, shape (2, H, W).
        static (torch.Tensor | None): the TESSERA map (129, H, W), or None.
        degree_channels (list[int]): the channels holding angles in degrees.
        rng (np.random.Generator | module): where the three draws come from; numpy's
            global generator by default, as released.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: (x, y, static) after the change.
    """
    hflip = bool(rng.random() > 0.5)
    vflip = bool(rng.random() > 0.5)
    rotate = int(np.floor(rng.random() * 4))

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
    """Encode a window for the model: sin() of the angles and one-hot land cover.

    Angles in degrees become sin(angle) [released]. The land-cover channel
    (class 1..17, 0 = missing) becomes 17 one-hot planes, so the stored 27
    channels become 43. A new tensor is returned and x is unchanged.

    Args:
        x (torch.Tensor): a window, shape (27, T, H, W).
        one_hot_matrix (torch.Tensor): landcover_one_hot_matrix().
        degree_channels (list[int]): channels holding angles in degrees.
        landcover_channel (int): the land-cover channel.

    Returns:
        torch.Tensor: shape (43, T, H, W).

    Raises:
        ValueError: if land cover holds a value outside 0..17.
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
