"""Per-window transforms: angle handling, land-cover encoding, augmentation consistency."""

import numpy as np
import torch

from firepred.config.dataset import (
    BASE_MODEL_CHANNELS,
    LANDCOVER_CHANNEL,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    WIND_DIRECTION_CHANNEL,
)
from firepred.data.fire_dataset import augment, landcover_one_hot_matrix, normalize, preprocess


def window(landcover):
    """A (27, 2, 4, 4) window with every angle 90 degrees and one land-cover class."""
    x = torch.zeros(27, 2, 4, 4)
    x[[12, 18, 24]] = 90.0  # wind direction -> 90 / 180 - 1 = -0.5; aspect, forecast wind -> sin(90) = 1
    x[LANDCOVER_CHANNEL] = landcover
    return x


def test_preprocess_gives_43_channels_with_angles_in_minus_one_to_one():
    """preprocess keeps 43 channels: the wind direction as degrees / 180 - 1, the other angles as sin(angle)."""
    out = preprocess(window(3.0), landcover_one_hot_matrix())
    assert out.shape == (BASE_MODEL_CHANNELS, 2, 4, 4) == (43, 2, 4, 4)
    assert torch.allclose(out[WIND_DIRECTION_CHANNEL], torch.full((2, 4, 4), -0.5))
    forecast = 24 + 17 - 1  # stored channel 24, after land cover (21) became 17 planes
    assert torch.allclose(out[18], torch.ones(2, 4, 4)) and torch.allclose(out[forecast], torch.ones(2, 4, 4))


def test_wind_direction_covers_the_whole_circle():
    """Every compass direction gets its own value in -1..1; sin gave north and south both 0."""
    degrees = torch.tensor([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 359.0, 360.0])
    x = torch.zeros(27, 1, 1, len(degrees))
    x[WIND_DIRECTION_CHANNEL, 0, 0] = degrees
    values = preprocess(x, landcover_one_hot_matrix())[WIND_DIRECTION_CHANNEL, 0, 0]
    expected = torch.tensor([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 359 / 180 - 1, -1.0])
    assert torch.allclose(values, expected)
    assert len(set(values[:-1].tolist())) == len(degrees) - 1  # 0..359: all different; 360 is north again
    assert values.min() >= -1 and values.max() < 1


def test_augmented_wind_direction_matches_the_rotated_map():
    """A 90-degree turn of the map turns the wind with it, so the encoding follows the new compass direction."""
    x = torch.zeros(27, 1, 4, 4)
    x[WIND_DIRECTION_CHANNEL] = 30.0
    for seed in range(8):  # covers flip and rotation draws
        np.random.seed(seed)
        turned, _, _ = augment(x.clone(), torch.zeros(2, 4, 4))
        encoded = preprocess(turned, landcover_one_hot_matrix())[WIND_DIRECTION_CHANNEL]
        assert torch.allclose(encoded, turned[WIND_DIRECTION_CHANNEL] / 180 - 1)


def test_landcover_classes_become_one_hot_and_zero_means_missing():
    """Land-cover class k becomes one-hot plane k-1, and 0 (missing) becomes all zeros."""
    out = preprocess(window(3.0), landcover_one_hot_matrix())
    planes = out[LANDCOVER_CHANNEL:LANDCOVER_CHANNEL + 17]
    assert torch.equal(planes[2], torch.ones(2, 4, 4)) and planes.sum() == 2 * 4 * 4
    missing = preprocess(window(0.0), landcover_one_hot_matrix())
    assert missing[LANDCOVER_CHANNEL:LANDCOVER_CHANNEL + 17].sum() == 0


def test_augment_moves_input_target_and_tessera_together():
    """Flips and rotations move input, target and TESSERA map together."""
    x = torch.zeros(27, 1, 4, 4)
    x[0, 0, 0, 1] = 1.0
    y = torch.zeros(2, 4, 4, dtype=torch.long)
    y[1, 0, 1] = 1
    static = torch.zeros(3, 4, 4)
    static[0, 0, 1] = 1.0
    np.random.seed(3)
    x2, y2, s2 = augment(x.clone(), y.clone(), static.clone())
    position = torch.nonzero(x2[0, 0])
    assert torch.equal(position, torch.nonzero(y2[1])) and torch.equal(position, torch.nonzero(s2[0]))


def test_augment_draws_exactly_three_random_numbers():
    """augment draws exactly three numbers from numpy's global generator."""
    np.random.seed(0)
    augment(torch.zeros(27, 1, 4, 4), torch.zeros(2, 4, 4))
    after = np.random.random()
    np.random.seed(0)
    np.random.random(3)
    assert after == np.random.random()


def test_transforms_leave_their_inputs_unchanged():
    """Normalize, preprocess and augment return new tensors; the caller's window is untouched."""
    x = window(3.0) + torch.arange(27.0)[:, None, None, None]
    x[LANDCOVER_CHANNEL] = 3.0
    original = x.clone()
    normalized = normalize(x, NORMALIZATION_MEAN, NORMALIZATION_STD, skip=[12, 18, 24, LANDCOVER_CHANNEL])
    assert torch.equal(x, original) and not torch.equal(normalized, original)
    preprocess(x, landcover_one_hot_matrix())
    assert torch.equal(x, original)
    y = torch.zeros(2, 4, 4)
    for seed in range(8):  # covers flip and rotation draws
        np.random.seed(seed)
        augment(x, y)
        assert torch.equal(x, original)
