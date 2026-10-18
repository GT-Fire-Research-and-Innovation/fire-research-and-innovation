"""Per-window transforms: angle handling, land-cover encoding, augmentation consistency."""

import numpy as np
import torch

from firepred.config.dataset import BASE_MODEL_CHANNELS, LANDCOVER_CHANNEL, NORMALIZATION_MEAN, NORMALIZATION_STD
from firepred.data.transforms import Normalize, augment, landcover_one_hot_matrix, preprocess


def window(landcover):
    """A (27, 2, 4, 4) window with wind direction 90 degrees and one land-cover class."""
    x = torch.zeros(27, 2, 4, 4)
    x[12] = 90.0  # wind direction: sin(90) = 1
    x[LANDCOVER_CHANNEL] = landcover
    return x


def test_preprocess_gives_43_channels_with_sine_angles():
    """preprocess gives 43 channels with sin(angle) in the angle channels."""
    out = preprocess(window(3.0), landcover_one_hot_matrix())
    assert out.shape == (BASE_MODEL_CHANNELS, 2, 4, 4)
    assert torch.allclose(out[12], torch.ones(2, 4, 4))


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
    normalize = Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD, dont_normalize_idc=[12, 18, 24, LANDCOVER_CHANNEL])
    normalized = normalize(x)
    assert torch.equal(x, original) and not torch.equal(normalized, original)
    preprocess(x, landcover_one_hot_matrix())
    assert torch.equal(x, original)
    y = torch.zeros(2, 4, 4)
    for seed in range(8):  # covers flip and rotation draws
        np.random.seed(seed)
        augment(x, y)
        assert torch.equal(x, original)
