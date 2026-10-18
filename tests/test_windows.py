"""GeoTIFFs -> windows: crop, NaN handling, next-day targets."""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from firepred.prep.windows import binary_mask, build_fire_windows  # noqa: E402


def test_binary_mask_treats_nan_inf_and_nodata_as_background():
    mask = binary_mask(np.array([np.nan, np.inf, -1.0, 0.0, 0.5, 2.0]))
    assert mask.tolist() == [False, False, False, False, True, True]


def test_windows_and_targets(tmp_path):
    import synth

    local_map = synth.fire_geotiffs(tmp_path)  # 8 days
    windows = build_fire_windows("fire", local_map, length=6, interval=1, label_sel=0)
    x, y = windows.inputs, windows.targets
    assert x.shape == (2, 27, 6, 256, 256)  # windows start on days 0 and 1; day 7 is the last target
    assert y.shape == (2, 256, 256)
    assert np.isfinite(x).all() and set(np.unique(y)) <= {0.0, 1.0}
    # Running maximum: I4 (channel 3) never decreases across a window's days.
    assert (np.diff(x[:, 3], axis=1) >= 0).all()


def test_too_few_days_gives_no_windows_and_says_why(tmp_path):
    import synth

    windows = build_fire_windows("fire", synth.fire_geotiffs(tmp_path, n_days=3), length=6, interval=1)
    assert windows.count == 0 and windows.inputs.shape == (0, 27, 6, 256, 256)
    assert any("not enough" in note for note in windows.notes)
