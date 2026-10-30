"""GeoTIFFs -> windows: crop, FirePred grid alignment, NaN handling, next-day targets."""

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from firepred.data.windows import binary_mask, build_fire_windows  # noqa: E402


def test_binary_mask_treats_nan_inf_and_nodata_as_background():
    """NaN, infinity, nodata (-1) and 0 are background; positive values are fire."""
    mask = binary_mask(np.array([np.nan, np.inf, -1.0, 0.0, 0.5, 2.0]))
    assert mask.tolist() == [False, False, False, False, True, True]


def test_windows_and_targets(tmp_path):
    """An 8-day fire gives two 6-day windows with finite inputs, binary targets and a non-decreasing I4 maximum."""
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
    """A fire with too few days gives an empty FireWindows whose notes say why."""
    import synth

    windows = build_fire_windows("fire", synth.fire_geotiffs(tmp_path, n_days=3), length=6, interval=1)
    assert windows.count == 0 and windows.inputs.shape == (0, 27, 6, 256, 256)
    assert any("not enough" in note for note in windows.notes)


# ----- FirePred on another grid than VIIRS ----- #

VIIRS_ORIGIN = (500000.0, 4200000.0)  # synth.fire_geotiffs: 375 m pixels from this corner
VIIRS_PIXEL, COARSE_PIXEL = 375.0, 500.0


def coarse_firepred(tmp_path):
    """synth's fire with every FirePred file replaced by one on a 500 m grid over the same ground.

    Band 0 holds the pixel centre's x coordinate in km (a linear field bilinear
    reproduces exactly), land cover holds one class per 10 km column strip, and
    the angle bands hold 355 west of x = 596 km and 5 east of it.
    """
    import synth
    from rasterio.transform import from_origin

    local_map = synth.fire_geotiffs(tmp_path)
    n = int(synth.GEO_SIZE * VIIRS_PIXEL / COARSE_PIXEL)  # 300 pixels: same 150 km extent
    x_km = (VIIRS_ORIGIN[0] + (np.arange(n) + 0.5) * COARSE_PIXEL) / 1000.0
    pred = np.zeros((19, n, n), dtype=np.float32)
    pred[0] = x_km[None, :]
    pred[13] = 1 + (np.floor(x_km - x_km[0]) // 10 % 17)[None, :]
    for band in (4, 10, 16):
        pred[band] = np.where(x_km < 596.0, 355.0, 5.0)[None, :]
    transform = from_origin(*VIIRS_ORIGIN, COARSE_PIXEL, COARSE_PIXEL)
    for name, path in local_map.items():
        if "/FirePred/" in name:
            synth._write_tif(Path(path), pred, transform)
    return local_map


def test_same_grid_compares_crs_pixel_size_and_origin():
    """Rasters line up only with the same CRS, pixel size and origin; their sizes may differ."""
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    from firepred.data.windows import same_grid

    def profile(pixel=375.0, x=500000.0, epsg=32610):
        """Build a minimal rasterio profile with a north-up grid.

        Only the CRS and transform matter to same_grid.
        """
        return {"crs": CRS.from_epsg(epsg), "transform": from_origin(x, 4200000.0, pixel, pixel)}

    assert same_grid(profile(), profile())
    assert same_grid(profile(x=500000.0 + 1.0), profile())  # under a hundredth of a pixel
    assert not same_grid(profile(pixel=500.0), profile())
    assert not same_grid(profile(x=500000.0 + 375.0), profile())
    assert not same_grid(profile(epsg=32611), profile())


def test_matching_firepred_is_cropped_by_index(tmp_path):
    """FirePred on the VIIRS grid is cropped by index, untouched: no resampling note, values as in the file."""
    import synth

    from firepred.data.windows import read_day_frame, read_tiff

    local_map = synth.fire_geotiffs(tmp_path)
    day_file = sorted(path for name, path in local_map.items() if "VIIRS_Day" in name)[0]
    frame = read_day_frame(day_file)
    pred, _ = read_tiff(day_file.replace("VIIRS_Day", "FirePred"))
    expected = np.nan_to_num(pred[:, 128:384, 128:384], nan=0.0, posinf=0.0, neginf=0.0)
    assert not any("resampled" in note for note in frame.notes)
    np.testing.assert_array_equal(frame.image[8:], expected)


def test_coarser_firepred_is_resampled_onto_the_viirs_crop(tmp_path):
    """FirePred on a 500 m grid is resampled so stored pixel (r, c) holds FirePred at that VIIRS pixel's ground."""
    from firepred.data.windows import build_fire_windows, read_day_frame

    local_map = coarse_firepred(tmp_path)
    day_file = sorted(path for name, path in local_map.items() if "VIIRS_Day" in name)[0]
    frame = read_day_frame(day_file)
    assert any("resampled onto the VIIRS grid" in note for note in frame.notes)

    # Continuous band: x of each VIIRS crop pixel centre, exactly (bilinear of a linear field).
    viirs_x_km = (VIIRS_ORIGIN[0] + (128 + np.arange(256) + 0.5) * VIIRS_PIXEL) / 1000.0
    np.testing.assert_allclose(frame.image[8], np.broadcast_to(viirs_x_km, (256, 256)), atol=1e-3)
    # Index cropping would have started at x = 564 km, 16 km east of the VIIRS crop's 548 km.
    assert abs(frame.image[8, 0, 0] - 548.1875) < 1e-3

    # Land cover stays whole classes (nearest), angles never average to the opposite direction.
    landcover = frame.image[21]
    assert np.array_equal(landcover, np.round(landcover)) and set(np.unique(landcover)) <= set(range(1, 18))
    angles = frame.image[[12, 18, 24]]
    assert ((angles >= 350.0) | (angles <= 10.0)).all()

    windows = build_fire_windows("fire", local_map, length=6, interval=1, label_sel=0)
    assert windows.count == 2 and np.isfinite(windows.inputs).all()
