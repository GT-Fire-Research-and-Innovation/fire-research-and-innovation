"""Deterministic synthetic inputs for the tests (data roots, GeoTIFFs, fake TESSERA).

Everything here is written with numpy/zipfile/rasterio directly, never with
the code under test, so both trees read byte-identical files.
"""

import zipfile
from pathlib import Path

import numpy as np

SIZE = 256
STORED_CHANNELS = 27
DEGREE_CHANNELS = (12, 18, 24)
LANDCOVER_CHANNEL = 21


def _write_npz(path, arrays, compressed):
    """Write .npy members like np.savez(_compressed), member order fixed."""
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for key, value in arrays.items():
            with archive.open(f"{key}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)


def window(rng, ts_length, positive_fraction):
    """One (27, T, 256, 256) input window and its (256, 256) next-day label."""
    data = rng.normal(20.0, 10.0, size=(STORED_CHANNELS, ts_length, SIZE, SIZE)).astype(np.float32)
    for channel in DEGREE_CHANNELS:
        data[channel] = rng.uniform(0.0, 360.0, size=(ts_length, SIZE, SIZE)).astype(np.float32)
    data[LANDCOVER_CHANNEL] = rng.integers(0, 18, size=(ts_length, SIZE, SIZE)).astype(np.float32)
    label = (rng.random((SIZE, SIZE)) < positive_fraction).astype(np.float32)
    return data, label


def fire_archive(path, rng, ts_length, n_windows, compressed=False, empty_window=None):
    """Write one fire archive of `n_windows` random windows (window `empty_window` has no fire)."""
    arrays = {
        "format_version": np.asarray(2, dtype=np.int16),
        "num_windows": np.asarray(n_windows, dtype=np.int32),
        "label_sel": np.asarray(0, dtype=np.int16),
    }
    for index in range(n_windows):
        data, label = window(rng, ts_length, 0.0 if index == empty_window else 0.02)
        arrays[f"data_{index:03d}"] = data
        arrays[f"label_{index:03d}"] = label
    _write_npz(path, arrays, compressed)


def tessera_file(path, rng, cells_per_pixel=1):
    """Write one fire's TESSERA file: (128, 256 k, 256 k) float16 embedding, coverage, top rows empty.

    k > 1 repeats each VIIRS pixel's values over its k x k cells, then perturbs
    the cells so they differ, and writes the file uncompressed (fast to write).
    """
    embedding = rng.normal(0.0, 0.3, size=(128, SIZE, SIZE)).astype(np.float16)
    coverage = rng.random((SIZE, SIZE)).astype(np.float16)
    coverage[:20] = 0
    embedding[:, :20] = 0
    if cells_per_pixel > 1:
        k = cells_per_pixel
        embedding = embedding.repeat(k, axis=1).repeat(k, axis=2)
        embedding += (0.05 * rng.standard_normal(embedding.shape[1:])).astype(np.float16)
        coverage = coverage.repeat(k, axis=0).repeat(k, axis=1)
        coverage[: 20 * k + 1, : 2 * k] = 0  # some cells empty inside otherwise covered pixels
    _write_npz(path, {"embedding": embedding, "coverage": coverage}, compressed=cells_per_pixel == 1)


# Fire layout of a data root: split -> [(fire name, number of windows)].
QUICK_FIRES = {
    "train": [("p20000001", 2), ("p20000002", 1)],
    "val": [("p20000003", 1)],
    "test": [("pUS_2021_AA0001", 1), ("p20210002", 1)],
}
E2E_FIRES = {
    "train": [("p20000001", 2), ("p20000002", 1)],
    "val": [("p20000003", 2)],
    "test": [("pUS_2021_AA0001", 1), ("p20210002", 1)],
}
# CSV order differs from sorted order; one id has no archive.
TEST_CSV_IDS = ["20210002", "US_2021_MISSING", "US_2021_AA0001"]


def data_root(root, fires, ts_length=6, seed=0, cells_per_pixel=1):
    """Write <root>/{train,val,test}/p<id>.npz, <root>/tessera/p<id>.npz and a test CSV."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    (root / "tessera").mkdir(parents=True, exist_ok=True)
    for split, entries in fires.items():
        (root / split).mkdir(parents=True, exist_ok=True)
        for position, (name, n_windows) in enumerate(entries):
            fire_archive(
                root / split / f"{name}.npz",
                rng,
                ts_length,
                n_windows,
                compressed=position % 2 == 1,
                empty_window=0 if split == "val" else None,
            )
            tessera_file(root / "tessera" / f"{name}.npz", rng, cells_per_pixel)
    csv_path = root.parent / f"{root.name}_test.csv"
    csv_path.write_text("Id,area\n" + "".join(f"{fire_id},1\n" for fire_id in TEST_CSV_IDS))
    return root, csv_path


# ----- GeoTIFFs for the GeoTIFF -> window step ----- #

GEO_SIZE = 400


def _write_tif(path, array, transform):
    """Write a float32 GeoTIFF in EPSG:32610 with the given transform."""
    import rasterio
    from rasterio.crs import CRS

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype="float32",
        crs=CRS.from_epsg(32610),
        transform=transform,
    ) as dst:
        dst.write(array.astype(np.float32))


def fire_geotiffs(root, n_days=8, seed=1):
    """Write VIIRS_Day / VIIRS_Night / FirePred GeoTIFFs for one fire.

    Edge cases: NaN and negative nodata in AF/BA, a missing night file, a
    5-band and a 1-band night file, a 7-band day file (no BA band), a night
    file of a different size on the same grid, Inf in FirePred.
    Returns local_map {remote name: local path}.
    """
    from rasterio.transform import from_origin

    root = Path(root)
    rng = np.random.default_rng(seed)
    transform = from_origin(500000.0, 4200000.0, 375.0, 375.0)
    local_map = {}
    for day in range(n_days):
        stem = f"2020-08-{day + 1:02d}"
        n_day_bands = 7 if day == 5 else 8
        day_array = rng.normal(300.0, 20.0, size=(n_day_bands, GEO_SIZE, GEO_SIZE)).astype(np.float32)
        af = (rng.random((GEO_SIZE, GEO_SIZE)) < 0.01 * (day + 1)).astype(np.float32)
        af[rng.random((GEO_SIZE, GEO_SIZE)) < 0.01] = np.nan
        af[rng.random((GEO_SIZE, GEO_SIZE)) < 0.01] = -1.0
        day_array[6] = af
        if n_day_bands == 8:
            ba = (rng.random((GEO_SIZE, GEO_SIZE)) < 0.02 * (day + 1)).astype(np.float32)
            ba[rng.random((GEO_SIZE, GEO_SIZE)) < 0.01] = np.nan
            day_array[7] = ba
        day_array[0, :5, :5] = np.nan
        day_path = root / "VIIRS_Day" / f"{stem}_VIIRS_Day.tif"
        _write_tif(day_path, day_array, transform)
        local_map[f"fire/VIIRS_Day/{stem}_VIIRS_Day.tif"] = str(day_path)

        night_path = root / "VIIRS_Night" / f"{stem}_VIIRS_Night.tif"
        if day != 2:
            if day == 3:
                night = rng.normal(290.0, 10.0, size=(5, GEO_SIZE, GEO_SIZE))
            elif day == 4:
                night = rng.normal(290.0, 10.0, size=(1, GEO_SIZE, GEO_SIZE))
            elif day == 6:
                night = rng.normal(290.0, 10.0, size=(2, GEO_SIZE + 20, GEO_SIZE + 20))
            else:
                night = rng.normal(290.0, 10.0, size=(2, GEO_SIZE, GEO_SIZE))
            _write_tif(night_path, night, transform)
            local_map[f"fire/VIIRS_Night/{stem}_VIIRS_Night.tif"] = str(night_path)

        pred = rng.normal(10.0, 5.0, size=(19, GEO_SIZE, GEO_SIZE)).astype(np.float32)
        pred[3, :3, :3] = np.inf
        pred[4, 10:12, 10:12] = np.nan
        pred_path = root / "FirePred" / f"{stem}_FirePred.tif"
        _write_tif(pred_path, pred, transform)
        local_map[f"fire/FirePred/{stem}_FirePred.tif"] = str(pred_path)
    return local_map


class FakeGeoTessera:
    """Stands in for geotessera.GeoTesseraZarr: yields coarse UTM blocks."""

    url = "fake://tessera"
    model_version = "fake"

    def __init__(self, seed=2):
        """A fake store whose blocks are seeded from `seed` and the year."""
        self.seed = seed

    def iter_region(self, bbox, year, strip_rows=128):
        """Yield (block (rows, cols, 128), transform, crs) strips of 750 m pixels with some NaN pixels."""
        from rasterio.transform import from_origin

        rng = np.random.default_rng(self.seed + int(year))
        # 750 m pixels over the synthetic crop (west 548000, north 4152000).
        rows_total, cols = 140, 140
        for row0 in range(0, rows_total, strip_rows // 2):
            rows = min(strip_rows // 2, rows_total - row0)
            block = rng.normal(0.0, 0.2, size=(rows, cols, 128)).astype(np.float32)
            block[rng.random((rows, cols)) < 0.05] = np.nan
            transform = from_origin(546000.0, 4154000.0 - row0 * 750.0, 750.0, 750.0)
            yield block, transform, "EPSG:32610"
