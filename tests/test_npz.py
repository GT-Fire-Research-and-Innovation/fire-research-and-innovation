"""Archives are written with Zstandard level 6 members and read back on any Python, old zlib ones too."""

import io

import numpy as np
import pytest

import synth
from firepred.config.dataset import NPZ_COMPRESSLEVEL
from firepred.data.archive import (
    ZIP_ZSTANDARD,
    open_fire,
    open_npz,
    read_tessera,
    save_tessera_npz,
    write_fire_archive,
    write_npz,
    zipfile,
)


def methods(path):
    """{member: zip compression method} of a zip file."""
    with zipfile.ZipFile(path) as bundle:
        return {info.filename: info.compress_type for info in bundle.infolist()}


def test_level_is_zstandard_six():
    """The archive level is Zstandard 6, recorded once in config."""
    assert NPZ_COMPRESSLEVEL == 6 and ZIP_ZSTANDARD == 93


def test_window_archives_are_zstandard_and_read_back_exactly(tmp_path):
    """Every member of a window archive is Zstandard; open_fire returns the arrays that went in."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 27, 2, 8, 8)).astype(np.float32)
    labels = (rng.random((2, 8, 8)) < 0.3).astype(np.float32)
    path = write_fire_archive(tmp_path / "p1.npz", data, labels, label_sel=0)
    assert set(methods(path).values()) == {ZIP_ZSTANDARD}
    with open_fire(path) as archive:
        assert int(archive["num_windows"]) == 2 and int(archive["label_sel"]) == 0
        np.testing.assert_array_equal(archive["data_001"], data[1])
        np.testing.assert_array_equal(archive["label_000.npy"], labels[0])  # NpzFile also takes ".npy"
        with pytest.raises(KeyError):
            archive["data_002"]


def test_tessera_files_are_zstandard(tmp_path):
    """TESSERA files use the same Zstandard members and read back as float16."""
    rng = np.random.default_rng(1)
    embedding = rng.normal(size=(128, 256, 256)).astype(np.float32)
    embedding[:, :3] = np.nan
    coverage = rng.random((256, 256)).astype(np.float32)
    path = save_tessera_npz(tmp_path / "p1.npz", embedding, coverage, {"fire_id": "1", "day_files": ["a", "b"]})
    assert set(methods(path).values()) == {ZIP_ZSTANDARD}
    stored, cover = read_tessera(path)
    assert stored.dtype == np.float16 and not stored[:, :3].any()
    np.testing.assert_array_equal(cover, coverage.astype(np.float16))
    with open_npz(path) as archive:
        assert list(archive["day_files"]) == ["a", "b"]


def test_old_zlib_archives_still_read(tmp_path):
    """An archive written with zlib (as before) opens through the same reader with the same arrays."""
    path = tmp_path / "p1.npz"
    synth.fire_archive(path, np.random.default_rng(2), ts_length=1, n_windows=1, compressed=True)
    assert set(methods(path).values()) == {zipfile.ZIP_DEFLATED}
    with np.load(path) as reference, open_fire(path) as archive:
        assert archive.files == reference.files
        np.testing.assert_array_equal(archive["data_000"], reference["data_000"])


def test_level_shrinks_quantized_data_and_reads_from_memory():
    """Level 6 packs sensor-like (quantized) values tighter than level 1; archives also open from memory."""
    smooth = (np.round(np.cumsum(np.random.default_rng(3).standard_normal(200_000)) * 10) / 10).astype(np.float32)
    sizes = {}
    for level in (1, NPZ_COMPRESSLEVEL):
        buffer = io.BytesIO()
        write_npz(buffer, {"x": smooth}, compresslevel=level)
        sizes[level] = buffer.tell()
        buffer.seek(0)
        with open_npz(buffer) as archive:
            np.testing.assert_array_equal(archive["x"], smooth)
    assert sizes[NPZ_COMPRESSLEVEL] < sizes[1]
