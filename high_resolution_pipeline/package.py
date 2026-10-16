"""Write self-describing, versioned patch files for Hugging Face storage."""

import json
import io
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


FORMAT_VERSION = 1


def _npy_bytes(array) -> bytes:
    """Serialize one NumPy member without retaining a full patch archive in RAM."""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


class PatchArchiveWriter:
    """Incrementally write one patch's static TESSERA and temporal examples.

    The resulting `.npz` is a standard ZIP of NPY members readable by
    `numpy.load`. Only one temporal example is serialized at a time, so the
    writer does not need to keep every window for a patch in memory.
    """

    def __init__(self, path, embedding, coverage, valid_mask, metadata):
        self.path = Path(path)
        self.embedding = np.nan_to_num(np.asarray(embedding), nan=0.0, posinf=0.0, neginf=0.0)
        self.coverage = np.nan_to_num(np.asarray(coverage), nan=0.0, posinf=0.0, neginf=0.0)
        self.valid_mask = np.asarray(valid_mask, dtype=np.uint8)
        self.metadata = metadata
        self.window_count = 0
        self.temporary = None
        self.archive = None

    def __enter__(self):
        if self.embedding.ndim != 3 or self.embedding.shape[0] != 128:
            raise ValueError(f"Expected TESSERA shape (128,H,W), got {self.embedding.shape}")
        shape = self.embedding.shape[-2:]
        if self.coverage.shape != shape or self.valid_mask.shape != shape:
            raise ValueError("TESSERA, coverage, and valid mask shapes must match")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, self.temporary = tempfile.mkstemp(
            prefix=self.path.stem + ".", suffix=".npz", dir=self.path.parent
        )
        os.close(handle)
        self.archive = zipfile.ZipFile(self.temporary, mode="w", compression=zipfile.ZIP_DEFLATED)
        self._write("embedding", self.embedding.astype(np.float16))
        self._write("coverage", self.coverage.astype(np.float16))
        self._write("valid_mask", self.valid_mask)
        self._write("metadata", _metadata_array(self.metadata))
        return self

    def _write(self, key, array):
        self.archive.writestr(f"{key}.npy", _npy_bytes(array))

    def add_window(self, data, label, metadata):
        """Append one `(27,T,H,W)` input, `(H,W)` target, and window metadata."""
        data = np.asarray(data)
        label = np.asarray(label)
        shape = self.embedding.shape[-2:]
        if data.ndim != 4 or data.shape[0] != 27 or data.shape[-2:] != shape:
            raise ValueError(f"Invalid window input shape {data.shape}; expected (27,T,{shape[0]},{shape[1]})")
        if label.shape != shape or not np.isfinite(data).all():
            raise ValueError("Window label shape mismatch or input contains non-finite values")
        data = data.astype(np.float16)
        if not np.isfinite(data).all():
            raise ValueError("Input values overflowed float16 during storage conversion")
        label = (label > 0).astype(np.uint8)
        label[self.valid_mask == 0] = 0
        suffix = f"{self.window_count:05d}"
        self._write(f"data_{suffix}", data)
        self._write(f"label_{suffix}", label)
        self._write(f"window_meta_{suffix}", _metadata_array(metadata))
        self.window_count += 1

    def __exit__(self, exc_type, exc_value, traceback):
        if self.archive is not None:
            if exc_type is None:
                self._write("format_version", np.asarray(FORMAT_VERSION, dtype=np.int16))
                self._write("num_windows", np.asarray(self.window_count, dtype=np.int32))
            self.archive.close()
        if exc_type is None and self.temporary is not None:
            os.replace(self.temporary, self.path)
        elif self.temporary and os.path.exists(self.temporary):
            os.remove(self.temporary)
        return False


def patch_archive_window_count(path) -> int:
    """Read the number of temporal examples in a completed patch archive."""
    with np.load(path, allow_pickle=False) as archive:
        version = int(np.asarray(archive["format_version"]).item())
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported patch archive version {version}")
        return int(np.asarray(archive["num_windows"]).item())


def _atomic_npz(path: str | Path, **arrays) -> Path:
    """Write compressed NPZ to a temporary sibling and atomically rename it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _metadata_array(metadata: dict) -> np.ndarray:
    return np.asarray(json.dumps(metadata, sort_keys=True, default=str))


def write_tessera_patch(path, embedding, coverage, valid_mask, metadata) -> Path:
    """Save one static TESSERA patch plus coverage and geographic metadata."""
    embedding = np.asarray(embedding)
    coverage = np.asarray(coverage)
    valid_mask = np.asarray(valid_mask, dtype=np.uint8)
    if embedding.ndim != 3 or embedding.shape[0] != 128:
        raise ValueError(f"Expected TESSERA shape (128,H,W), got {embedding.shape}")
    if coverage.shape != embedding.shape[-2:] or valid_mask.shape != coverage.shape:
        raise ValueError("TESSERA, coverage, and valid mask shapes must match")
    embedding = np.nan_to_num(embedding, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    coverage = np.nan_to_num(coverage, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(embedding).all() or not np.isfinite(coverage).all():
        raise ValueError("TESSERA data and coverage must be finite before packaging")
    return _atomic_npz(
        path,
        format_version=np.asarray(FORMAT_VERSION, dtype=np.int16),
        embedding=embedding.astype(np.float16),
        coverage=coverage.astype(np.float16),
        valid_mask=valid_mask,
        metadata=_metadata_array(metadata),
    )


def write_window_patch(path, data, label, valid_mask, metadata) -> Path:
    """Save one temporal input window and its next-day target for a patch."""
    data = np.asarray(data)
    label = np.asarray(label)
    valid_mask = np.asarray(valid_mask, dtype=np.uint8)
    if data.ndim != 4 or data.shape[0] != 27:
        raise ValueError(f"Expected data shape (27,T,H,W), got {data.shape}")
    expected = data.shape[-2:]
    if label.shape != expected or valid_mask.shape != expected:
        raise ValueError("Input, label, and validity mask spatial shapes must match")
    if not np.isfinite(data).all():
        raise ValueError("Input data contains NaN or infinity")
    label = (label > 0).astype(np.uint8)
    label[valid_mask == 0] = 0
    data = data.astype(np.float16)
    if not np.isfinite(data).all():
        raise ValueError("Input values overflowed float16 during storage conversion")
    return _atomic_npz(
        path,
        format_version=np.asarray(FORMAT_VERSION, dtype=np.int16),
        data=data,
        label=label,
        valid_mask=valid_mask,
        metadata=_metadata_array(metadata),
    )


def write_fire_manifest(path, fire_metadata: dict, grid_metadata: dict, patch_count: int, window_count: int) -> Path:
    """Write a readable JSON inventory for one processed fire."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": FORMAT_VERSION,
        "fire": fire_metadata,
        "grid": grid_metadata,
        "patch_count": int(patch_count),
        "window_patch_count": int(window_count),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)
    return path


def write_dataset_card(dataset_dir, config) -> Path:
    """Write a Hugging Face dataset README documenting the patch format."""
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    settings = config.to_dict()
    card = f'''---
task_categories:
- image-to-image
language:
- en
tags:
- wildfire
- satellite-imagery
- tessera
---

# TS-SatFire high-resolution patches

This dataset is generated by the separate `high_resolution_pipeline` in the
TS-SatFire repository. It stores VIIRS, auxiliary channels, labels, and
TESSERA on a common projected grid, split into fixed-size spatial patches.

## Generation settings

- Target grid spacing: **{settings['resolution_m']} m per pixel**
- Patch size: **{settings['patch_size_pixels']} × {settings['patch_size_pixels']} pixels**
- Patch stride: **{settings['patch_stride_pixels']} pixels**
- Input time series: **{settings['time_series_length_days']} days**
- Window stride: **{settings['window_stride_days']} days**
- Source dataset: `{settings['source_repo']}`

## Layout and contents

Files are organized as `train|val|test/p<fire_id>/manifest.json` and
`train|val|test/p<fire_id>/patches/<patch_id>.npz`. Each patch NPZ stores
`embedding` with shape `(128, patch_height, patch_width)`, `coverage`, a shared
`valid_mask`, and temporal members named `data_#####` and `label_#####`.
Temporal inputs have shape `(27, time, patch_height, patch_width)` and are
stored as float16; labels and masks are uint8. Scalar metadata members describe
the fire, patch coordinates, CRS, affine transform, and window indices.

Continuous input channels are bilinearly resampled; labels, land cover, and
angular channels use nearest-neighbor. Fine-grid interpolation of coarse
measurements aligns arrays but does not add source detail. TESSERA features are
streamed and aggregated to the target grid. A training loader for this format
must pair each patch with its matching TESSERA data; the legacy 256 × 256
loader does not understand these archives.
'''
    path = dataset_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path
