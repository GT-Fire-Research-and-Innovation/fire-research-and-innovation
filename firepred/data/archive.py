"""Every file format on disk: .npz with Zstandard members, window archives, TESSERA files.

Level 1 (data). numpy and the zipfile below only, so it loads on any training machine.

.npz files [ours]: window and TESSERA archives compress every .npy member with Zstandard
(zip method 93) at NPZ_COMPRESSLEVEL. On a quantized float32 field, level 6 decompressed
about 1.8x faster than zlib level 1 and packed 16% tighter, for 14% slower writing; on
noisy float32 the sizes match. Python 3.14 reads such zips natively (PEP 784); 3.10-3.13
use the zipfile of the backports.zstd package, a copy of 3.14's. np.load cannot open them
before 3.14 ("That compression method is not supported"), so every archive is read
through open_npz, which answers archive["name"] and archive.files like np.load's NpzFile.
Older zlib archives read the same way, since each member records its own method.

Window archives [ours, format version 2], one per fire:

    p<id>.npz
    ├── format_version   scalar, 2
    ├── num_windows      scalar
    ├── label_sel        scalar, the label rule that built the targets (optional)
    ├── data_000         (27, T, 256, 256) float32 input window
    ├── label_000        (256, 256) next-day target (or (T, 256, 256))
    ├── data_001, label_001, ...

Each window is its own member, so reading one window decompresses only that window.
unzip_data_root() can unpack every archive once into a p<id>/ folder of the same .npy
members (byte-identical arrays, no decompression on every read).

TESSERA files [ours], tessera/p<id>.npz, built by data/tessera.py:

    embedding      (128, 256 k, 256 k) float16, 0 where coverage is 0 (k = 1 by default)
    coverage       (256 k, 256 k) float16, fraction of 10 m pixels with data
    format_version, plus provenance metadata (fire_id, fire_year, tessera_year, crs,
    transform, bounds, bounds_lonlat, source_store, model_version, geotessera_version,
    code_version, day_files, and cells_per_pixel when k > 1)

With quantization "int8" (TESSERA_QUANTIZATION) the embedding member is replaced by

    embedding_int8     (128, 256 k, 256 k) int8 codes, 0 where coverage is 0
    embedding_scale    (128,) float32, one step of each band (its range / 254)
    embedding_offset   (128,) float32, the middle of each band's range
    quantization       "int8"

value = code x scale + offset. read_tessera returns float16 for both kinds, so
nothing after it changes. A reader that predates int8 finds no "embedding" member
and fails, instead of using the codes as values.
"""

import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path

import numpy as np

from firepred.config.dataset import (
    ARCHIVE_FORMAT_VERSION,
    CROP_SIZE,
    NPZ_COMPRESSLEVEL,
    TESSERA_BANDS,
    TESSERA_FORMAT_VERSION,
    TESSERA_QUANTIZATION,
    TESSERA_QUANTIZATIONS,
)

if sys.version_info >= (3, 14):
    import zipfile
else:
    try:
        from backports.zstd import zipfile
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Window and TESSERA archives are Zstandard-compressed. Python < 3.14 needs the backport: "
            "pip install 'backports.zstd; python_version<\"3.14\"'"
        ) from error

ZIP_ZSTANDARD = zipfile.ZIP_ZSTANDARD


# ===== .npz files with Zstandard members ===== #


class NpzArchive:
    """Read an .npz file member by member, like np.load's NpzFile, with Zstandard support.

    Only the zip directory is read on opening. Each archive["name"] decompresses
    that one member, so a window archive never has to be loaded whole.

    Attributes:
        files (list[str]): the member names without .npy, e.g. ["num_windows", "data_000"].
    """

    def __init__(self, path):
        """Open the zip and list its .npy members; no member is decompressed yet."""
        self._zip = zipfile.ZipFile(path)
        self.files = [name[:-4] for name in self._zip.namelist() if name.endswith(".npy")]

    def __getitem__(self, key):
        """Decompress and return one member.

        Like NpzFile, the name may be given with or without .npy. A name that is
        not in the archive is a KeyError.

        Args:
            key (str): the member name, e.g. "data_000".

        Returns:
            np.ndarray: the member's array.

        Raises:
            KeyError: if the archive has no such member.
        """
        name = key if key.endswith(".npy") else f"{key}.npy"
        try:
            member = self._zip.open(name)
        except KeyError:
            raise KeyError(f"{key} is not a file in the archive") from None
        with member:
            return np.lib.format.read_array(member, allow_pickle=False)

    def close(self):
        """Close the zip file; members already returned stay valid."""
        self._zip.close()

    def __enter__(self):
        """Start a `with` block."""
        return self

    def __exit__(self, *exc_info):
        """Close the archive at the end of a `with` block."""
        self.close()
        return False


def open_npz(path):
    """Open an .npz file for reading, Zstandard or zlib members alike."""
    return NpzArchive(path)


def write_npz(path, arrays, compresslevel=NPZ_COMPRESSLEVEL):
    """Write arrays as an .npz with one Zstandard-compressed .npy member each, in the given order."""
    with zipfile.ZipFile(path, "w", compression=ZIP_ZSTANDARD, compresslevel=compresslevel, allowZip64=True) as bundle:
        for name, value in arrays.items():
            with bundle.open(f"{name}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)
    return path


# ===== Window archives ===== #


# Written into each unzipped folder: the archive (path, size, mtime) it came from.
_UNZIP_MARKER = ".source.json"


# ----- Writing ----- #


def write_fire_archive(save_path, data, labels, label_sel=None, compresslevel=NPZ_COMPRESSLEVEL):
    """Write one fire's windows into a single archive, one member per window.

    The archive is a zip of .npy members: a format version, the window count,
    the label rule, then data_000/label_000, data_001/label_001, and so on.
    Storing each window separately lets training read one window without
    unpacking the whole fire. The file is read back once to check it is complete.

    Args:
        save_path (str | Path): where to write p<id>.npz.
        data (np.ndarray): inputs, shape (windows, 27, T, 256, 256).
        labels (np.ndarray): next-day targets, shape (windows, 256, 256).
        label_sel (int | None): the label rule that built the targets, recorded
            in the archive; None leaves it out.
        compresslevel (int): Zstandard level (higher packs tighter, writes slower).

    Returns:
        Path: save_path.

    Raises:
        ValueError: if data and labels have different window counts.
        RuntimeError: if the written file does not read back correctly.
    """
    if len(data) != len(labels):
        raise ValueError(f"Data/label window mismatch: {len(data)} versus {len(labels)}")

    # Archive contents: two small header members (format version, window count),
    # the optional label rule, then one data_### and one label_### member per window.
    number_of_windows = len(data)
    payload = {
        "format_version": np.asarray(ARCHIVE_FORMAT_VERSION, dtype=np.int16),
        "num_windows": np.asarray(number_of_windows, dtype=np.int32),
    }
    if label_sel is not None:
        # Records which label rule built the targets; readers ignore it.
        payload["label_sel"] = np.asarray(label_sel, dtype=np.int16)

    for window_idx in range(number_of_windows):
        suffix = f"{window_idx:03d}"
        payload[f"data_{suffix}"] = data[window_idx]
        payload[f"label_{suffix}"] = labels[window_idx]

    write_npz(save_path, payload, compresslevel=compresslevel)

    # Check the member index and the tiny count member, without decompressing any window again.
    with open_npz(save_path) as archive:
        saved_count = int(archive["num_windows"])
        expected_keys = {f"{kind}_{window_idx:03d}" for window_idx in range(number_of_windows)
                         for kind in ("data", "label")}
        missing_keys = expected_keys.difference(archive.files)
    if saved_count != number_of_windows or missing_keys:
        raise RuntimeError(f"Failed to verify {save_path}: saved_count={saved_count}, missing={sorted(missing_keys)}")
    return Path(save_path)


# ----- Finding and opening fires ----- #


def fire_name(source):
    """Name a fire by its p<id>.npz archive or p<id>/ folder: p<id> (also its TESSERA file's name)."""
    source = Path(source)
    return source.stem if source.suffix == ".npz" else source.name


def find_fire_sources(split_dir):
    """List a split's fires, each as its archive or its unzipped folder.

    An unzipped folder is used instead of an archive of the same name once it
    holds num_windows.npy (it is faster to read). Hidden names are half-finished
    unzips and are skipped. The order matches sorting the archive names.

    Args:
        split_dir (str | Path): a split folder such as data/train.

    Returns:
        list[Path]: one entry per fire, sorted by fire name ([] if the folder is missing).
    """
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return []
    sources = {}
    for path in split_dir.iterdir():
        if path.name.startswith("."):
            continue
        if path.suffix == ".npz" and path.is_file():
            sources.setdefault(path.stem, path)
        elif path.is_dir() and (path / "num_windows.npy").is_file():
            sources[path.name] = path
    # Same order as sorting the archive names, so window indexes match runs
    # made before unzipped folders existed.
    return [sources[name] for name in sorted(sources, key=lambda n: f"{n}.npz")]


class _FireFolder:
    """Read an unzipped fire folder exactly like an open .npz archive.

    open_npz returns an object that answers archive["data_000"] and
    archive.files. This class gives an unzipped p<id>/ folder (one .npy file per
    member) the same interface, so the reading code works for both.
    """

    def __init__(self, folder):
        """Point the reader at one unzipped fire folder."""
        self.folder = Path(folder)

    @cached_property
    def files(self):
        """List the folder's member names, like NpzFile.files."""
        return [path.stem for path in self.folder.glob("*.npy")]

    def __getitem__(self, key):
        """Load one member's .npy file (pickles refused)."""
        return np.load(self.folder / f"{key}.npy", allow_pickle=False)

    def __enter__(self):
        """Start a `with` block; a folder needs no open file handle."""
        return self

    def __exit__(self, *exc_info):
        """End a `with` block; there is nothing to close."""
        return False


def open_fire(source):
    """Open a p<id>.npz archive or unzipped p<id>/ folder; both answer archive["data_000"] with the same array."""
    if Path(source).is_dir():
        return _FireFolder(source)
    return open_npz(source)


def index_fires(sources):
    """List every window of every fire, checking each archive on the way.

    The result is FireDataset's index: entry i says which fire and which window
    number dataset item i is, so shuffling it mixes windows from all fires. Each
    archive must use the current format and have a data_### and label_### member
    for every window. Only the zip directory is read, never the window arrays.

    Args:
        sources (list[Path]): fire archives or folders (find_fire_sources).

    Returns:
        list[tuple[Path, int]]: (fire source, window number) for every window, in order.

    Raises:
        RuntimeError: on an old-format archive, an unknown format version or a missing member.
    """
    index = []
    for npz_path in sources:
        with open_fire(npz_path) as archive:
            # Reject old archives rather than silently loading the whole
            # `data` member and defeating lazy decompression.
            if "num_windows" not in archive.files:
                raise RuntimeError(f"{npz_path} uses the old NPZ format. "
                                   "Expected num_windows plus data_###/label_### members.")
            number_of_windows = int(np.asarray(archive["num_windows"]).item())
            if "format_version" in archive.files:
                format_version = int(np.asarray(archive["format_version"]).item())
                if format_version != ARCHIVE_FORMAT_VERSION:
                    raise RuntimeError(f"{npz_path} has unsupported format version {format_version}; "
                                       f"expected version {ARCHIVE_FORMAT_VERSION}")
            for window_idx in range(number_of_windows):
                for key in (f"data_{window_idx:03d}", f"label_{window_idx:03d}"):
                    if key not in archive.files:
                        raise RuntimeError(f"{npz_path} is missing member {key}")
                index.append((npz_path, window_idx))
    return index


# ----- Unzipping once for faster reads ----- #


def _archive_signature(archive_path):
    """Describe an archive by path, size and mtime, so a later unzip can tell whether it changed."""
    archive_path = Path(archive_path).resolve()
    stat = archive_path.stat()
    return {"archive": str(archive_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_unzip_marker(folder):
    """Read the marker an unzip left in a fire folder; None (missing or broken) means unzip again."""
    try:
        return json.loads((folder / _UNZIP_MARKER).read_text())
    except (OSError, ValueError):
        return None


def _unzip_fire(archive_path, folder):
    """Unzip one fire archive into its folder, safely.

    The archive is first unpacked into a hidden temporary folder, which is
    renamed into place only when complete. An interrupted unzip therefore never
    looks finished. Only plain .npy members are accepted.

    Args:
        archive_path (Path): the p<id>.npz archive.
        folder (Path): the p<id>/ folder to create.

    Returns:
        Path: folder.

    Raises:
        ValueError: if the archive holds anything other than plain .npy members.
    """
    work = folder.with_name(f".{folder.name}.partial")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = member.filename
            if not name.endswith(".npy") or name.startswith(".") or "/" in name or "\\" in name:
                raise ValueError(f"{archive_path}: unexpected member {name!r}")
            # ZipFile checks each member's CRC as it is read.
            with archive.open(member) as src, open(work / name, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    (work / _UNZIP_MARKER).write_text(json.dumps(_archive_signature(archive_path)))
    if folder.exists():
        shutil.rmtree(folder)
    work.rename(folder)
    return folder


def unzip_data_root(data_root, unzip_root, splits=("train", "val", "test"), workers=None):
    """Unzip every split's archives once, so training reads windows faster.

    Reading an unzipped .npy file skips decompression on every read; the arrays
    are identical. Archives already unzipped (and unchanged) are skipped, stale
    folders are removed, and the work runs in parallel threads. If anything is
    missing or there is too little disk, the original data root is used instead.

    Args:
        data_root (str | Path): the root with the archives.
        unzip_root (str | Path): where the unzipped folders go.
        splits (tuple[str, ...]): which splits to unzip.
        workers (int | None): unzip threads (None picks a number).

    Returns:
        Path: unzip_root when every archive is unzipped, else data_root.
    """
    data_root, unzip_root = Path(data_root), Path(unzip_root)
    # Nothing to do when there are no archives, or when the data root is already unzipped.
    sources = {split: find_fire_sources(data_root / split) for split in splits}
    if not any(sources.values()):
        return data_root
    if any(path.suffix != ".npz" for paths in sources.values() for path in paths):
        print(f"{data_root} already holds unzipped fire folders; using it as is")
        return data_root

    # Plan the work: unzip only archives whose folder is missing or out of date
    # (the marker file records which archive, size and time it came from), and
    # remove folders whose archive no longer exists.
    jobs, stale = [], []
    for split, archives in sources.items():
        split_dir = unzip_root / split
        for archive in archives:
            folder = split_dir / archive.stem
            if _read_unzip_marker(folder) != _archive_signature(archive):
                jobs.append((archive, folder))
        # Folders this function made for fires that data_root no longer has.
        names = {archive.stem for archive in archives}
        if split_dir.is_dir():
            stale += [folder for folder in split_dir.iterdir()
                      if folder.name not in names and (folder / _UNZIP_MARKER).is_file()]

    for folder in stale:
        shutil.rmtree(folder)
    if not jobs:
        print(f"Using unzipped windows in {unzip_root}")
        return unzip_root

    # Unzipped windows take several times the disk of the archives: check the space
    # first and fall back to reading the archives directly if it does not fit.
    needed = 0
    for archive, _ in jobs:
        with zipfile.ZipFile(archive) as zipped:
            needed += sum(member.file_size for member in zipped.infolist())
    try:
        unzip_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(unzip_root).free
    except OSError as error:
        print(f"Cannot write {unzip_root} ({error}); reading the archives directly")
        return data_root
    if needed + (1 << 30) > free:
        print(f"Unzipping needs {needed / 2**30:.1f} GiB but {unzip_root} has "
              f"{free / 2**30:.1f} GiB free; reading the archives directly")
        return data_root

    print(f"Unzipping {len(jobs)} fire archives ({needed / 2**30:.1f} GiB) into {unzip_root}")
    try:
        # Decompression and file writes release the GIL, so threads unzip in parallel.
        with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
            for _ in pool.map(lambda job: _unzip_fire(*job), jobs):
                pass
    except OSError as error:
        print(f"Unzipping failed ({error}); reading the archives directly")
        return data_root
    return unzip_root


# ===== TESSERA files ===== #


# ----- Grid size ----- #


def cells_per_pixel_of(shape, expected=None):
    """Find how many TESSERA cells per VIIRS pixel side a stored map has.

    A map holds k x k cells for every one of the 256 x 256 VIIRS pixels, so its
    side is 256 k. k = 1 is one averaged vector per VIIRS pixel. A map whose
    side is not a whole multiple of 256 is refused.

    Args:
        shape (tuple[int, ...]): the embedding shape (128, S, S) or coverage shape (S, S).
        expected (int | None): the k the caller needs, or None for any.

    Returns:
        int: k.

    Raises:
        ValueError: if the shape is not square, not a multiple of 256, or not the expected k.
    """
    side = shape[-1]
    if len(shape) < 2 or shape[-2] != side or side % CROP_SIZE or side == 0:
        raise ValueError(f"TESSERA map shape {tuple(shape)} is not a square multiple of {CROP_SIZE}")
    if len(shape) == 3 and shape[0] != TESSERA_BANDS:
        raise ValueError(f"TESSERA map shape {tuple(shape)} does not have {TESSERA_BANDS} bands")
    cells = side // CROP_SIZE
    if expected is not None and cells != expected:
        raise ValueError(f"TESSERA map has {cells} cells per VIIRS pixel; this run needs {expected}")
    return cells


# ----- int8 quantization ----- #


def quantize_int8(embedding, has_data):
    """Store an embedding as int8 codes: each band's range in equal steps.

    Each band's range over the cells with data, minimum to maximum, is cut
    into 254 equal steps (codes -127 .. 127), so no value is clipped and
    dequantize_int8 is off by at most half a step. TESSERA bands have heavy
    tails (up to 18 standard deviations from the mean on real maps), so a range
    fixed in standard deviations would clip them. Cells without data get code
    0. Bands are done one at a time, so a 5 x 5 map never has a second full
    float copy. A band with a single value gets step 1 and comes back exactly
    as the offset.

    Args:
        embedding (np.ndarray): (128, S, S) float, NaN allowed only where has_data is False.
        has_data (np.ndarray): (S, S) bool, the cells with data.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (codes (128, S, S) int8,
        scale (128,) float32, offset (128,) float32); value = code x scale + offset.

    Raises:
        ValueError: if a cell with data holds NaN or infinity.
    """
    codes = np.zeros(embedding.shape, dtype=np.int8)
    scale = np.ones(len(embedding), dtype=np.float32)
    offset = np.zeros(len(embedding), dtype=np.float32)
    for band, values in enumerate(embedding):
        kept = values[has_data].astype(np.float32)
        if kept.size == 0:
            continue
        if not np.isfinite(kept).all():
            raise ValueError(f"TESSERA band {band} has a missing value in a cell with data")
        low, high = float(kept.min()), float(kept.max())
        offset[band] = (low + high) / 2
        if high > low:
            scale[band] = (high - low) / 254
        # Codes come from the stored float32 offset and scale; the clip only
        # guards against float rounding at the two ends of the range.
        codes[band][has_data] = np.clip(np.rint((kept - offset[band]) / scale[band]), -127, 127)
    return codes, scale, offset


def dequantize_int8(codes, scale, offset, dtype=np.float16):
    """Turn int8 codes back into values (code x scale + offset), one band at a time to keep memory low."""
    values = np.empty(codes.shape, dtype=dtype)
    for band in range(len(codes)):
        values[band] = codes[band] * scale[band] + offset[band]
    return values


# ----- Writing ----- #


def save_tessera_npz(save_path, embedding, coverage, metadata, quantization=TESSERA_QUANTIZATION):
    """Write one fire's TESSERA file and check that it reads back.

    By default the embedding is stored as float16 (missing cells as 0) to halve
    the file size. Quantization "int8" stores int8 codes plus one scale and
    offset per band instead, about 2.3x smaller again on real 5 x 5 maps
    (quantize_int8). The coverage map and provenance metadata are stored with
    it, all Zstandard-compressed. The file is read back once to make sure it
    is complete.

    Args:
        save_path (str | Path): where to write tessera/p<id>.npz.
        embedding (np.ndarray): (128, 256 k, 256 k) float32, NaN where there is no data
            (k TESSERA cells per VIIRS pixel side; 1 by default).
        coverage (np.ndarray): (256 k, 256 k), the fraction of 10 m pixels with data.
        metadata (dict): extra members (fire id, years, grid, versions).
        quantization (str): "none" (float16) or "int8".

    Returns:
        Path: save_path.

    Raises:
        ValueError: on a wrong shape, an unknown quantization, a value too large
            for float16, or (int8) a missing value in a cell with data.
        RuntimeError: if the written file does not read back correctly.
    """
    cells = cells_per_pixel_of(embedding.shape)
    if coverage.shape != embedding.shape[1:]:
        raise ValueError(f"Unexpected coverage shape {coverage.shape}")
    if quantization not in TESSERA_QUANTIZATIONS:
        raise ValueError(f"Unknown TESSERA quantization {quantization!r}; choose from {TESSERA_QUANTIZATIONS}")

    if quantization == "int8":
        codes, scale, offset = quantize_int8(embedding, np.asarray(coverage) > 0)
        stored = {"embedding_int8": codes, "embedding_scale": scale, "embedding_offset": offset,
                  "quantization": np.asarray(quantization)}
    else:
        # float16 halves the file size; TESSERA values are small enough not to overflow.
        stored = {"embedding": np.nan_to_num(embedding, nan=0.0).astype(np.float16)}
        if not np.isfinite(stored["embedding"]).all():
            raise ValueError(f"{save_path}: embedding overflows float16")

    write_npz(save_path, {
        "format_version": np.asarray(TESSERA_FORMAT_VERSION, dtype=np.int16),
        **stored,
        "coverage": coverage.astype(np.float16),
        **{key: np.asarray(value) for key, value in metadata.items()},
    })

    # Read the file back once to make sure it is complete.
    member = next(iter(stored))
    with open_npz(save_path) as archive:
        if archive[member].shape != (TESSERA_BANDS, CROP_SIZE * cells, CROP_SIZE * cells):
            raise RuntimeError(f"Failed to verify {save_path}")
    return Path(save_path)


# ----- Converting ----- #


# The members that hold the map itself; every other member is provenance metadata.
_TESSERA_MAP_MEMBERS = frozenset(("format_version", "embedding", "embedding_int8", "embedding_scale",
                                  "embedding_offset", "quantization", "coverage"))


def tessera_quantization_of(path):
    """Say how a TESSERA file stores its embedding: "int8" or "none" (float16); only the zip index is read."""
    with open_npz(path) as archive:
        return "int8" if "embedding_int8" in archive.files else "none"


def convert_tessera_npz(source_path, save_path, quantization, metadata=None):
    """Rewrite one TESSERA file with another storage, keeping its coverage and provenance.

    The embedding is read back as float16 (read_tessera) and written again with
    the new quantization, so float16 -> int8 adds only int8's rounding. Every
    provenance member (fire id, years, grid, versions) is kept as it was. A
    file already stored that way is copied unchanged, so nothing is rounded
    twice.

    Args:
        source_path (str | Path): the TESSERA file to convert (not changed).
        save_path (str | Path): where to write the converted file.
        quantization (str): the storage to convert to, "none" (float16) or "int8".
        metadata (dict | None): extra members for a rewritten file, e.g. where it came from.

    Returns:
        Path: save_path.

    Raises:
        ValueError: on an unknown quantization or a malformed source file.
    """
    if quantization not in TESSERA_QUANTIZATIONS:
        raise ValueError(f"Unknown TESSERA quantization {quantization!r}; choose from {TESSERA_QUANTIZATIONS}")
    if tessera_quantization_of(source_path) == quantization:
        shutil.copyfile(source_path, save_path)
        return Path(save_path)
    with open_npz(source_path) as archive:
        provenance = {name: archive[name] for name in archive.files if name not in _TESSERA_MAP_MEMBERS}
    embedding, coverage = read_tessera(source_path)
    return save_tessera_npz(save_path, embedding, coverage, {**provenance, **(metadata or {})},
                            quantization=quantization)


# ----- Reading and model input ----- #


def read_tessera(path, cells_per_pixel=None):
    """Read one fire's TESSERA embedding and coverage.

    int8 files are turned back into float16 here, with 0 where coverage is 0
    as in float16 files, so callers never see how a file was stored. The
    arrays are marked read-only, because one fire's map is shared by all of
    its windows (and by DataLoader worker processes). Nothing may change it.

    Args:
        path (str | Path): a tessera/p<id>.npz file.
        cells_per_pixel (int | None): the k the caller needs, or None for any.

    Returns:
        tuple[np.ndarray, np.ndarray]: (embedding (128, 256 k, 256 k) float16,
        coverage (256 k, 256 k) float16).

    Raises:
        ValueError: if the embedding has the wrong shape or another k.
    """
    path = Path(path)
    with open_npz(path) as archive:
        quantized = "embedding_int8" in archive.files
        embedding = archive["embedding_int8" if quantized else "embedding"]
        coverage = archive["coverage"]
        if quantized:
            scale, offset = archive["embedding_scale"], archive["embedding_offset"]
    try:
        cells_per_pixel_of(embedding.shape, cells_per_pixel)
    except ValueError as error:
        raise ValueError(f"{path}: embedding has shape {embedding.shape}; {error}") from None
    if quantized:
        embedding = dequantize_int8(embedding, scale, offset)
        embedding[:, coverage == 0] = 0
    embedding.flags.writeable = False
    coverage.flags.writeable = False
    return embedding, coverage


def tessera_paths(tessera_dir, fire_sources):
    """Map every fire to its TESSERA file, {"p<id>": path}; a missing one fails now, before training."""
    tessera_dir = Path(tessera_dir)
    names = [fire_name(path) for path in fire_sources]
    missing = [f"{name}.npz" for name in names if not (tessera_dir / f"{name}.npz").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} fires have no TESSERA file in {tessera_dir}: {missing[:10]}")
    return {name: tessera_dir / f"{name}.npz" for name in names}


def load_tessera_maps(tessera_dir, fire_sources, cells_per_pixel=None):
    """Read every fire's TESSERA file once; pass the result to both compute_tessera_stats and FireDataset."""
    return {name: read_tessera(path, cells_per_pixel)
            for name, path in tessera_paths(tessera_dir, fire_sources).items()}


def iter_tessera_maps(tessera_dir, fire_sources, cells_per_pixel=None):
    """Read the fires' TESSERA files one at a time: k-cell maps (about 400 MiB at k = 5) do not fit together."""
    paths = tessera_paths(tessera_dir, fire_sources)
    return (read_tessera(path, cells_per_pixel) for path in paths.values())


def compute_tessera_stats(tessera_maps, min_coverage=0.5):
    """Compute the per-band mean and spread of TESSERA over well-covered pixels.

    These statistics standardize the embeddings before the model sees them.
    Compute them from the training fires only and reuse them for validation
    and test, so nothing is learned from held-out fires. Pixels with less
    coverage than min_coverage are left out.

    Args:
        tessera_maps (Iterable[tuple[np.ndarray, np.ndarray]]): (embedding, coverage) pairs.
        min_coverage (float): the smallest coverage a pixel needs to count.

    Returns:
        dict: {"mean": (128,) float32, "std": (128,) float32, "min_coverage": float}.

    Raises:
        RuntimeError: if no pixel has enough coverage.
    """
    # Accumulate sum and sum of squares over all well-covered pixels of the
    # training fires (float64 to avoid rounding drift), then mean and std per band.
    total = np.zeros(TESSERA_BANDS, dtype=np.float64)
    total_sq = np.zeros(TESSERA_BANDS, dtype=np.float64)
    count = 0
    for embedding, coverage in tessera_maps:
        valid = coverage.astype(np.float32) >= min_coverage
        # Select first, then widen: only the kept pixels are copied to float64.
        values = embedding[:, valid].astype(np.float64)
        total += values.sum(axis=1)
        total_sq += (values ** 2).sum(axis=1)
        count += values.shape[1]
    if count == 0:
        raise RuntimeError("No TESSERA pixels with sufficient coverage")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-12))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "min_coverage": float(min_coverage)}


def static_map(embedding, valid, stats):
    """Build one fire's TESSERA input (129, H, W): standardized embedding (0 where invalid) plus a validity mask."""
    embedding = embedding.astype(np.float32)
    mean = stats["mean"][:, None, None]
    std = stats["std"][:, None, None]
    embedding = np.where(valid, (embedding - mean) / std, 0.0)
    return np.concatenate((embedding, valid[None].astype(np.float32)), axis=0).astype(np.float32)
