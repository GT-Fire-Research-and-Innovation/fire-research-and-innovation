"""Every file format on disk: .npz with Zstandard members, window archives, TESSERA files.

Level 1 (data). numpy and the zipfile below only, so it loads on any training machine.

.npz files [ours]: window and TESSERA archives compress every .npy member with Zstandard
(zip method 93) at NPZ_COMPRESSLEVEL. On a quantized float32 field, level 6 decompressed
about 1.8x faster than zlib level 1 and packed 16% tighter, for 14% slower writing; on
noisy float32 the sizes match. Python 3.14 reads such zips natively (PEP 784); 3.9-3.13
use the zipfile of the backports.zstd package, a copy of 3.14's (1.3.0 is its last release
for 3.9). np.load cannot open them before 3.14 ("That compression method is not
supported"), so every archive is read through open_npz, which answers archive["name"]
and archive.files like np.load's NpzFile.
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

Daily archives [ours, format version 3], one per fire, hold each day once instead of
each window (DAILY_FORMAT_VERSION in firepred/config/dataset.py says why):

    p<id>.npz
    ├── format_version   scalar, 3
    ├── num_windows      scalar, windows of the archive's own settings (format-2 readers stop
    │                    on the version instead of misreading the file)
    ├── ts_length, ts_interval   the window settings the archive was built with (its default view)
    ├── num_days         scalar D: input days 0 .. D-1 are stored
    ├── block_days       scalar B
    ├── frame_shape      (3,) int32: channels, height, width
    ├── label_sel        scalar (optional, as in format 2)
    ├── static_channels  (k,) int16: channels identical on every day, stored once
    ├── static_c17       (256, 256) float32, one per static channel
    ├── days_c00_000     (n <= B, 256, 256) float32: channel 0, days 0 .. n-1
    ├── days_c00_006, ..., days_c01_000, ...
    ├── target_days      (targets,) int16: the day of each stored next-day target
    ├── targets          (targets, 256, 32) uint8: the targets as bits (np.packbits on the last axis)
    └── target_dtype     the targets' dtype as written by format 2 (float32)

open_fire() answers archive["data_007"], archive["label_007"], archive["num_windows"] and
archive.files for a daily archive exactly as for a format-2 one, so every reader works
unchanged: window k starts on day k x ts_interval, takes ts_length days and predicts the
next day, the same windows data/windows.build_fire_windows makes. Another ts_length or
ts_interval reads the same file, as long as it holds every target day those windows need.

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

Zone files [ours, format version 1], zones/p<id>.npz, built by data/zones_master.py
from a window archive's targets (see data/fire_tracks.py and data/fire_zones.py):

    format_version, num_windows, label_sel, first_label_day    scalars
    setting_<name>        the SplitSettings used (scale_km, link_pixels, ...)
    events_<field>        (fires,) one column per FireEvent field
    zones_000             (256, 256) int16, the fire owning each area of window 0
    target_owners_000     (256, 256) int16, the fire of each new target pixel (-1: none)
    zones_001, target_owners_001, ...
"""

import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from functools import cached_property
from pathlib import Path

import numpy as np

from firepred.config.dataset import (
    ARCHIVE_FORMAT_VERSION,
    ARCHIVE_FORMAT_VERSIONS,
    CROP_SIZE,
    DAILY_BLOCK_DAYS,
    DAILY_COMPRESSLEVEL,
    DAILY_FORMAT_VERSION,
    NIGHT_I4_CHANNEL,
    NPZ_COMPRESSLEVEL,
    TESSERA_BANDS,
    TESSERA_FORMAT_VERSION,
    TESSERA_QUANTIZATION,
    TESSERA_QUANTIZATIONS,
    ZONE_FORMAT_VERSION,
)
from firepred.data.fire_records import FireEvent, SplitSettings, ZoneFile

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


def open_fire(source, length=None, interval=None):
    """Open a fire's windows from a format-2 archive, an unzipped p<id>/ folder or a daily archive.

    All three answer archive["data_000"], archive["label_000"], archive["num_windows"]
    and archive.files the same way. A daily archive slices its windows from its days,
    by default with the window settings it was built with.

    Args:
        source (str | Path): a p<id>.npz archive or unzipped p<id>/ folder.
        length (int | None): input days per window of a daily archive (None: its own).
        interval (int | None): days between window starts of a daily archive (None: its own).

    Returns:
        NpzArchive | _FireFolder | DailyArchive: the open fire.

    Raises:
        ValueError: if length or interval is given for a format-2 archive or folder,
            whose windows are fixed when built.
    """
    fire = open_windows(source, length, interval)
    if not isinstance(fire, DailyArchive) and (length is not None or interval is not None):
        if isinstance(fire, NpzArchive):
            fire.close()
        raise ValueError(f"{source} stores fixed windows; only daily archives take other window settings")
    return fire


def open_windows(source, length=None, interval=None):
    """Open a fire's windows as a run reads them: a daily archive cut at the run's -ts/-it.

    A daily archive stores days, not windows, so it gives windows of any length
    and stride whose days it holds; training cuts them at its own -ts/-it.
    Format-2 archives and unzipped folders store fixed windows and open as
    built, whatever is asked (their shapes are checked when read).

    Args:
        source (str | Path): a p<id>.npz archive or unzipped p<id>/ folder.
        length (int | None): input days per window for a daily archive (None: its own).
        interval (int | None): days between window starts for a daily archive (None: its own).

    Returns:
        NpzArchive | _FireFolder | DailyArchive: the open fire.

    Raises:
        ValueError: if a daily archive lacks a day or target those windows need.
    """
    if Path(source).is_dir():
        return _FireFolder(source)
    fire = open_npz(source)
    if DAILY_MARKER in fire.files:
        try:
            return DailyArchive(fire, source, length, interval)
        except Exception:
            fire.close()
            raise
    return fire


def index_fires(sources, length=None, interval=None):
    """List every window of every fire, checking each archive on the way.

    The result is FireDataset's index: entry i says which fire and which window
    number dataset item i is, so shuffling it mixes windows from all fires. Each
    archive must use the current format and have a data_### and label_### member
    for every window. Only the zip directory is read, never the window arrays.

    Args:
        sources (list[Path]): fire archives or folders (find_fire_sources).
        length (int | None): input days per window; daily archives are cut at it (None: their own).
        interval (int | None): days between window starts for daily archives (None: their own).

    Returns:
        list[tuple[Path, int]]: (fire source, window number) for every window, in order.

    Raises:
        RuntimeError: on an old-format archive, an unknown format version or a missing member.
    """
    index = []
    for npz_path in sources:
        with open_windows(npz_path, length, interval) as archive:
            # Reject old archives rather than silently loading the whole
            # `data` member and defeating lazy decompression.
            if "num_windows" not in archive.files:
                raise RuntimeError(f"{npz_path} uses the old NPZ format. "
                                   "Expected num_windows plus data_###/label_### members.")
            number_of_windows = int(np.asarray(archive["num_windows"]).item())
            if "format_version" in archive.files:
                format_version = int(np.asarray(archive["format_version"]).item())
                if format_version not in ARCHIVE_FORMAT_VERSIONS:
                    raise RuntimeError(f"{npz_path} has unsupported format version {format_version}; "
                                       f"expected one of {ARCHIVE_FORMAT_VERSIONS}")
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
    # Daily archives would unzip into raw days several times their size, for no
    # faster reads: they are read directly.
    if any(is_daily_archive(path) for paths in sources.values() for path in paths):
        print(f"{data_root} holds daily archives; reading them directly (nothing to unzip)")
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


# ===== Daily archives (format version 3) ===== #


# A member only daily archives have; open_fire tells the formats apart by it.
DAILY_MARKER = "num_days"


def is_daily_archive(path):
    """Tell whether a .npz file is a daily archive, from its zip directory alone."""
    with zipfile.ZipFile(path) as bundle:
        return f"{DAILY_MARKER}.npy" in bundle.namelist()


def block_member(channel, first_day):
    """Name the member holding one channel's block of days that starts on first_day."""
    return f"days_c{channel:02d}_{first_day:03d}"


def static_member(channel):
    """Name the member holding a channel that is the same on every day."""
    return f"static_c{channel:02d}"


def window_starts(last_target_day, length, interval):
    """Give the first day of every window, the same windows windows.build_fire_windows makes.

    build_fire_windows starts a window every `interval` days while the day after
    its last input day still exists: with N days, starts 0, interval, ... below
    N - length. last_target_day is N - 1.

    Args:
        last_target_day (int): the last day a target can fall on (-1 without days).
        length (int): input days per window.
        interval (int): days between window starts.

    Returns:
        tuple[int, ...]: the start days.
    """
    return tuple(range(0, max(last_target_day - length + 1, 0), interval))


@dataclass(frozen=True)
class DailyFire:
    """One fire's days and next-day targets, each stored once: what a daily archive holds.

    days_from_windows rebuilds one from format-2 windows and write_daily_archive
    stores it. read_daily_fire gives it back whole.

    Attributes:
        days (np.ndarray): (channels, num_days, H, W) float32 input frames, day 0 first.
        target_days (np.ndarray): (targets,) int16, the day each target is for.
        targets (np.ndarray): (targets, H, W) bool next-day targets.
        target_dtype (str): the dtype format 2 stored the targets in.
    """

    days: np.ndarray
    target_days: np.ndarray
    targets: np.ndarray
    target_dtype: str = "float32"


def days_from_windows(count, read, length, interval):
    """Rebuild one frame per day from a fire's overlapping windows, checking every copy is identical.

    Window k holds days k x interval .. k x interval + length - 1 and the target
    for the day after. A day that sits in several windows must be bit-identical
    in all of them (it is, for windows built by build_fire_windows); anything
    else means the windows do not come from one frame per day and is refused.
    Windows are read one at a time, so a long fire never sits in memory twice.

    Args:
        count (int): the number of windows.
        read (Callable[[int], tuple[np.ndarray, np.ndarray]]): window k -> (inputs
            (C, length, H, W) float32, target (H, W) of 0/1 values).
        length (int): input days per window.
        interval (int): days between window starts.

    Returns:
        DailyFire: the days and targets.

    Raises:
        ValueError: on a window of another shape or dtype, a day whose copies differ,
            a target that is not one 0/1 map, or an interval that skips days.
    """
    if interval > length and count > 1:
        raise ValueError(f"-it {interval} > -ts {length} leaves days in no window; they cannot be rebuilt")
    days = filled = None
    target_days, targets, target_dtype = [], [], "float32"
    for window in range(count):
        inputs, target = read(window)
        inputs, target = np.asarray(inputs), np.asarray(target)
        if inputs.dtype != np.float32 or inputs.ndim != 4 or inputs.shape[1] != length:
            raise ValueError(f"window {window}: inputs {inputs.dtype} {inputs.shape}; expected float32 (C, {length}, H, W)")
        if days is None:
            channels, _, height, width = inputs.shape
            num_days = (count - 1) * interval + length
            days = np.empty((channels, num_days, height, width), dtype=np.float32)
            filled = np.zeros(num_days, dtype=bool)
            target_dtype = str(target.dtype)
        elif inputs.shape[0] != days.shape[0] or inputs.shape[2:] != days.shape[2:]:
            raise ValueError(f"window {window}: inputs {inputs.shape} differ from window 0's frames")
        start = window * interval
        for offset in range(length):
            day = start + offset
            if not filled[day]:
                days[:, day] = inputs[:, offset]
                filled[day] = True
            elif not np.array_equal(days[:, day].view(np.uint32), inputs[:, offset].view(np.uint32)):
                raise ValueError(f"window {window} holds another day {day} than an earlier window: "
                                 "these windows were not cut from one frame per day")
        if target.shape != days.shape[2:] or str(target.dtype) != target_dtype or not np.isin(target, (0, 1)).all():
            raise ValueError(f"window {window}: the target must be one {days.shape[2:]} map of 0/1 in {target_dtype}")
        target_days.append(start + length)
        targets.append(target == 1)
    if days is None:
        days = np.zeros((0, 0, 0, 0), dtype=np.float32)
        return DailyFire(days, np.zeros(0, dtype=np.int16), np.zeros((0, 0, 0), dtype=bool), target_dtype)
    return DailyFire(days, np.asarray(target_days, dtype=np.int16), np.stack(targets), target_dtype)


def write_daily_archive(save_path, fire, length, interval, label_sel=None, block_days=DAILY_BLOCK_DAYS,
                        compresslevel=DAILY_COMPRESSLEVEL):
    """Write one fire's days and targets as a daily archive.

    A channel that is bit-identical on every day (terrain, land cover) is
    stored once; the others in blocks of block_days days. The targets are
    stored as bits. length and interval are recorded as the archive's own
    window settings, the windows it gives when none are asked for.

    Args:
        save_path (str | Path): where to write p<id>.npz.
        fire (DailyFire): the days and targets (days_from_windows).
        length (int): input days per window it was built with.
        interval (int): days between window starts it was built with.
        label_sel (int | None): the label rule that built the targets; None leaves it out.
        block_days (int): days per block.
        compresslevel (int): Zstandard level.

    Returns:
        Path: save_path.

    Raises:
        RuntimeError: if the written file does not read back with the same windows count.
    """
    channels, num_days, height, width = fire.days.shape
    static = [channel for channel in range(channels)
              if num_days and all(np.array_equal(fire.days[channel, day].view(np.uint32),
                                                 fire.days[channel, 0].view(np.uint32)) for day in range(num_days))]
    last_target_day = int(fire.target_days.max()) if len(fire.target_days) else -1
    payload = {
        "format_version": np.asarray(DAILY_FORMAT_VERSION, dtype=np.int16),
        "num_windows": np.asarray(len(window_starts(last_target_day, length, interval)), dtype=np.int32),
        "ts_length": np.asarray(length, dtype=np.int16),
        "ts_interval": np.asarray(interval, dtype=np.int16),
        DAILY_MARKER: np.asarray(num_days, dtype=np.int32),
        "block_days": np.asarray(block_days, dtype=np.int16),
        "frame_shape": np.asarray((channels, height, width), dtype=np.int32),
    }
    if label_sel is not None:
        payload["label_sel"] = np.asarray(label_sel, dtype=np.int16)
    payload["static_channels"] = np.asarray(static, dtype=np.int16)
    for channel in range(channels):
        if channel in static:
            payload[static_member(channel)] = fire.days[channel, 0]
            continue
        for first_day in range(0, num_days, block_days):
            payload[block_member(channel, first_day)] = fire.days[channel, first_day:first_day + block_days]
    payload["target_days"] = np.asarray(fire.target_days, dtype=np.int16)
    payload["targets"] = np.packbits(fire.targets, axis=-1)
    payload["target_dtype"] = np.asarray(fire.target_dtype)
    write_npz(save_path, payload, compresslevel=compresslevel)

    with open_fire(save_path) as archive:
        if int(archive["num_windows"]) != int(payload["num_windows"]):
            raise RuntimeError(f"Failed to verify {save_path}: it does not read back its own windows")
    return Path(save_path)


class DailyArchive:
    """Read a daily archive as the windows of one window setting, like a format-2 archive.

    archive["data_007"] slices window 7 from the stored days, archive["label_007"]
    unpacks its target, and archive["num_windows"] counts the windows; any other
    name reads the stored member. Blocks decompressed for one window are kept
    while the next windows still need them, so reading every window in order
    (statistics) decompresses each block about once.

    Attributes:
        files (list[str]): the window names plus every stored member.
        length (int): input days per window.
        interval (int): days between window starts.
        starts (tuple[int, ...]): the first day of every window.
    """

    def __init__(self, npz, source, length=None, interval=None):
        """Read the archive's header and check it holds every day and target the windows need.

        Only the small header members are read here. A window setting the file
        cannot give is refused now, before any window is asked for.

        Args:
            npz (NpzArchive): the open archive.
            source (str | Path): its path, for messages.
            length (int | None): input days per window (None: the archive's own).
            interval (int | None): days between window starts (None: the archive's own).

        Raises:
            ValueError: if the archive lacks a target day or input day those windows need.
        """
        self._npz, self.source = npz, source
        read = lambda name: int(np.asarray(npz[name]).item())  # noqa: E731
        self.length = int(length) if length is not None else read("ts_length")
        self.interval = int(interval) if interval is not None else read("ts_interval")
        self.num_days, self.block_days = read(DAILY_MARKER), read("block_days")
        self.frame_shape = tuple(int(n) for n in npz["frame_shape"])
        self.static_channels = frozenset(int(c) for c in npz["static_channels"])
        self.target_days = [int(day) for day in npz["target_days"]]
        self._target_index = {day: index for index, day in enumerate(self.target_days)}
        last_target_day = max(self.target_days, default=-1)
        self.starts = window_starts(last_target_day, self.length, self.interval)
        missing = [start + self.length for start in self.starts if start + self.length not in self._target_index]
        if missing or (self.starts and self.starts[-1] + self.length > self.num_days):
            raise ValueError(f"{source} cannot give -ts {self.length} -it {self.interval} windows: it lacks target "
                             f"days {missing[:5]} (it holds {self.num_days} input days and targets for days "
                             f"{self.target_days[:1]}..{self.target_days[-1:]}; it was built with "
                             f"-ts {read('ts_length')} -it {read('ts_interval')})")
        self._targets = None
        self._blocks, self._static = {}, {}
        names = ["format_version", "num_windows"] + [f"{kind}_{window:03d}" for window in range(len(self.starts))
                                                     for kind in ("data", "label")]
        self.files = names + [name for name in npz.files if name not in names]

    def _block(self, channel, first_day):
        """Decompress one channel's block of days once while it is needed."""
        key = (channel, first_day)
        if key not in self._blocks:
            self._blocks[key] = self._npz[block_member(channel, first_day)]
        return self._blocks[key]

    def window(self, index):
        """Slice one window from the stored days.

        Each channel copies its days from the one or two blocks they sit in;
        static channels repeat their one frame. Blocks wholly before this window
        are dropped first, since windows are read in increasing order.

        Args:
            index (int): the window number.

        Returns:
            np.ndarray: (channels, length, H, W) float32.
        """
        start = self.starts[index]
        stop = start + self.length
        # Blocks wholly before this window are no longer needed by later ones.
        self._blocks = {key: block for key, block in self._blocks.items() if key[1] + self.block_days > start}
        channels, height, width = self.frame_shape
        out = np.empty((channels, self.length, height, width), dtype=np.float32)
        for channel in range(channels):
            self.channel_days(channel, start, stop, out=out[channel])
        return out

    def channel_days(self, channel, start, stop, out=None):
        """Read one channel's days from start up to stop, decompressing only the blocks they sit in.

        A static channel repeats its one frame. Fire tracking reads the night I4
        days this way without slicing all 27 channels of a window.

        Args:
            channel (int): the stored channel.
            start (int): the first day.
            stop (int): the day after the last.
            out (np.ndarray | None): a (stop - start, H, W) float32 array to fill.

        Returns:
            np.ndarray: (stop - start, H, W) float32.
        """
        if out is None:
            out = np.empty((stop - start, *self.frame_shape[1:]), dtype=np.float32)
        if channel in self.static_channels:
            if channel not in self._static:
                self._static[channel] = self._npz[static_member(channel)]
            out[:] = self._static[channel]
            return out
        day = start
        while day < stop:
            first_day = day - day % self.block_days
            block = self._block(channel, first_day)
            end = min(first_day + len(block), stop)
            out[day - start:end - start] = block[day - first_day:end - first_day]
            day = end
        return out

    def target(self, index):
        """Unpack the next-day target of window `index`, in the dtype format 2 stored it in."""
        if self._targets is None:
            self._targets = self._npz["targets"]
        packed = self._targets[self._target_index[self.starts[index] + self.length]]
        dtype = str(np.asarray(self._npz["target_dtype"]).item())
        return np.unpackbits(packed, axis=-1, count=self.frame_shape[2]).astype(dtype)

    def __getitem__(self, key):
        """Answer a member name the way a format-2 archive would.

        data_### and label_### are sliced from the days and targets, and
        num_windows counts this setting's windows. Any other name reads the
        stored member, like NpzArchive.

        Args:
            key (str): a member name, with or without .npy.

        Returns:
            np.ndarray: the window, target or member.

        Raises:
            KeyError: for a window number past the last window, or no such member.
        """
        name = key[:-4] if key.endswith(".npy") else key
        kind, _, number = name.partition("_")
        if kind in ("data", "label") and number.isdigit():
            index = int(number)
            if index >= len(self.starts):
                raise KeyError(f"{key} is not a window of {self.source}")
            return self.window(index) if kind == "data" else self.target(index)
        if name == "num_windows":
            return np.asarray(len(self.starts), dtype=np.int32)
        return self._npz[name]

    def close(self):
        """Close the archive and drop the decompressed blocks."""
        self._blocks, self._static = {}, {}
        self._npz.close()

    def __enter__(self):
        """Start a `with` block."""
        return self

    def __exit__(self, *exc_info):
        """Close the archive at the end of a `with` block."""
        self.close()
        return False


def read_daily_fire(source):
    """Read a daily archive's days and targets whole.

    Tests and checks use this. Training reads one window at a time instead.

    Args:
        source (str | Path): a daily p<id>.npz archive.

    Returns:
        DailyFire: every day and target of the fire.
    """
    with open_fire(source) as archive:
        channels, height, width = archive.frame_shape
        days = np.empty((channels, archive.num_days, height, width), dtype=np.float32)
        for channel in range(channels):
            if channel in archive.static_channels:
                days[channel] = archive[static_member(channel)]
            else:
                for first_day in range(0, archive.num_days, archive.block_days):
                    block = archive[block_member(channel, first_day)]
                    days[channel, first_day:first_day + len(block)] = block
        packed = archive["targets"]
        targets = np.unpackbits(packed, axis=-1, count=width).astype(bool) if len(packed) else np.zeros((0, 0, 0), bool)
        return DailyFire(days, np.asarray(archive["target_days"], dtype=np.int16), targets,
                         str(np.asarray(archive["target_dtype"]).item()))


def convert_to_daily(source, save_path, length, interval, block_days=DAILY_BLOCK_DAYS,
                     compresslevel=DAILY_COMPRESSLEVEL):
    """Convert one format-2 fire archive into a daily archive and prove every window reads back bit-identical.

    The source is read one window at a time. After writing, every window and
    target is read again from both files and compared bit for bit; any
    difference deletes the new file and raises.

    Args:
        source (str | Path): the format-2 p<id>.npz archive (or unzipped folder).
        save_path (str | Path): where the daily archive goes.
        length (int): input days per window of the source.
        interval (int): days between window starts of the source.
        block_days (int): days per block.
        compresslevel (int): Zstandard level.

    Returns:
        int: the number of windows checked.

    Raises:
        ValueError: if the source windows cannot be rebuilt from one frame per day.
        RuntimeError: if a window or target does not read back bit-identical.
    """
    with open_fire(source) as archive:
        count = int(np.asarray(archive["num_windows"]).item())
        label_sel = int(np.asarray(archive["label_sel"]).item()) if "label_sel" in archive.files else None
        fire = days_from_windows(count, lambda k: (archive[f"data_{k:03d}"], archive[f"label_{k:03d}"]),
                                 length, interval)
    write_daily_archive(save_path, fire, length, interval, label_sel, block_days, compresslevel)
    del fire
    with open_fire(source) as old, open_fire(save_path) as new:
        problems = [] if int(new["num_windows"]) == count else ["num_windows"]
        for key in (f"{kind}_{k:03d}" for k in range(count) for kind in ("data", "label")):
            a, b = np.asarray(old[key]), np.asarray(new[key])
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                problems.append(key)
    if problems:
        Path(save_path).unlink(missing_ok=True)
        raise RuntimeError(f"{save_path} did not read back like {source}: {problems[:5]}")
    return count


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


# ===== Zone files ===== #


# ----- Reading what fire tracking needs from a window archive ----- #


def open_daily_windows(source, length=None):
    """Open a fire's windows that start every day, the windows fire tracking reads.

    A daily archive gives them for `length`, whatever -it it was built with, as
    long as it holds those target days. Format-2 archives and folders open as
    they are: their windows must already start every day (zones_master checks).

    Args:
        source (str | Path): a p<id>.npz archive (format 2 or daily) or unzipped p<id>/ folder.
        length (int | None): input days per window; None opens a daily archive with its own settings.

    Returns:
        NpzArchive | _FireFolder | DailyArchive: the open fire.
    """
    return open_windows(source, length, None if length is None else 1)


def read_fire_labels(source, length=None):
    """Read only the targets of a fire archive, as booleans, with the label rule that built them.

    The input windows are never decompressed, so this takes moments per fire.
    Fire tracking needs nothing else from an archive except the seed heat
    (read_seed_heat).

    Args:
        source (str | Path): a p<id>.npz archive (format 2 or daily) or unzipped p<id>/ folder.
        length (int | None): input days per window; a daily archive is read at -it 1 for it.

    Returns:
        tuple[np.ndarray, int]: ((windows, H, W) bool targets, label_sel or -1 when unrecorded).

    Raises:
        ValueError: if a target is not one (H, W) map.
    """
    with open_daily_windows(source, length) as archive:
        count = int(np.asarray(archive["num_windows"]).item())
        label_sel = int(np.asarray(archive["label_sel"]).item()) if "label_sel" in archive.files else -1
        labels = [np.asarray(archive[f"label_{window:03d}"]) > 0 for window in range(count)]
    # Accept the harmless leading singleton of older exports, like read_window.
    labels = [label[0] if label.ndim == 3 and label.shape[0] == 1 else label for label in labels]
    if any(label.ndim != 2 for label in labels):
        raise ValueError(f"{source}: fire tracking needs one (H, W) target per window")
    if not labels:
        return np.zeros((0, CROP_SIZE, CROP_SIZE), dtype=bool), label_sel
    return np.stack(labels), label_sel


def read_seed_heat(source, channel=NIGHT_I4_CHANNEL, length=None):
    """Read the night I4 running maximum of a fire's first window, one map per input day.

    These are the only days before the first target, so fire tracking looks
    for its heat seeds here. An archive without windows gives no days. A daily
    archive reads only this channel's blocks.

    Args:
        source (str | Path): a p<id>.npz archive (format 2 or daily) or unzipped p<id>/ folder.
        channel (int): the stored channel of night I4.
        length (int | None): input days per window; a daily archive is read at -it 1 for it.

    Returns:
        np.ndarray: (days, H, W) float32.
    """
    with open_daily_windows(source, length) as archive:
        if int(np.asarray(archive["num_windows"]).item()) == 0:
            return np.zeros((0, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        if isinstance(archive, DailyArchive):
            return archive.channel_days(channel, 0, archive.length)
        return np.asarray(archive["data_000"][channel], dtype=np.float32)


# ----- Writing and reading zone files ----- #


def event_columns(events):
    """Lay out fire events as one array per field, so they store without pickling.

    Text fields become fixed-width strings and the others int32. An empty
    tuple of events still gives every column, with the right type.

    Args:
        events (tuple[FireEvent, ...]): the fires.

    Returns:
        dict[str, np.ndarray]: {field name: (fires,) array}.
    """
    return {
        field.name: np.array([getattr(event, field.name) for event in events],
                             dtype=np.str_ if field.type is str else np.int32)
        for field in fields(FireEvent)
    }


def write_zone_file(save_path, zone_file, compresslevel=NPZ_COMPRESSLEVEL):
    """Write one archive's zones next to it, one member per window like the archive itself.

    The settings and every fire event are stored with the zones, so a zone
    file says how it was made. The file is read back once to check it.

    Args:
        save_path (str | Path): where to write zones/p<id>.npz.
        zone_file (ZoneFile): the zones (fire_zones.zone_file).
        compresslevel (int): Zstandard level.

    Returns:
        Path: save_path.

    Raises:
        RuntimeError: if the written file does not read back the same.
    """
    payload = {
        "format_version": np.asarray(zone_file.format_version, dtype=np.int16),
        "num_windows": np.asarray(len(zone_file.zones), dtype=np.int32),
        "label_sel": np.asarray(zone_file.label_sel, dtype=np.int16),
        "first_label_day": np.asarray(zone_file.first_label_day, dtype=np.int16),
    }
    for name, value in asdict(zone_file.settings).items():
        payload[f"setting_{name}"] = np.asarray(value)
    for name, column in event_columns(zone_file.events).items():
        payload[f"events_{name}"] = column
    for window in range(len(zone_file.zones)):
        payload[f"zones_{window:03d}"] = zone_file.zones[window].astype(np.int16)
        payload[f"target_owners_{window:03d}"] = zone_file.target_owners[window].astype(np.int16)
    write_npz(save_path, payload, compresslevel=compresslevel)

    saved = read_zone_file(save_path)
    same = (len(saved.zones) == len(zone_file.zones) and saved.events == tuple(zone_file.events)
            and saved.settings == zone_file.settings
            and all(np.array_equal(a, b) for a, b in zip(saved.zones, zone_file.zones))
            and all(np.array_equal(a, b) for a, b in zip(saved.target_owners, zone_file.target_owners)))
    if not same:
        raise RuntimeError(f"Zone file did not read back correctly: {save_path}")
    return Path(save_path)


def read_zone_file(path):
    """Read one archive's zone file back into a ZoneFile.

    Every member is read, since a zone file is small. A file written by another
    format version is refused rather than misread.

    Args:
        path (str | Path): a zones/p<id>.npz file.

    Returns:
        ZoneFile: zones, target owners, fires and settings.

    Raises:
        RuntimeError: on another format version.
    """
    with open_npz(path) as archive:
        version = int(np.asarray(archive["format_version"]).item())
        if version != ZONE_FORMAT_VERSION:
            raise RuntimeError(f"{path} has zone format version {version}; expected {ZONE_FORMAT_VERSION}")
        count = int(np.asarray(archive["num_windows"]).item())
        settings = SplitSettings(**{field.name: archive[f"setting_{field.name}"].item()
                                    for field in fields(SplitSettings)})
        columns = {field.name: archive[f"events_{field.name}"] for field in fields(FireEvent)}
        events = tuple(FireEvent(**{name: column[row].item() for name, column in columns.items()})
                       for row in range(len(columns["event_id"])))
        zones = [archive[f"zones_{window:03d}"] for window in range(count)]
        target_owners = [archive[f"target_owners_{window:03d}"] for window in range(count)]
        label_sel = int(np.asarray(archive["label_sel"]).item())
        first_label_day = int(np.asarray(archive["first_label_day"]).item())
    empty = np.zeros((0, CROP_SIZE, CROP_SIZE), dtype=np.int16)
    return ZoneFile(
        zones=np.stack(zones) if zones else empty,
        target_owners=np.stack(target_owners) if target_owners else empty,
        events=events,
        settings=settings,
        label_sel=label_sel,
        first_label_day=first_label_day,
        format_version=version,
    )


def zone_paths(zones_dir, fire_sources):
    """Map every fire to its zone file, {"p<id>": path}; a missing one fails now, before training."""
    zones_dir = Path(zones_dir)
    names = [fire_name(path) for path in fire_sources]
    missing = [f"{name}.npz" for name in names if not (zones_dir / f"{name}.npz").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} fires have no zone file in {zones_dir}: {missing[:10]}")
    return {name: zones_dir / f"{name}.npz" for name in names}


@dataclass(frozen=True)
class ZoneHeader:
    """Hold what a zone file says about itself, without its per-window maps.

    Training reads the headers of every zone file before the first window, to
    check that each file fits its archive and that all were built alike. The
    maps themselves are read one window at a time (read_zone_window).

    Attributes:
        num_windows (int): windows in the file, one per day.
        first_label_day (int): the day of window 0's target (the window length).
        label_sel (int): the label rule of the targets the zones came from.
        settings (SplitSettings): the thresholds the zones were built with.
    """

    num_windows: int
    first_label_day: int
    label_sel: int
    settings: SplitSettings


def read_zone_header(path):
    """Read a zone file's scalars and settings, leaving its per-window maps unread.

    Only the small members are decompressed, so checking every fire's file
    before training takes a moment. A file of another format version is refused.

    Args:
        path (str | Path): a zones/p<id>.npz file.

    Returns:
        ZoneHeader: the file's window count, first target day, label rule and settings.

    Raises:
        RuntimeError: on another format version.
    """
    with open_npz(path) as archive:
        version = int(np.asarray(archive["format_version"]).item())
        if version != ZONE_FORMAT_VERSION:
            raise RuntimeError(f"{path} has zone format version {version}; expected {ZONE_FORMAT_VERSION}")
        return ZoneHeader(
            num_windows=int(np.asarray(archive["num_windows"]).item()),
            first_label_day=int(np.asarray(archive["first_label_day"]).item()),
            label_sel=int(np.asarray(archive["label_sel"]).item()),
            settings=SplitSettings(**{field.name: archive[f"setting_{field.name}"].item()
                                      for field in fields(SplitSettings)}),
        )


def read_zone_window(path, window):
    """Read one window's zone maps: (zones, target owners), each (256, 256) int16."""
    with open_npz(path) as archive:
        return archive[f"zones_{window:03d}"], archive[f"target_owners_{window:03d}"]
