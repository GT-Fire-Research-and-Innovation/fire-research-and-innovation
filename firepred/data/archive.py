"""The window archive format: one file (or folder) per fire, one member per window.

Level 1 (data). Written by prep/ and hub/ while building the dataset; read
by data/dataset.py while training.

A fire's windows live in p<id>.npz [ours, format version 2]:

    p<id>.npz
    ├── format_version   scalar, 2
    ├── num_windows      scalar
    ├── label_sel        scalar, the label rule that built the targets (optional)
    ├── data_000         (27, T, 256, 256) float32 input window
    ├── label_000        (256, 256) next-day target (or (T, 256, 256))
    ├── data_001, label_001, ...

Each window is its own zip member, so reading one window decompresses only
that window. unzip_data_root() can unpack every archive once into a p<id>/
folder of the same .npy members (byte-identical arrays, no zlib on every read).
"""

import json
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path

import numpy as np

from firepred.config.data import ARCHIVE_FORMAT_VERSION, NPZ_COMPRESSLEVEL

# Written into each unzipped folder: the archive (path, size, mtime) it came from.
_UNZIP_MARKER = ".source.json"


# ----- Writing ----- #


def write_fire_archive(
    save_path,
    data,
    labels,
    label_sel=None,
    compresslevel=NPZ_COMPRESSLEVEL,
):
    """Save each temporal window as its own compressed NPZ member.

    `data` is (windows, C, T, H, W), `labels` (windows, H, W). Unlike saving
    `data` as one five-dimensional member, this layout lets a Dataset
    decompress only the requested window. The dictionary values below are
    views, so building `payload` does not duplicate the fire in memory.
    Returns the path written (its `num_windows` member holds the count).
    """
    if len(data) != len(labels):
        raise ValueError(
            f"Data/label window mismatch: {len(data)} versus "
            f"{len(labels)}"
        )

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

    # np.savez_compressed's own writer with an explicit zlib level: the
    # same .npy members and arrays, only the compressed bytes differ.
    with zipfile.ZipFile(
        save_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compresslevel,
        allowZip64=True,
    ) as zip_file:
        # Write each member as its own compressed .npy entry; readers can then open one
        # window without decompressing the others.
        for key, value in payload.items():
            with zip_file.open(f"{key}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(
                    member,
                    np.asanyarray(value),
                    allow_pickle=False,
                )

    # This checks the ZIP member index and tiny metadata member without
    # decompressing every data window again.
    with np.load(save_path, allow_pickle=False) as archive:
        saved_count = int(archive["num_windows"])
        expected_keys = {
            key
            for window_idx in range(number_of_windows)
            for key in (
                f"data_{window_idx:03d}",
                f"label_{window_idx:03d}",
            )
        }
        missing_keys = expected_keys.difference(archive.files)

    if saved_count != number_of_windows or missing_keys:
        raise RuntimeError(
            f"Failed to verify {save_path}: saved_count={saved_count}, "
            f"missing={sorted(missing_keys)}"
        )

    return Path(save_path)


# ----- Finding and opening fires ----- #


def fire_name(source):
    """Return p<id> for either a p<id>.npz archive or a p<id>/ folder."""
    source = Path(source)
    return source.stem if source.suffix == ".npz" else source.name


def find_fire_sources(split_dir):
    """List a split's fires, each as its archive or its unzipped folder.

    A folder counts once it holds num_windows.npy and is used instead of an
    archive of the same name. Hidden names (unzip work folders) are skipped.
    """
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return []
    # One entry per fire. An unzipped folder wins over an archive of the same name
    # (faster to read); hidden names are half-finished unzips and are skipped.
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
    """Read an unzipped p<id>/ folder through np.load's NpzFile interface."""

    def __init__(self, folder):
        self.folder = Path(folder)

    @cached_property
    def files(self):
        return [path.stem for path in self.folder.glob("*.npy")]

    def __getitem__(self, key):
        return np.load(self.folder / f"{key}.npy", allow_pickle=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def open_fire(source):
    """Open a fire archive or unzipped folder for `with ... as archive`."""
    # Both kinds answer archive[key] with the same numpy array.
    if Path(source).is_dir():
        return _FireFolder(source)
    return np.load(source, allow_pickle=False)


# ----- Unzipping once for faster reads ----- #


def _archive_signature(archive_path):
    archive_path = Path(archive_path).resolve()
    stat = archive_path.stat()
    return {
        "archive": str(archive_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_unzip_marker(folder):
    try:
        return json.loads((folder / _UNZIP_MARKER).read_text())
    except (OSError, ValueError):
        return None


def _unzip_fire(archive_path, folder):
    """Unzip one archive into a hidden work folder, then rename it to folder."""
    # Unzip into a hidden temporary folder first, then rename it into place, so an
    # interrupted unzip never looks like a finished one.
    work = folder.with_name(f".{folder.name}.partial")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = member.filename
            if (
                not name.endswith(".npy")
                or name.startswith(".")
                or "/" in name
                or "\\" in name
            ):
                raise ValueError(f"{archive_path}: unexpected member {name!r}")
            # ZipFile checks each member's CRC as it is read.
            with archive.open(member) as src, open(work / name, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    (work / _UNZIP_MARKER).write_text(json.dumps(_archive_signature(archive_path)))
    if folder.exists():
        shutil.rmtree(folder)
    work.rename(folder)


def unzip_data_root(
    data_root,
    unzip_root,
    splits=("train", "val", "test"),
    workers=None,
):
    """Unzip each split's archives once into unzip_root; return the root to load.

    Returns unzip_root once every data_root/<split>/p<id>.npz has a matching
    unzip_root/<split>/p<id>/ folder. Folders already unzipped from the same
    archive (same path, size and modification time) are reused, so this is
    cheap after the first call. Returns data_root unchanged, and FireDataset
    reads the archives directly, when data_root holds no archives (it is
    already unzipped) or when unzip_root is not writable or lacks the space.
    [ours]
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
    jobs = []
    stale = []
    for split, archives in sources.items():
        split_dir = unzip_root / split
        for archive in archives:
            folder = split_dir / archive.stem
            if _read_unzip_marker(folder) != _archive_signature(archive):
                jobs.append((archive, folder))
        # Folders this function made for fires that data_root no longer has.
        names = {archive.stem for archive in archives}
        if split_dir.is_dir():
            stale += [
                folder
                for folder in split_dir.iterdir()
                if folder.name not in names and (folder / _UNZIP_MARKER).is_file()
            ]

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
        print(
            f"Unzipping needs {needed / 2**30:.1f} GiB but {unzip_root} has "
            f"{free / 2**30:.1f} GiB free; reading the archives directly"
        )
        return data_root

    print(
        f"Unzipping {len(jobs)} fire archives ({needed / 2**30:.1f} GiB) "
        f"into {unzip_root}"
    )
    try:
        # zlib and file writes release the GIL, so threads unzip in parallel.
        with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
            for _ in pool.map(lambda job: _unzip_fire(*job), jobs):
                pass
    except OSError as error:
        print(f"Unzipping failed ({error}); reading the archives directly")
        return data_root
    return unzip_root
