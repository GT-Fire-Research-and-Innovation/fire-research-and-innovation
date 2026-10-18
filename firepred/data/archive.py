"""The window archive format: one file (or folder) per fire, one member per window.

Level 1 (data). Written by prep/ while building the dataset; read
by data/fire_dataset.py while training.

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

from firepred.config.dataset import ARCHIVE_FORMAT_VERSION, NPZ_COMPRESSLEVEL

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
        compresslevel (int): zlib level (1 = fastest).

    Returns:
        Path: save_path.

    Raises:
        ValueError: if data and labels have different window counts.
        RuntimeError: if the written file does not read back correctly.
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
    """Name a fire by its archive or folder: p<id>.

    A fire is stored either as p<id>.npz or, after unzipping, as a p<id>/
    folder. Both give the same name, which also names its TESSERA file.

    Args:
        source (str | Path): the archive or folder.

    Returns:
        str: "p<id>".
    """
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
    """Read an unzipped fire folder exactly like an open .npz archive.

    np.load returns an object that answers archive["data_000"] and
    archive.files. This class gives an unzipped p<id>/ folder (one .npy file per
    member) the same interface, so the reading code works for both.
    """

    def __init__(self, folder):
        """Point the reader at one unzipped fire folder.

        Nothing is read yet. Members are loaded one at a time when asked for.

        Args:
            folder (str | Path): the p<id>/ folder.
        """
        self.folder = Path(folder)

    @cached_property
    def files(self):
        """List the folder's member names, like NpzFile.files.

        Each member is one .npy file. The names are the file names without .npy.

        Returns:
            list[str]: e.g. ["num_windows", "data_000", "label_000"].
        """
        return [path.stem for path in self.folder.glob("*.npy")]

    def __getitem__(self, key):
        """Load one member as a numpy array.

        Only the requested .npy file is read. Pickled objects are refused.

        Args:
            key (str): the member name, e.g. "data_000".

        Returns:
            np.ndarray: the member's array.
        """
        return np.load(self.folder / f"{key}.npy", allow_pickle=False)

    def __enter__(self):
        """Start a `with` block; a folder needs no open file handle.

        This exists so the folder can be used like an open archive. It does nothing
        but return the reader.

        Returns:
            _FireFolder: this reader.
        """
        return self

    def __exit__(self, *exc_info):
        """End a `with` block; there is nothing to close.

        This exists so the folder can be used like an open archive. Exceptions
        raised inside the block are not swallowed.

        Args:
            *exc_info: the exception details, if any (ignored).

        Returns:
            bool: False, so exceptions propagate.
        """
        return False


def open_fire(source):
    """Open a fire's archive or unzipped folder for reading.

    Use it as `with open_fire(source) as archive:` and read members with
    archive["data_000"]. Archives and unzipped folders give the same arrays.

    Args:
        source (str | Path): a p<id>.npz archive or p<id>/ folder.

    Returns:
        numpy.lib.npyio.NpzFile | _FireFolder: the open reader.
    """
    # Both kinds answer archive[key] with the same numpy array.
    if Path(source).is_dir():
        return _FireFolder(source)
    return np.load(source, allow_pickle=False)


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
                raise RuntimeError(
                    f"{npz_path} uses the old NPZ format. Expected "
                    "num_windows plus data_###/label_### members."
                )

            number_of_windows = int(
                np.asarray(archive["num_windows"]).item()
            )

            if "format_version" in archive.files:
                format_version = int(
                    np.asarray(archive["format_version"]).item()
                )
                if format_version != ARCHIVE_FORMAT_VERSION:
                    raise RuntimeError(
                        f"{npz_path} has unsupported format version "
                        f"{format_version}; expected version {ARCHIVE_FORMAT_VERSION}"
                    )

            # Checking member names reads only the ZIP directory; it does
            # not decompress the large window arrays.
            for window_idx in range(number_of_windows):
                suffix = f"{window_idx:03d}"
                data_key = f"data_{suffix}"
                label_key = f"label_{suffix}"

                if data_key not in archive.files:
                    raise RuntimeError(
                        f"{npz_path} is missing member {data_key}"
                    )
                if label_key not in archive.files:
                    raise RuntimeError(
                        f"{npz_path} is missing member {label_key}"
                    )

                index.append((npz_path, window_idx))

    return index


# ----- Unzipping once for faster reads ----- #


def _archive_signature(archive_path):
    """Describe an archive file so a later unzip can tell whether it changed.

    The description (path, size, modification time) is stored in a marker
    inside the unzipped folder. A different description means the archive was
    replaced and must be unzipped again.

    Args:
        archive_path (str | Path): the archive.

    Returns:
        dict: {"archive": path, "size": bytes, "mtime_ns": modification time}.
    """
    archive_path = Path(archive_path).resolve()
    stat = archive_path.stat()
    return {
        "archive": str(archive_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_unzip_marker(folder):
    """Read the marker an unzip left in a fire folder.

    The marker says which archive the folder came from. A missing or broken
    marker means the folder cannot be trusted and is unzipped again.

    Args:
        folder (Path): an unzipped p<id>/ folder.

    Returns:
        dict | None: the marker, or None when it is missing or unreadable.
    """
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
    return folder


def unzip_data_root(
    data_root,
    unzip_root,
    splits=("train", "val", "test"),
    workers=None,
):
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
