"""Local files: zip a data root, unpack zips safely, find a data root, the code version.

Level 1 (data, dataset building). Standard library and config only, so it
runs before any training dependency is installed (Kaggle/Colab notebooks
import it through the masters). Every function returns what it made or
found; none prints.

    zip_folders(folders, zip)   bundle folders (data.zip, tessera.zip)
    unzip_into(zip, folder)     unpack with a path-escape check
    extract_zip / extract_data  unpack once per session -> train/val/test root
    code_version()              git commit of this checkout (provenance)
"""

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from firepred.config.dataset import SPLITS


def find_data_root(base):
    """Find the one folder under a path that holds train/, val/ and test/ archives.

    An unzipped data.zip may put the data root at any depth. This searches for
    the folder with all three split folders and checks that each contains
    window archives.

    Args:
        base (str | Path): where to start looking.

    Returns:
        Path: the data root.

    Raises:
        RuntimeError: if there is not exactly one such folder, or a split is empty.
    """
    base = Path(base)
    # The root may sit at any depth inside the unzipped folder; there must be
    # exactly one folder holding train/, val/ and test/.
    candidates = [base] + [p.parent for p in base.rglob("train") if p.is_dir()]
    matches = sorted({p.resolve() for p in candidates
                      if all((p / split).is_dir() for split in SPLITS)})
    if len(matches) != 1:
        raise RuntimeError(f"Expected one train/val/test root under {base}; found {matches}")
    root = matches[0]
    for split in SPLITS:
        if not next((root / split).glob("*.npz"), None):
            raise RuntimeError(f"No NPZ files directly inside {root / split}")
    return root


def count_archives(root):
    """Count the window archives in each split folder of a data root.

    The notebooks print these counts after staging data. Only archives
    directly inside each split folder are counted.

    Args:
        root (str | Path): a data root with train/, val/ and test/.

    Returns:
        dict[str, int]: {split: number of .npz archives}.
    """
    root = Path(root)
    return {split: sum(1 for _ in (root / split).glob("*.npz")) for split in SPLITS}


def check_members_inside(archive, destination):
    """Refuse a zip whose members would be written outside the target folder.

    A malicious zip can name a member "../../file" to escape the folder it is
    unpacked into ("zip slip"). Python already guards against this; this check
    makes it explicit and fails loudly instead of quietly renaming.

    Args:
        archive (zipfile.ZipFile): the open zip to check.
        destination (str | Path): the folder it will be extracted into.

    Returns:
        list[str]: the member names checked.

    Raises:
        ValueError: if any member would land outside destination.
    """
    root = Path(destination).resolve()
    names = []
    for item in archive.infolist():
        if not (root / item.filename).resolve().is_relative_to(root):
            raise ValueError(f"Archive member escapes extraction folder: {item.filename}")
        names.append(item.filename)
    return names


def zip_folders(folders, output_path):
    """Zip every file under the given folders into one zip file.

    Each member keeps its folder name (train/p123.npz), so unzipping recreates
    the same layout. Files are stored without recompression, because the
    archives inside are already compressed. Folders that do not exist are skipped.

    Args:
        folders (list[str | Path]): folders to include.
        output_path (str | Path): the zip file to write.

    Returns:
        list[str]: the member names written.
    """
    members = []
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as bundle:
        for folder in map(Path, folders):
            if not folder.exists():
                continue
            for file_path in folder.rglob("**/*"):
                if file_path.is_file():
                    name = file_path.relative_to(folder.parent)
                    bundle.write(file_path, name)
                    members.append(str(name))
    return members


def unzip_into(zip_path, destination):
    """Unpack a zip into a folder after checking its member paths.

    This is the plain unpack used after downloading a zip. Member paths are
    checked first, so nothing can be written outside the folder.

    Args:
        zip_path (str | Path): the zip file.
        destination (str | Path): the folder to unpack into.

    Returns:
        Path: destination.
    """
    destination = Path(destination)
    with zipfile.ZipFile(zip_path, "r") as bundle:
        check_members_inside(bundle, destination)
        bundle.extractall(destination)
    return destination


def extract_zip(archive_path, destination):
    """Unpack a zip once per session, checking disk space and member paths.

    A marker file records which zip was unpacked: the same zip again does
    nothing, while a different zip is refused instead of being mixed in. The
    disk must have room for the unpacked files plus 1 GiB.

    Args:
        archive_path (str | Path): the zip file.
        destination (str | Path): the folder to unpack into.

    Returns:
        Path: destination.

    Raises:
        RuntimeError: if a different zip was unpacked there, or the disk is too small.
    """
    archive_path, destination = Path(archive_path), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # A marker records which zip was unpacked here: the same zip again is a
    # no-op, a different zip is refused rather than mixed in.
    marker = destination / ".extraction-complete.json"
    stat = archive_path.stat()
    signature = {"path": str(archive_path.resolve()), "size": stat.st_size,
                 "mtime_ns": stat.st_mtime_ns}
    if marker.exists():
        if json.loads(marker.read_text()) == signature:
            return destination
        raise RuntimeError("The archive changed. Choose a fresh extraction folder first.")
    with zipfile.ZipFile(archive_path) as archive:
        # Check free disk space (with 1 GiB to spare) before unpacking.
        needed = sum(item.file_size for item in archive.infolist())
        free = shutil.disk_usage(destination).free
        if needed + 1024**3 > free:
            raise RuntimeError(
                f"Insufficient disk space: {needed / 1024**3:.1f} GiB needed, {free / 1024**3:.1f} GiB free. "
                "Use an attached, extracted Kaggle Dataset."
            )
        check_members_inside(archive, destination)
        archive.extractall(destination)
    marker.write_text(json.dumps(signature))
    return destination


def extract_data(archive_path, destination):
    """Unpack a data.zip and return its train/val/test root.

    This combines extract_zip and find_data_root. The training notebooks use it
    to turn a downloaded or attached zip into a data root.

    Args:
        archive_path (str | Path): the data.zip file.
        destination (str | Path): the folder to unpack into.

    Returns:
        Path: the data root inside destination.
    """
    return find_data_root(extract_zip(archive_path, destination))


def code_version():
    """Read the git commit of this checkout.

    The commit is stored with every TESSERA file and window_config.json, so
    anyone can see which code built them. Outside a git checkout it is empty.

    Returns:
        str: the commit hash, or "" when it cannot be read.
    """
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""
