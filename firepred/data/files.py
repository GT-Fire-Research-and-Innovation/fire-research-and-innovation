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
    """Return the one folder under `base` holding train/, val/ and test/ archives."""
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
    """{split: number of .npz archives directly inside root/split}."""
    root = Path(root)
    return {split: sum(1 for _ in (root / split).glob("*.npz")) for split in SPLITS}


def check_members_inside(archive, destination):
    """Refuse a zip whose member paths would land outside `destination` ("zip slip").

    CPython's extractall already strips ".." and absolute paths; this makes the
    guarantee explicit and fails loudly instead of rewriting names. Returns the
    member names checked.
    """
    root = Path(destination).resolve()
    names = []
    for item in archive.infolist():
        if not (root / item.filename).resolve().is_relative_to(root):
            raise ValueError(f"Archive member escapes extraction folder: {item.filename}")
        names.append(item.filename)
    return names


def zip_folders(folders, output_path):
    """Zip every file under each existing folder into `output_path`; return the member names.

    Members keep their folder name (train/p123.npz), so unzipping recreates the
    layout. ZIP_STORED: the archives inside are already compressed. Missing
    folders are skipped.
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
    """Extract `zip_path` into `destination` after the path-escape check; return destination."""
    destination = Path(destination)
    with zipfile.ZipFile(zip_path, "r") as bundle:
        check_members_inside(bundle, destination)
        bundle.extractall(destination)
    return destination


def extract_zip(archive_path, destination):
    """Extract once per session, with disk-space and path-escape checks; return destination."""
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
    """Extract a data.zip and return its train/val/test root."""
    return find_data_root(extract_zip(archive_path, destination))


def code_version():
    """Git commit of this repository, or "" when not run from a checkout."""
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""
