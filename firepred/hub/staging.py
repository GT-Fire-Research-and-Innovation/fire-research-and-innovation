"""Turn downloaded zips into a training data root, without modifying the data.

Level 2 (hub). Used by the Kaggle notebook and cli/check.py. Standard library
only, so it runs before any training dependency is installed.

    extract_data(zip, folder)              unpack once per session -> the train/val/test root
    build_tessera_view(root, tessera, out) symlink view: kept fires + tessera/
"""

import json
import shutil
import zipfile
from pathlib import Path

from firepred.config.dataset import SPLITS, TESSERA_EXCLUDED_LIST


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


def check_members_inside(archive, destination):
    """Refuse a zip whose member paths would land outside `destination` ("zip slip").

    CPython's extractall already strips ".." and absolute paths; this makes the
    guarantee explicit and fails loudly instead of rewriting names.
    """
    root = Path(destination).resolve()
    for item in archive.infolist():
        if not (root / item.filename).resolve().is_relative_to(root):
            raise ValueError(f"Archive member escapes extraction folder: {item.filename}")


def extract_zip(archive_path, destination):
    """Extract once per session, with disk-space and path-escape checks."""
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
        members = archive.infolist()
        # Check free disk space (with 1 GiB to spare) before unpacking.
        needed = sum(item.file_size for item in members)
        free = shutil.disk_usage(destination).free
        print(f"Uncompressed data: {needed / 1024**3:.1f} GiB; free disk: {free / 1024**3:.1f} GiB")
        if needed + 1024**3 > free:
            raise RuntimeError("Insufficient disk space. Use an attached, extracted Kaggle Dataset.")
        check_members_inside(archive, destination)
        archive.extractall(destination)
    marker.write_text(json.dumps(signature))
    return destination


def extract_data(archive_path, destination):
    """Extract a data.zip and return its train/val/test root."""
    return find_data_root(extract_zip(archive_path, destination))


def build_tessera_view(data_root, tessera_dir, view_dir):
    """Symlink the kept window archives and tessera/ into view_dir.

    Fires listed in tessera_dir/excluded_npz.txt (no prior-year TESSERA) are
    left out; the extracted data itself is never modified, so the TESSERA run
    and its matched no-TESSERA baseline read the same fires. [ours]
    """
    data_root, tessera_dir, view_dir = Path(data_root), Path(tessera_dir), Path(view_dir)
    excluded = set((tessera_dir / TESSERA_EXCLUDED_LIST).read_text().split())
    if view_dir.exists():
        shutil.rmtree(view_dir)  # Removes only the symlinks, never the data.
    dropped = []
    # Symlink every kept archive into view_dir/<split>/; excluded ones are only counted.
    for split in SPLITS:
        (view_dir / split).mkdir(parents=True)
        for archive in sorted((data_root / split).glob("*.npz")):
            if f"{split}/{archive.name}" in excluded:
                dropped.append(f"{split}/{archive.name}")
            else:
                (view_dir / split / archive.name).symlink_to(archive.resolve())
    (view_dir / "tessera").symlink_to(tessera_dir.resolve(), target_is_directory=True)
    # Every kept archive needs its TESSERA file; fail before training if any is missing.
    missing = [f"{split}/{archive.name}" for split in SPLITS
               for archive in (view_dir / split).glob("*.npz")
               if not (tessera_dir / archive.name).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} kept archives have no TESSERA file: {missing[:10]}")
    print(f"TESSERA mode: dropped {len(dropped)} archives without prior-year TESSERA")
    return view_dir
