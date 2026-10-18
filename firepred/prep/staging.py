"""Local files: zip a data root, unpack zips safely, stage a training root.

Level 1 (prep). Standard library and config only, so it runs before any
training dependency is installed (Kaggle/Colab notebooks import it through
the masters). Every function returns what it made, found or removed; none
prints.

    zip_folders(folders, zip)              bundle folders (data.zip, tessera.zip)
    unzip_into(zip, folder)                unpack with a path-escape check
    extract_zip / extract_data             unpack once per session -> train/val/test root
    build_tessera_view(root, tessera, out) symlink view: kept fires + tessera/
    drop_excluded_archives(root, names)    delete listed window archives (explicit)
"""

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from firepred.config.data import SPLITS, TESSERA_EXCLUDED_LIST


@dataclass(frozen=True)
class TesseraView:
    """A symlinked training root with fires lacking TESSERA left out."""

    root: Path
    dropped: tuple

    def summary(self):
        """One line: where the view is and how many archives it leaves out."""
        return f"TESSERA view {self.root}: dropped {len(self.dropped)} archives without prior-year TESSERA"


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


def read_excluded_list(tessera_dir):
    """The window archives listed in tessera_dir/excluded_npz.txt, e.g. ['train/p1.npz']."""
    path = Path(tessera_dir) / TESSERA_EXCLUDED_LIST
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_excluded_list(tessera_dir, excluded):
    """Write the drop list into tessera_dir/excluded_npz.txt; return its path."""
    path = Path(tessera_dir) / TESSERA_EXCLUDED_LIST
    path.write_text("\n".join(excluded) + "\n")
    return path


def drop_excluded_archives(data_root, excluded):
    """Delete the listed window archives under `data_root`; return the names removed."""
    removed = []
    for name in excluded:
        path = Path(data_root) / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def clear_view(view_dir):
    """Remove an old symlink view; refuse a folder that holds real files. Returns view_dir."""
    view_dir = Path(view_dir)
    if not view_dir.exists():
        return view_dir
    real_files = [p for p in view_dir.rglob("*") if p.is_file() and not p.is_symlink()
                  and not any(parent.is_symlink() for parent in p.parents)]
    if real_files:
        raise RuntimeError(f"{view_dir} holds real files (e.g. {real_files[0]}); refusing to clear it")
    shutil.rmtree(view_dir)
    return view_dir


def build_tessera_view(data_root, tessera_dir, view_dir):
    """Symlink the kept window archives and tessera/ into view_dir; return a TesseraView.

    Fires listed in tessera_dir/excluded_npz.txt (no prior-year TESSERA) are
    left out; the extracted data itself is never modified, so the TESSERA run
    and its matched no-TESSERA baseline read the same fires. [ours]
    """
    data_root, tessera_dir, view_dir = Path(data_root), Path(tessera_dir), Path(view_dir)
    excluded = set(read_excluded_list(tessera_dir))
    clear_view(view_dir)
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
    return TesseraView(view_dir, tuple(dropped))
