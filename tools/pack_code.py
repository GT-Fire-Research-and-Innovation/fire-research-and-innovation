"""Pack this local checkout's code into one zip for Kaggle or Colab, uncommitted changes included.

    python tools/pack_code.py                         # -> dist/firepred-code.zip
    python tools/pack_code.py --out /tmp/code.zip

The notebooks never clone from GitHub: they run exactly the code on this
machine. Upload the zip as a Kaggle Dataset (Kaggle unpacks it; point the
training notebook's CODE_SOURCE at the dataset folder) or to Colab's /content
(the data-grab notebook's CODE_SOURCE). The zip holds firepred/, checks/, roi/ and
pyproject.toml, plus code_version.json: the commit it was packed from, whether
the checkout had uncommitted changes, and a hash of every packed file.
firepred.data.files.code_version() reads that file when there is no git
history, so data built from a packed copy still records its code.

Members are zlib-compressed, not Zstandard: Kaggle unpacks uploaded zips with
its own tools, and the code is small.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firepred.config.dataset import CODE_MANIFEST_FILE, REPO_ROOT  # noqa: E402

# What a notebook needs: the package, diagnostic scripts, fire lists and project file.
INCLUDE = ("firepred", "checks", "roi", "pyproject.toml")
SKIP_PARTS = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}
DEFAULT_OUT = REPO_ROOT / "dist" / "firepred-code.zip"


def code_files(root=REPO_ROOT):
    """List the files to pack, sorted, without caches.

    Only INCLUDE is packed: tests, notebooks, docs and data stay behind.
    Compiled files and editor clutter are skipped.

    Args:
        root (str | Path): the checkout.

    Returns:
        list[Path]: the files, relative to root.
    """
    root = Path(root)
    found = []
    for name in INCLUDE:
        path = root / name
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        found += [p.relative_to(root) for p in candidates
                  if not SKIP_PARTS.intersection(p.parts) and p.suffix != ".pyc"]
    return sorted(found)


def git(root, *args):
    """Run one git command in the checkout and return its output.

    A machine without git, or a folder outside a checkout, gives None. Nothing
    is ever changed in the repository.

    Args:
        root (Path): the checkout.
        *args (str): the git arguments.

    Returns:
        str | None: the output, stripped, or None when git fails.
    """
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def manifest(root, files):
    """Describe the packed code: its commit, local changes and a hash of its contents.

    The hash covers every packed path and its bytes, so two zips of the same
    code have the same hash. code_version is the commit, with "+uncommitted"
    when the checkout differs from it.

    Args:
        root (Path): the checkout.
        files (list[Path]): the packed files, relative to root.

    Returns:
        dict: code_version, git_commit, uncommitted_changes, sha256, files, packed_at.
    """
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.as_posix().encode() + b"\0")
        digest.update((Path(root) / relative).read_bytes())
    commit = git(root, "rev-parse", "HEAD")
    status = git(root, "status", "--porcelain", "--", *INCLUDE)
    uncommitted = None if status is None else bool(status)
    version = (commit or "unknown") + ("+uncommitted" if uncommitted or commit is None else "")
    return {
        "code_version": version,
        "git_commit": commit,
        "uncommitted_changes": uncommitted,
        "sha256": digest.hexdigest(),
        "files": len(files),
        "packed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def pack(root=REPO_ROOT, out=DEFAULT_OUT):
    """Write the code zip: every packed file plus code_version.json at its root.

    Members keep their paths (firepred/..., roi/...), so unpacking gives a
    folder that runs `python -m firepred.cli.<name>`. An existing zip is replaced.

    Args:
        root (str | Path): the checkout.
        out (str | Path): the zip to write (its folder is created).

    Returns:
        tuple[Path, dict]: (the zip, its manifest).
    """
    root, out = Path(root), Path(out)
    files = code_files(root)
    record = manifest(root, files)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative in files:
            bundle.write(root / relative, relative.as_posix())
        bundle.writestr(CODE_MANIFEST_FILE, json.dumps(record, indent=2) + "\n")
    return out, record


def main(argv=None):
    """Pack the checkout and print where the zip went and which code it holds.

    This is `python tools/pack_code.py`. The printed version says whether
    uncommitted changes went in.

    Args:
        argv (list[str] | None): the arguments (default: sys.argv).

    Returns:
        Path: the zip.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"zip to write (default: {DEFAULT_OUT})")
    args = parser.parse_args(argv)
    out, record = pack(REPO_ROOT, args.out)
    print(f"Packed {record['files']} files into {out} ({out.stat().st_size / 2**20:.1f} MiB)")
    print(f"code_version {record['code_version']} | sha256 {record['sha256'][:12]}")
    return out


if __name__ == "__main__":
    main()
