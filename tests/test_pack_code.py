"""tools/pack_code.py: the notebooks' code comes from this checkout, packed with what it was packed from."""

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import pack_code  # noqa: E402


def test_zip_holds_the_package_and_a_manifest_but_no_caches(tmp_path):
    """firepred/, roi/ and pyproject.toml go in with code_version.json; caches, tests and notebooks stay out."""
    out, record = pack_code.pack(ROOT, tmp_path / "code.zip")
    with zipfile.ZipFile(out) as bundle:
        names = bundle.namelist()
        manifest = json.loads(bundle.read("code_version.json"))
    assert "firepred/cli/train.py" in names and "roi/us_fire_2021_out_new.csv" in names and "pyproject.toml" in names
    assert not [name for name in names if "__pycache__" in name or name.endswith(".pyc")]
    assert not [name for name in names if name.startswith(("tests/", "notebooks/", "docs/"))]
    assert manifest == record and manifest["files"] == len(names) - 1
    assert manifest["code_version"].startswith(manifest["git_commit"] or "unknown")


def test_same_code_gives_the_same_hash(tmp_path):
    """The content hash depends only on the packed files, not on when or where the zip was written."""
    _, first = pack_code.pack(ROOT, tmp_path / "a.zip")
    _, second = pack_code.pack(ROOT, tmp_path / "b.zip")
    assert first["sha256"] == second["sha256"]


def test_unpacked_copy_reports_its_code_version(tmp_path):
    """Without git history, code_version() reads the packed copy's manifest (data it builds records it)."""
    import subprocess

    out, record = pack_code.pack(ROOT, tmp_path / "code.zip")
    with zipfile.ZipFile(out) as bundle:
        bundle.extractall(tmp_path / "copy")
    result = subprocess.run([sys.executable, "-c", "from firepred.data.files import code_version; print(code_version())"],
                            cwd=tmp_path / "copy", capture_output=True, text=True, check=True)
    assert result.stdout.strip() == record["code_version"]
