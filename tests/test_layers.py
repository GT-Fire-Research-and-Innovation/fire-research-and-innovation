"""The package is a hierarchy: each subpackage imports only from the ones it builds on."""

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "firepred"

# subpackage -> subpackages it may import (besides itself).
ALLOWED = {
    "config": set(),
    "data": {"config"},
    "models": {"config"},
    "prep": {"config", "data"},
    "training": {"config", "data", "models"},
    "cli": {"config", "data", "models", "prep", "training"},
}


def imported_subpackages(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "firepred" and len(parts) > 1:
                yield parts[1], node.lineno


def test_every_subpackage_is_known():
    found = {path.name for path in PACKAGE.iterdir() if path.is_dir() and not path.name.startswith("__")}
    assert found <= set(ALLOWED), found - set(ALLOWED)


def test_imports_only_go_down_the_hierarchy():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        if len(relative.parts) < 2:
            continue
        owner = relative.parts[0]
        for target, line in imported_subpackages(path):
            if target != owner and target not in ALLOWED[owner]:
                violations.append(f"firepred/{relative}:{line} imports firepred.{target}")
    assert not violations, "\n".join(violations)


def test_package_init_files_import_nothing():
    for path in PACKAGE.rglob("__init__.py"):
        tree = ast.parse(path.read_text())
        assert not [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))], path


def test_training_path_skips_dataset_building_dependencies():
    """Kaggle installs no geospatial packages for training, so training never
    imports the dataset-building subpackages. (MONAI may import huggingface_hub
    optionally when it is installed; that is MONAI's business.)"""
    code = (
        "import sys, importlib, pkgutil\n"
        "import firepred.cli.train\n"
        "heavy = {'rasterio', 'pyproj', 'geotessera', 'firepred.prep'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PACKAGE.parent, capture_output=True, text=True)
    if "No module named 'firepred.cli.train'" in result.stderr:
        return  # not written yet
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout
