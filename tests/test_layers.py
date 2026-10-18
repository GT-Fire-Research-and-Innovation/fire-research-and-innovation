"""The package is a hierarchy: each subpackage imports only from the ones it builds on."""

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "firepred"

# Dataset-building modules of data/: training must never load them (Kaggle has no geo packages).
BUILD_ONLY = ("windows", "tessera", "hub", "tessera_master", "windows_master")

# subpackage -> subpackages it may import (besides itself).
ALLOWED = {
    "config": set(),
    "data": {"config"},
    "models": {"config"},
    "training": {"config", "data", "models"},
    "cli": {"config", "data", "models", "training"},
}


def imported_subpackages(path):
    """Yield (subpackage, line) for every absolute firepred.<subpackage> import in a file."""
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
    """Every folder under firepred/ is a known subpackage."""
    found = {path.name for path in PACKAGE.iterdir() if path.is_dir() and not path.name.startswith("__")}
    assert found <= set(ALLOWED), found - set(ALLOWED)


def test_imports_only_go_down_the_hierarchy():
    """Each subpackage imports only the subpackages ALLOWED lists for it."""
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
    """Package __init__ files hold only a docstring, so importing a subpackage costs nothing."""
    for path in PACKAGE.rglob("__init__.py"):
        tree = ast.parse(path.read_text())
        assert not [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))], path


def test_training_path_skips_dataset_building_dependencies():
    """Kaggle installs no geospatial packages for training, so training never
    imports the dataset-building modules of data/. (MONAI may import huggingface_hub
    optionally when it is installed; that is MONAI's business.)"""
    code = (
        "import sys, importlib, pkgutil\n"
        "import firepred.cli.train\n"
        f"heavy = {{'rasterio', 'pyproj', 'geotessera'}} | {set(f'firepred.data.{m}' for m in BUILD_ONLY)!r}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PACKAGE.parent, capture_output=True, text=True)
    if "No module named 'firepred.cli.train'" in result.stderr:
        return  # not written yet
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


# Import order inside a subpackage: each module imports only modules listed before it.
ORDER = {
    "data": ["fires", "archive", "tessera_archive", "transforms", "fire_dataset", "files", "tessera", "hub",
             "windows", "tessera_master", "windows_master"],
    "training": ["config", "accelerator", "flags", "objectives", "checkpoints", "tracker", "evaluator", "trainer",
                 "experiment"],
}


def sibling_imports(path, package):
    """Modules of `package` that the file at `path` imports (`from firepred.<package> import x` counts x)."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            parts = node.module.split(".")
            if parts[:2] == ["firepred", package]:
                if len(parts) > 2:
                    yield parts[2], node.lineno
                else:
                    for alias in node.names:
                        yield alias.name, node.lineno


def test_modules_inside_a_subpackage_import_only_earlier_ones():
    """Reading a subpackage in ORDER never needs a module that comes later."""
    for package, order in ORDER.items():
        folder = PACKAGE / package
        listed = {path.stem for path in folder.glob("*.py") if path.stem != "__init__"}
        assert listed == set(order), f"{package}: ORDER lists {sorted(order)}, folder has {sorted(listed)}"
        for path in folder.glob("*.py"):
            if path.stem == "__init__":
                continue
            for target, line in sibling_imports(path, package):
                assert order.index(target) < order.index(path.stem), (
                    f"firepred/{package}/{path.name}:{line} imports {target}, which comes later"
                )


def test_dataset_masters_import_without_geo_or_training_packages():
    """The training notebooks stage data through the masters on machines without
    rasterio/pyproj/geotessera, and pin_caches must run before huggingface_hub loads."""
    code = (
        "import sys\n"
        "for name in ('rasterio', 'pyproj', 'geotessera'):\n"
        "    sys.modules[name] = None  # importing them now raises ImportError\n"
        "import firepred.data.windows_master, firepred.data.tessera_master\n"
        "print(sorted({'torch', 'huggingface_hub', 'wandb', 'monai'} & set(sys.modules)))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PACKAGE.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout


def test_only_the_tracker_talks_to_wandb():
    """W&B global state is touched in exactly one module."""
    users = []
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            if any(name.split(".")[0] == "wandb" for name in names):
                users.append(str(path.relative_to(PACKAGE)))
    assert sorted(set(users)) == ["training/tracker.py"], users


def test_run_config_is_light():
    """A notebook can build a RunConfig and a training command without loading torch."""
    code = (
        "import sys\n"
        "from firepred.training.config import RunConfig\n"
        "RunConfig('swinunetr3d').run_name\n"
        "print(sorted({'torch', 'monai', 'wandb'} & set(sys.modules)))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PACKAGE.parent, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout
