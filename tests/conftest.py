"""Shared pytest setup: CPU only, repo root importable, synthetic data fixtures."""

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "support"))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    """A small synthetic train/val/test root with TESSERA files and a test CSV."""
    import synth

    root, csv_path = synth.data_root(tmp_path_factory.mktemp("data") / "data", synth.QUICK_FIRES)
    return root, csv_path
