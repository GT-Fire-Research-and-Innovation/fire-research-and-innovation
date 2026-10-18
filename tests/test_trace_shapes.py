"""tools/trace_shapes.py runs the real model; docs/swinunetr3d/shapes.md is its output."""

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "trace_shapes.py"


def load_tool():
    """Import tools/trace_shapes.py as a module."""
    spec = importlib.util.spec_from_file_location("trace_shapes", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_on_a_small_input():
    """The traced shapes, the output contract and the parameter count on a 64 x 64 input."""
    text = load_tool().main(["--size", "64", "--out", "-"])
    assert "| `swinViT.patch_embed` | (1, 43, 6, 64, 64) | (1, 36, 6, 32, 32) |" in text
    assert "-> mean over T -> (1, 2, 64, 64)" in text
    assert "| **Total** | **33,170,918** | 100% |" in text


def test_shapes_doc_is_current():
    """docs/swinunetr3d/shapes.md (when present) matches the full-size trace."""
    doc = TOOL.parents[1] / "docs" / "swinunetr3d" / "shapes.md"
    if doc.exists():
        assert "| `swinViT.patch_embed` | (1, 43, 6, 256, 256) | (1, 36, 6, 128, 128) |" in doc.read_text()
