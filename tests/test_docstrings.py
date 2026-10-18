"""Every module, class, function and method in firepred/, tools/ and tests/ explains itself in a docstring."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("firepred", "tools", "tests")


def missing_docstrings():
    """'path:line name' for every module and definition (nested ones included) without a docstring."""
    missing = []
    for folder in FOLDERS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            tree = ast.parse(path.read_text())
            relative = path.relative_to(ROOT)
            if not ast.get_docstring(tree) and path.name != "__init__.py" or (
                path.name == "__init__.py" and path.read_text().strip() and not ast.get_docstring(tree)
            ):
                missing.append(f"{relative}: module")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not ast.get_docstring(node):
                    missing.append(f"{relative}:{node.lineno} {node.name}")
    return missing


def test_everything_has_a_docstring():
    """No module, class, function or method is left without a docstring."""
    missing = missing_docstrings()
    assert not missing, f"{len(missing)} without a docstring:\n" + "\n".join(missing)
