"""Every source file the README and docs point to exists, so the docs cannot drift from the code."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 10-pathways proposes files that do not exist yet; the explainer page mirrors it.
PROPOSALS = {"10-pathways.md", "explainer.html"}
# Example names for a model you create with `python -m firepred.cli.new_model NAME`.
PLACEHOLDERS = ("mymodel", "NAME")
PATH = re.compile(r"(?<![\w/])((?:firepred|tools|tests)/[\w/]+\.py|(?:config|data|prep|models|training|cli)/[\w/]+\.py)")


def documents():
    """README.md and every Markdown file under docs/ that describes the code as it is."""
    docs = [ROOT / "README.md"] + sorted((ROOT / "docs").rglob("*.md"))
    return [path for path in docs if path.name not in PROPOSALS]


def resolve(reference):
    """The repo file a reference names (bare package paths are under firepred/)."""
    if reference.split("/")[0] in ("firepred", "tools", "tests"):
        return ROOT / reference
    return ROOT / "firepred" / reference


def test_documented_source_files_exist():
    """No README or doc line names a .py file that is not in the repository."""
    missing = [
        f"{path.relative_to(ROOT)}: {reference}"
        for path in documents()
        for reference in PATH.findall(path.read_text())
        if not resolve(reference).exists() and not any(word in reference for word in PLACEHOLDERS)
    ]
    assert not missing, "\n".join(missing)
