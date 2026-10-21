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


# ----- The docstring format of the package and tools ----- #

API_FOLDERS = ("firepred", "tools")
SECTIONS = ("Args:", "Returns:", "Yields:", "Raises:", "Attributes:", "Example:")
# A function with at most this many code lines may have a one-line docstring: its code shows the rest.
SHORT_FUNCTION_LINES = 6


def parameters(node):
    """The parameter names a function's Args section must list (self/cls and *args/**kwargs excluded)."""
    args = node.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    return [name for name in names if name not in ("self", "cls")]


def returns_value(node):
    """True when the function itself (not a nested one) returns something other than None."""
    stack = list(node.body)
    while stack:
        item = stack.pop()
        if isinstance(item, ast.Return) and item.value is not None:
            if not (isinstance(item.value, ast.Constant) and item.value.value is None):
                return True
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(item))
    return False


def yields(node):
    """True when the function itself is a generator."""
    stack = list(node.body)
    while stack:
        item = stack.pop()
        if isinstance(item, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(item))
    return False


def code_lines(node, source):
    """Count the non-blank lines of a function's body after its docstring."""
    if len(node.body) < 2:
        return 0
    return sum(1 for line in source[node.body[1].lineno - 1:node.end_lineno] if line.strip())


def is_dataclass(node):
    """True for a class decorated with @dataclass(...)."""
    return any(getattr(getattr(d, "func", d), "id", None) == "dataclass" for d in node.decorator_list)


def format_problems(node, doc, source=()):
    """What a definition's docstring is missing from the format (an empty list when it complies)."""
    problems = []
    lines = doc.splitlines()
    summary = lines[0].strip() if lines else ""
    if not summary.endswith(".") or summary.count(". ") > 0:
        problems.append("first line must be one sentence ending in '.'")
    short = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and code_lines(node, source) <= SHORT_FUNCTION_LINES
    if len(lines) == 1 and short:
        return problems
    if len(lines) < 3 or lines[1].strip():
        problems.append("needs a blank line, then a plain-language explanation")
        return problems
    paragraph = []
    for line in lines[2:]:
        if line.strip() in SECTIONS:
            break
        paragraph.append(line)
    if sum(text.count(".") for text in paragraph) < 2:
        problems.append("explanation must be at least two sentences")
    if isinstance(node, ast.ClassDef):
        if is_dataclass(node):
            fields = [s.target.id for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
            if fields and "Attributes:" not in doc:
                problems.append("dataclass needs an Attributes: section")
            problems += [f"Attributes: missing {name}" for name in fields if f"{name} (" not in doc]
        return problems
    names = parameters(node)
    if names and "Args:" not in doc:
        problems.append("needs an Args: section")
    problems += [f"Args: missing '{name} (type)'" for name in names if f"{name} (" not in doc]
    if returns_value(node) and node.name != "__init__" and "Returns:" not in doc:
        problems.append("needs a Returns: section")
    if "Returns:" in doc and not returns_value(node):
        problems.append("documents a Returns: section but returns nothing")
    if yields(node) and "Yields:" not in doc:
        problems.append("needs a Yields: section")
    return problems


def api_docstring_problems():
    """'path:line name: problem' for every package or tool definition whose docstring breaks the format."""
    found = []
    for folder in API_FOLDERS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            source = path.read_text()
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    doc = ast.get_docstring(node) or ""
                    for problem in format_problems(node, doc, source.splitlines()):
                        found.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name}: {problem}")
    return found


def test_api_docstrings_follow_the_format():
    """Every package and tool docstring: one-sentence summary, a plain explanation, typed Args and Returns.

    A function of at most SHORT_FUNCTION_LINES code lines may have the summary line alone.
    """
    problems = api_docstring_problems()
    assert not problems, f"{len(problems)} problems:\n" + "\n".join(problems[:60])


def test_package_docstrings_list_their_modules():
    """Every package __init__ docstring names each module of its package, so the overview stays complete."""
    missing = []
    for init in sorted((ROOT / "firepred").rglob("__init__.py")):
        doc = ast.get_docstring(ast.parse(init.read_text())) or ""
        if init.parent == ROOT / "firepred":
            continue  # the top-level package names its subpackages, not modules
        for module in sorted(init.parent.glob("*.py")):
            if module.name != "__init__.py" and module.stem not in doc:
                missing.append(f"{init.relative_to(ROOT)}: {module.stem}")
    assert not missing, "package docstrings do not mention:\n" + "\n".join(missing)
