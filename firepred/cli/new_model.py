"""Start a new model: copy the zoo template to firepred/models/zoo/NAME.py.

Level 4 (cli). `python -m firepred.cli.new_model mymodel`, then edit the new
file and run `python -m firepred.cli.check model -m mymodel`.
"""

import argparse
from pathlib import Path

ZOO_DIR = Path(__file__).resolve().parents[1] / "models" / "zoo"
TEMPLATE = ZOO_DIR / "_template.py"
PLACEHOLDER = "__MODEL_NAME__"


def render(name):
    """The template's text for a model called `name`."""
    text = TEMPLATE.read_text()
    # Swap the template's title line for a placeholder the author fills in, drop
    # the note that only applies to the template, and put in the model name.
    first_line = text.splitlines()[0]
    text = text.replace(first_line, f'"""{name}: <one line: what this model is and why>.', 1)
    text = text.replace('(Files whose name starts with "_" are never registered, so this one is inert.)\n\n', "", 1)
    return text.replace(PLACEHOLDER, name)


def create(name, directory=ZOO_DIR):
    """Write directory/NAME.py from the template and return its path."""
    # The name becomes a module and the -m value. Zoo discovery skips files that
    # start with '_', so such names are refused; an existing model is never overwritten.
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"{name!r} must be a Python identifier not starting with '_'")
    path = Path(directory) / f"{name}.py"
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.write_text(render(name))
    return path


def main(argv=None):
    """Parse NAME, write models/zoo/NAME.py from the template and print the next step; returns the path."""
    parser = argparse.ArgumentParser(prog="python -m firepred.cli.new_model", description=__doc__.splitlines()[0])
    parser.add_argument("name", help="Model name used with -m, e.g. convlstm_unet")
    args = parser.parse_args(argv)
    path = create(args.name)
    print(f"Created {path}")
    print(f"Next: edit it, then run  python -m firepred.cli.check model -m {args.name}")


if __name__ == "__main__":
    main()
