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
    """Fill the model template with a new model's name.

    The template's title becomes a placeholder for the author. The note that
    only applies to the template is removed.

    Args:
        name (str): the new model's name.

    Returns:
        str: the new file's text.
    """
    text = TEMPLATE.read_text()
    # Swap the template's title line for a placeholder the author fills in, drop
    # the note that only applies to the template, and put in the model name.
    first_line = text.splitlines()[0]
    text = text.replace(first_line, f'"""{name}: <one line: what this model is and why>.', 1)
    text = text.replace('(Files whose name starts with "_" are never registered, so this one is inert.)\n\n', "", 1)
    return text.replace(PLACEHOLDER, name)


def create(name, directory=ZOO_DIR):
    """Write a new zoo file for a model from the template.

    The name becomes the module name and the -m value. Names starting with "_"
    are refused (the zoo skips them), and an existing model is never overwritten.

    Args:
        name (str): the new model's name (a Python identifier).
        directory (str | Path): where to write it (default: models/zoo/).

    Returns:
        Path: the new file.

    Raises:
        ValueError: if the name is not a valid identifier or starts with "_".
        FileExistsError: if the model already exists.
    """
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
    """Create a new zoo model from the command line and print the next step.

    This is `python -m firepred.cli.new_model NAME`. The next step is to edit
    the file and run the model check.

    Args:
        argv (list[str] | None): the arguments (default: sys.argv).

    Returns:
        Path: the new file.
    """
    parser = argparse.ArgumentParser(prog="python -m firepred.cli.new_model", description=__doc__.splitlines()[0])
    parser.add_argument("name", help="Model name used with -m, e.g. convlstm_unet")
    args = parser.parse_args(argv)
    path = create(args.name)
    print(f"Created {path}")
    print(f"Next: edit it, then run  python -m firepred.cli.check model -m {args.name}")
    return path


if __name__ == "__main__":
    main()
