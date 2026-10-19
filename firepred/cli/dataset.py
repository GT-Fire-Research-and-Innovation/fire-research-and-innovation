"""Build the TS-SatFire next-day prediction dataset through Hugging Face.

Level 4 (cli). Each subcommand is one master function (data/windows_master.py,
data/tessera_master.py); its flags set that function's parameters of the same
name, and every default is the function's own (so the defaults live in one
place: the master signatures, which take them from firepred/config/dataset.py).
The token defaults to the HF_TOKEN environment variable; --window-repo to the
repo for the chosen -ts/-it (window_repo_for).

    python -m firepred.cli.dataset process -ts 6 -it 1 [--only-fires 21890003 ...]
    python -m firepred.cli.dataset tessera [--only-fires 21890003 ...]
    python -m firepred.cli.dataset tessera-zip
    python -m firepred.cli.dataset pull-zip --data-root ./data && python -m firepred.cli.dataset pull-tessera --data-root ./data
    python -m firepred.cli.dataset download -ts 6 -it 1 --data-root ./data [--tessera]
    python -m firepred.cli.dataset upload-zip --data-root ./data [--zip-repo tessera --zip-name data_tessera.zip]
    python -m firepred.cli.dataset pull-zip --data-root ./data [--zip-repo window|tessera]
    python -m firepred.cli.dataset rebuild-zip -ts 6 -it 1
    python -m firepred.cli.dataset stage --work-dir /kaggle/temp/fire-research [--tessera]
"""

import argparse
import inspect
import os
from dataclasses import dataclass, field

from firepred.config.dataset import DATA_DIR, SPLITS, window_repo_for
from firepred.data import tessera_master, windows_master

ZIP_REPOS = ("window", "tessera")


@dataclass(frozen=True)
class Option:
    """Describe one flag of a dataset subcommand.

    The flag sets the master function parameter named by dest. Its default is
    that parameter's default unless options give one.

    Attributes:
        flags (tuple[str, ...]): spellings, e.g. ("--batch-size",).
        dest (str): the master parameter it sets.
        options (dict): argparse keywords.
    """

    flags: tuple
    dest: str
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Command:
    """Describe one dataset subcommand: the master function it runs and its flags.

    Every subcommand is one master step. Its flags are passed to the function
    by name.

    Attributes:
        name (str): the subcommand, e.g. "process".
        function (Callable): the master function.
        help (str): the help line.
        options (tuple[Option, ...]): its own flags.
        windows (bool): it takes -ts/-it.
    """

    name: str
    function: object
    help: str
    options: tuple = ()
    windows: bool = False   # takes -ts/-it


def parameter_default(function, name):
    """Read a function parameter's default.

    The dataset CLI's defaults come from the master signatures, so they are
    written only once. Required parameters give None.

    Args:
        function (Callable): the master function.
        name (str): the parameter name.

    Returns:
        object: the default, or None.
    """
    parameter = inspect.signature(function).parameters.get(name)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


def splits_option(help_text):
    """Make the -mode flag: which splits to work on.

    Several splits may be given. The default is the master's, all three.

    Args:
        help_text (str): the flag's help.

    Returns:
        Option: the flag.
    """
    return Option(("-mode",), "splits", {"nargs": "+", "choices": SPLITS, "help": help_text})


COMMANDS = (
    # process (step 1): raw GeoTIFFs -> one windowed NPZ archive per fire in the
    # window repo. The label flags pick the burned-area mask (with or without NIFC).
    Command("process", windows_master.process_windows,
            "Stream raw fires from --src-repo into windowed NPZs in --window-repo", (
                splits_option("Splits to build"),
                Option(("--batch-size",), "batch_size", {"type": int}),
                Option(("--overwrite",), "overwrite", {
                    "action": "store_true",
                    "help": "Rebuild fires already in --window-repo and rewrite its window_config.json "
                            "(window length/stride, label rules, crop, split) with the settings of this run"}),
                Option(("--debug",), "debug", {"action": "store_true"}),
                Option(("--only-fires",), "only_fires", {"nargs": "+", "help": "Restrict to these fire IDs (debugging)"}),
                Option(("--ignore-roi-label-sel",), "use_roi_label_sel", {
                    "action": "store_false",
                    "help": "Use --train-val-label-sel for test fires too instead of their ROI value"}),
                Option(("--train-val-label-sel",), "train_val_label_sel", {
                    "type": int, "choices": (0, 1),
                    "help": "Train/val burned-area mask: 0 = AF + NIFC perimeters (paper), 1 = AF only (released code)"}),
            ), windows=True),
    # Optional TESSERA steps: build per-fire embeddings (tessera), bundle them into
    # one zip (tessera-zip), unpack it beside the window data (pull-tessera).
    Command("tessera", tessera_master.process_tessera,
            "Build prior-year TESSERA embeddings per fire into --tessera-repo", (
                splits_option("Splits whose fires get TESSERA"),
                Option(("--batch-size",), "batch_size", {"type": int}),
                Option(("--strip-rows",), "strip_rows", {"type": int}),
                Option(("--overwrite",), "overwrite", {"action": "store_true"}),
                Option(("--only-fires",), "only_fires", {"nargs": "+", "help": "Restrict to these fire IDs (timing test)"}),
                Option(("--tessera-cells-per-pixel",), "cells_per_pixel", {
                    "type": int, "help": "TESSERA cells per VIIRS pixel side: 1 averages each pixel, 5 keeps 5x5 "
                    "cells of 75 m (use a separate --tessera-repo per value)"}),
            )),
    Command("tessera-zip", tessera_master.upload_tessera_zip,
            "Bundle --tessera-repo tessera/ and the drop list into tessera.zip there"),
    Command("pull-tessera", tessera_master.pull_tessera_zip,
            "Unpack tessera.zip into --data-root/tessera and drop 2017 window archives", (
                Option(("--data-root",), "data_root", {"default": DATA_DIR}),  # a required parameter
                Option(("--keep-excluded",), "drop_excluded", {"action": "store_false"}),
            )),
    # Move window archives between the repo and a local data root: file by file
    # (download), or as one zip (upload-zip / pull-zip): one file to fetch instead
    # of one per fire.
    Command("download", windows_master.download_windows,
            "Download --window-repo NPZs into --data-root/{train,val,test}", (
                Option(("--data-root",), "data_root"),
                Option(("--tessera",), "include_tessera", {
                    "action": "store_true",
                    "help": "TESSERA mode: drop fires without prior-year TESSERA and download "
                            "--tessera-repo embeddings into --data-root/tessera"}),
            ), windows=True),
    Command("upload-zip", windows_master.upload_data_zip, "Zip --data-root and upload it to --zip-repo", (
                Option(("--data-root",), "data_root"),
                Option(("--zip-path",), "zip_path", {"help": "Local zip file (default: <data-root>/../<zip-name>)"}),
                Option(("--zip-repo",), "zip_repo", {"choices": ZIP_REPOS, "default": "window"}),
                Option(("--zip-name",), "zip_name", {"help": "data_tessera.zip for TESSERA mode"}),
            ), windows=True),
    Command("pull-zip", windows_master.pull_data_zip,
            "Download a zip from --zip-repo and extract it into --data-root", (
                Option(("--data-root",), "data_root", {"default": DATA_DIR}),  # a required parameter
                Option(("--zip-repo",), "zip_repo", {"choices": ZIP_REPOS, "default": "window"}),
                Option(("--zip-name",), "zip_name"),
            ), windows=True),
    # rebuild-zip: download every archive of the window repo and upload it back as data.zip.
    Command("rebuild-zip", windows_master.rebuild_data_zip,
            "Download --window-repo into --work-dir and upload it there as data.zip", windows=True),
    # stage: what the training notebooks run before training (a training root under --work-dir).
    Command("stage", windows_master.stage_training_data,
            "Download/extract data.zip (and tessera.zip) under --work-dir and print the training root", (
                Option(("--tessera",), "use_tessera", {"action": "store_true",
                                                       "help": "Also stage TESSERA and drop excluded fires"}),
                Option(("--attached-data-root",), "attached_data_root", {"help": "An extracted data root to use as is"}),
                Option(("--attached-zip",), "attached_zip", {"help": "A local data.zip instead of downloading it"}),
                Option(("--attached-tessera-dir",), "attached_tessera_dir", {"help": "A folder of TESSERA files"}),
                Option(("--attached-tessera-zip",), "attached_tessera_zip", {"help": "A local tessera.zip"}),
            ), windows=True),
)
BY_NAME = {command.name: command for command in COMMANDS}


def add_option(parser, option, function):
    """Add one Option to a parser, defaulting to the master function's own default.

    Switches (store_true/store_false) need no default. An explicit default in
    the options wins.

    Args:
        parser (argparse.ArgumentParser): the parser.
        option (Option): the flag.
        function (Callable): the master whose parameter it sets.

    Returns:
        argparse.ArgumentParser: parser.
    """
    options = dict(option.options)
    if options.get("action") not in ("store_true", "store_false") and "default" not in options:
        options["default"] = parameter_default(function, option.dest)
    parser.add_argument(*option.flags, dest=option.dest, **options)
    return parser


def parse_args(argv=None):
    """Parse the dataset command line: global repo and folder flags, then one subcommand.

    The global flags apply to every subcommand. The window flags (-ts/-it)
    belong to the subcommands that read or write windows.

    Args:
        argv (list[str] | None): the arguments (default: sys.argv).

    Returns:
        argparse.Namespace: the parsed flags, named like the master parameters.
    """
    parser = argparse.ArgumentParser(prog="python -m firepred.cli.dataset", description=__doc__.splitlines()[0])
    # Shared by every subcommand: credentials, the three Hugging Face repos (raw
    # source, windows, TESSERA) and the local working folders. Defaults: process_windows'.
    builder = windows_master.process_windows
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (default: $HF_TOKEN)")
    add_option(parser, Option(("--src-repo",), "src_repo", {"help": "Raw GeoTIFFs + ROI CSVs"}), builder)
    add_option(parser, Option(("--window-repo",), "window_repo",
                              {"help": "Windowed NPZ per fire (default: the repo for -ts/-it)"}), builder)
    add_option(parser, Option(("--tessera-repo",), "tessera_repo", {"help": "Prior-year TESSERA per fire"}),
               tessera_master.process_tessera)
    add_option(parser, Option(("--roi-dir",), "roi_dir"), builder)
    add_option(parser, Option(("--work-dir",), "work_dir"), builder)
    # One subcommand per pipeline step; the module docstring lists them in run order.
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        sub = subparsers.add_parser(command.name, help=command.help)
        if command.windows:
            # Window settings, shared by every command that reads or writes windows.
            add_option(sub, Option(("-ts", "--length"), "length", {"type": int, "help": "Input days per window"}),
                       builder)
            add_option(sub, Option(("-it", "--interval"), "interval",
                                   {"type": int, "help": "Days between window starts"}), builder)
        for option in command.options:
            add_option(sub, option, command.function)
    return parser.parse_args(argv)


def window_repo_of(args):
    """Give --window-repo, or the repo for -ts/-it when it is not set.

    Each window shape has its own repo. This picks it from the window flags.

    Args:
        args (argparse.Namespace): the parsed flags.

    Returns:
        str: the window repo.
    """
    return args.window_repo or window_repo_for(args.length, args.interval)


def call_arguments(command, args):
    """Collect the keyword arguments for a subcommand's master function.

    Every parsed value whose name is a parameter of the function is passed.
    --zip-repo becomes repo_id, and `stage` gets the notebook Workspace.

    Args:
        command (Command): the subcommand.
        args (argparse.Namespace): the parsed flags.

    Returns:
        dict: the keyword arguments.
    """
    values = vars(args)
    if "zip_repo" in values:  # --zip-repo picks between the window repo and the TESSERA repo
        values["repo_id"] = {"window": window_repo_of(args), "tessera": args.tessera_repo}[args.zip_repo]
    if command.name == "stage":  # the training notebooks' paths, from --work-dir and the window settings
        values["work"] = windows_master.workspace(args.work_dir, args.length, args.interval, args.use_tessera,
                                                  args.window_repo)
    parameters = inspect.signature(command.function).parameters
    return {name: value for name, value in values.items() if name in parameters}


def run(args):
    """Run the parsed subcommand through its master function.

    The master prints its own progress. Its report is returned.

    Args:
        args (argparse.Namespace): the parsed flags.

    Returns:
        object: the master's report.
    """
    command = BY_NAME[args.command]
    return command.function(**call_arguments(command, args))


def main(argv=None):
    """Parse the command line and run the subcommand.

    This is `python -m firepred.cli.dataset`. The report is returned for callers
    that run it in Python.

    Args:
        argv (list[str] | None): the arguments (default: sys.argv).

    Returns:
        object: the master's report.
    """
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
