"""The training command line, defined once: every flag, the RunConfig field it sets, its help.

Level 3 (training). Used by cli/train.py (parse_args, config_from_args,
argv_for) and cli/check.py (add_flags for the model and loss flags).

FLAGS is the single source. Each Flag names the RunConfig field it fills
("loss.ce_weight" is a field of the nested LossConfig); its default comes from
that field unless the flag needs another one (None where the field is filled
later, e.g. --loss picks the paper's loss for the model). The parser, the
args -> RunConfig conversion and the RunConfig -> argv conversion are all
built from this table, so adding a flag is one entry here plus one field.
Nothing here imports torch.
"""

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from firepred.config.dataset import DEFAULT_TEST_CSV
from firepred.config.training import LOSSES, LR_SCHEDULES, PAPER_CE_WEIGHT
from firepred.models.registry import parse_model_option
from firepred.training.accelerator import DEVICES
from firepred.training.config import LossConfig, RunConfig, ScheduleConfig

FROM_FIELD = object()   # default: the RunConfig field's own default
OMIT = object()         # a converted value that leaves the field at its default
NESTED = {"loss": LossConfig, "schedule": ScheduleConfig}


@dataclass(frozen=True)
class Flag:
    """Describe one training flag: how it is written, which RunConfig field it sets, its argparse settings.

    The table of these is the single place a training flag is defined. The
    parser, the conversion to a RunConfig and the conversion back to a command
    line are all built from it.

    Attributes:
        flags (tuple[str, ...]): spellings, e.g. ("-b",).
        target (str): the RunConfig field, or "loss.<field>" / "schedule.<field>".
        options (dict): argparse keywords (help, type, choices, ...).
        default (object): the argparse default, or FROM_FIELD for the field's own.
        to_config (Callable | None): turns the parsed value into the field value (or OMIT).
    """

    flags: tuple
    target: str                       # RunConfig field, or "loss.<field>" / "schedule.<field>"
    options: dict = field(default_factory=dict)   # argparse keywords (help, type, choices, ...)
    default: object = FROM_FIELD
    to_config: object = None          # parsed value -> field value (or OMIT)

    @property
    def dest(self):
        """Give the argparse attribute name of the flag.

        It is the explicit dest when given. Otherwise it is derived from the last
        spelling ("--lr-min" becomes "lr_min").

        Returns:
            str: the attribute name.
        """
        return self.options.get("dest") or self.flags[-1].lstrip("-").replace("-", "_")

    @property
    def action(self):
        """Give the flag's argparse action.

        Plain value flags have none. Switches use "store_true", lists use "append".

        Returns:
            object: the action, or None.
        """
        return self.options.get("action")


def path_or_omit(value):
    """Turn an optional path flag into a Path, or OMIT when it was not given.

    OMIT leaves the RunConfig field at its default. A given path has ~
    expanded.

    Args:
        value (str | None): the parsed value.

    Returns:
        Path | object: the path, or OMIT.
    """
    return OMIT if value is None else Path(value).expanduser()


FLAGS = (
    Flag(("-m",), "model", {"required": True, "help": "Model name"}),
    Flag(("-mode",), "mode", {"required": True, "help": "Task name, e.g. pred"}),
    Flag(("-b",), "batch_size", {"type": int, "required": True, "help": "Batch size"}),
    Flag(("-r",), "run_number", {"type": int, "required": True, "help": "Run number"}),
    Flag(("-lr",), "learning_rate", {"type": float, "required": True, "help": "Learning rate"}),
    Flag(("-nh",), "num_heads", {"type": int, "required": True, "help": "Number of heads"}),
    Flag(("-ed",), "feature_size", {"type": int, "required": True, "help": "Embedding size"}),
    Flag(("-nc",), "expected_channels", {"type": int, "help": "Model input channels after preprocessing: 43, or 172 "
                                                             "with --tessera. Derived when omitted; checked when given."}),
    Flag(("-ts",), "ts_length", {"type": int, "required": True, "help": "Time-series length"}),
    Flag(("-it",), "ts_interval", {"type": int, "required": True,
                                   "help": "Generation interval; retained for experiment/checkpoint naming"}),
    Flag(("-test",), "test_only", {"dest": "run_test", "action": "store_true",
                                   "help": "Run test evaluation instead of training"}),
    Flag(("-seed",), "seed", {"type": int}),
    Flag(("--epochs",), "epochs", {"type": int}),
    Flag(("--num-workers",), "num_workers", {"type": int}),
    Flag(("--device",), "device", {"choices": DEVICES, "help": "auto (default): the GPU when visible, else the CPU. "
                                   "NEW: tpu trains on one TPU core/chip through PyTorch/XLA (torch_xla); results are "
                                   "close to, not identical to, a GPU run"}),
    Flag(("--data-root",), "data_root", {"help": "Directory containing train/, val/, and test/ NPZ folders"},
         default=None, to_config=path_or_omit),
    Flag(("--unzip-dir",), "unzip_dir", {"help": "Where the window archives are unzipped once, so each window is read "
                                         "without decompressing (same arrays). 'off' (default): read the archives; "
                                         "'auto': <data-root>-npy beside the data root; or a path. Skipped when the "
                                         "data root is already unzipped or the disk lacks space."}),
    Flag(("--unetr-version",), "unetr_version", {"choices": ("v0", "v1")}),
    Flag(("--checkpoint",), "checkpoint", {"help": "Checkpoint path for -test; overrides the generated path"}),
    Flag(("--trust-checkpoint",), "trust_checkpoint", {
        "action": "store_true", "help": "Allow full unpickling of the -test checkpoint (it can run code stored in the "
        "file). Needed only for checkpoints with TESSERA statistics on torch < 2.4; use only for checkpoints you or "
        "your runs created"}),
    Flag(("--load-epoch",), "load_epoch", {"type": int}),
    Flag(("--test-csv",), "test_csv", {}, default=str(DEFAULT_TEST_CSV), to_config=Path),
    Flag(("--plot-dir",), "plot_dir", {}, default="evaluation_plot", to_config=Path),
    Flag(("--test-every",), "test_every", {"type": int, "help": "NEW, not in the paper: also score the test fires with "
                                           "the current weights every N epochs (0 = never). Logging only: no gradients, "
                                           "eval mode, and checkpoint choice still uses validation loss"}),
    Flag(("--test-after-train",), "test_after_train", {
        "action": argparse.BooleanOptionalAction, "help": "After training, score the test fires with the "
        "lowest-validation-loss checkpoint and log test F1/IoU to the same W&B run"}),
    Flag(("--tessera",), "tessera", {"action": "store_true",
                                     "help": "Append per-fire TESSERA embeddings (128 bands + validity mask)"}),
    Flag(("--tessera-dir",), "tessera_dir", {"help": "Folder of tessera/p<id>.npz files; default <data-root>/tessera"}),
    Flag(("--loss",), "loss.name", {
        "choices": LOSSES, "help": "dice: released TS-SatFire code. dicece: the paper's prediction loss, 1:1 Dice + CE "
        "with --ce-weight (use with -lr 0.001). dicefocal (NEW, not in the paper): --lambda-dice x Dice + "
        "--lambda-focal x focal loss on the fire channel, no CE. Default: the paper's choice per model, dice for "
        "attunet and dicece otherwise"}, default=None),
    Flag(("--swin-temporal-downsample",), "swin_temporal_downsample", {
        "action": "store_true", "help": "swinunetr3d: also halve the days in patch merging, as the released code "
        "does. Default follows the paper: no temporal downsampling"}),
    Flag(("--ce-weight",), "loss.ce_weight", {"type": float, "nargs": 2, "metavar": ("BACKGROUND", "FIRE"),
                                              "help": "dicece class weights (default: the paper's 1.0 446.7836)"},
         default=list(PAPER_CE_WEIGHT), to_config=lambda value: tuple(float(v) for v in value)),
    Flag(("--lambda-dice",), "loss.lambda_dice", {"type": float}),
    Flag(("--lambda-ce",), "loss.lambda_ce", {"type": float}),
    Flag(("--lambda-focal",), "loss.lambda_focal", {
        "type": float, "help": "dicefocal only: weight of the focal term (--lambda-dice weighs the Dice term)"}),
    Flag(("--focal-gamma",), "loss.focal_gamma", {
        "type": float, "help": "dicefocal only: focusing exponent; 0 makes the focal term plain binary cross entropy"}),
    Flag(("--focal-alpha",), "loss.focal_alpha", {
        "type": float, "help": "dicefocal only: weight of fire pixels in the focal term (e.g. 0.25); default off"}),
    Flag(("--lr-schedule",), "schedule.name", {
        "choices": LR_SCHEDULES, "help": "NEW, not in the paper: learning-rate decay. none = constant rate (paper and "
        "official code); plateau = multiply by --lr-decay-factor after --lr-patience epochs without a lower "
        "validation loss; cosine = smooth decay to --lr-min over --epochs"}),
    Flag(("--lr-decay-factor",), "schedule.decay_factor", {
        "type": float, "help": "NEW, plateau only: multiply the learning rate by this"}),
    Flag(("--lr-patience",), "schedule.patience", {
        "type": int, "help": "NEW, plateau only: epochs without a lower validation loss before decaying"}),
    Flag(("--lr-min",), "schedule.min_lr", {"type": float, "help": "NEW, plateau and cosine: the lowest learning rate"}),
    Flag(("--upload-checkpoints",), "upload_checkpoints", {
        "action": "store_true", "help": "Also store each best checkpoint as a versioned W&B model Artifact"}),
    Flag(("--tessera-min-coverage",), "tessera_min_coverage", {
        "type": float, "help": "TESSERA pixels below this coverage are zeroed and masked"}),
    Flag(("--model-option",), "extra_model_options", {
        "action": "append", "type": parse_model_option, "metavar": "KEY=VALUE",
        "help": "Model-specific setting passed to the model builder as ModelSpec.options[KEY]; repeatable. VALUE is "
        "read as a Python literal when possible (3, 0.5, True, (1, 2)), else as text"},
         # Repeated --model-option flags: later ones win, as in a dict.
         default=None, to_config=lambda pairs: tuple(dict(pairs or ()).items())),
    Flag(("--wandb-user",), "wandb_entity", {}),
)
BY_TARGET = {flag.target: flag for flag in FLAGS}


def field_default(target):
    """Look up the default of a RunConfig field, or of a nested LossConfig/ScheduleConfig field.

    Flag defaults come from here, so a default is written only once: on the
    field. A required field (the model name) has none.

    Args:
        target (str): the field, e.g. "batch_size" or "loss.lambda_dice".

    Returns:
        object: the default, or None.
    """
    owner, _, name = target.rpartition(".")
    cls = NESTED[owner] if owner else RunConfig
    spec = next(f for f in dataclasses.fields(cls) if f.name == name)
    if spec.default is not dataclasses.MISSING:
        return spec.default
    if spec.default_factory is not dataclasses.MISSING:
        return spec.default_factory()
    return None  # required field (e.g. the model name)


def add_flags(parser, targets=None, overrides=None):
    """Add training flags to a parser, from the flag table.

    Without targets every flag is added, in table order. cli.check adds a
    subset with overrides (defaults where training requires a value).

    Args:
        parser (argparse.ArgumentParser): the parser.
        targets (list[str] | None): the fields whose flags to add, or None for all.
        overrides (dict | None): {field: argparse keywords that replace the table's}.

    Returns:
        argparse.ArgumentParser: parser.
    """
    overrides = overrides or {}
    for flag in FLAGS if targets is None else [BY_TARGET[t] for t in targets]:
        options = {**flag.options, **overrides.get(flag.target, {})}
        if flag.action != "store_true" and not options.get("required") and "default" not in options:
            options["default"] = field_default(flag.target) if flag.default is FROM_FIELD else flag.default
        parser.add_argument(*flag.flags, **options)
    return parser


def build_parser(prog="python -m firepred.cli.train"):
    """Build the training command-line parser from the flag table.

    Every training flag is added. cli.train parses with it.

    Args:
        prog (str): the program name shown in --help.

    Returns:
        argparse.ArgumentParser: the parser.
    """
    parser = argparse.ArgumentParser(prog=prog, description="Train or test spatial-temporal fire prediction models")
    return add_flags(parser)


def config_from_args(args):
    """Turn parsed training flags into one frozen RunConfig.

    Each flag fills its field, converted when needed; nested loss and schedule
    fields become a LossConfig and a ScheduleConfig. Without --loss the model
    gets the paper's loss, still with any loss flags given.

    Args:
        args (argparse.Namespace): the parsed flags.

    Returns:
        RunConfig: the run's settings.
    """
    values = {"": {}, "loss": {}, "schedule": {}}
    for flag in FLAGS:
        value = getattr(args, flag.dest)
        if flag.to_config is not None:
            value = flag.to_config(value)
        if value is OMIT:
            continue
        owner, _, name = flag.target.rpartition(".")
        values[owner][name] = value
    # Without --loss: the paper's loss for the model, still with any loss flags given.
    values["loss"]["name"] = values["loss"].get("name") or LossConfig.paper_loss(values[""]["model"])
    return RunConfig(**values[""], loss=LossConfig(**values["loss"]), schedule=ScheduleConfig(**values["schedule"]))


def config_value(config, target):
    """Read a field of a RunConfig, following nested loss and schedule fields.

    argv_for uses this to read every flag's value. A dotted target goes one
    level down.

    Args:
        config (RunConfig): the settings.
        target (str): the field, e.g. "loss.ce_weight".

    Returns:
        object: the value.
    """
    value = config
    for part in target.split("."):
        value = getattr(value, part)
    return value


def text(value):
    """Write a value as command-line text.

    Floats use repr so they come back exactly. Everything else uses str.

    Args:
        value (object): the value.

    Returns:
        str: the text.
    """
    return repr(value) if isinstance(value, float) else str(value)


def argv_for(config):
    """Write the training flags that reproduce a RunConfig.

    Parsing the result gives back the same RunConfig. Notebooks use it to build
    a RunConfig in Python and launch the run in a fresh process.

    Args:
        config (RunConfig): the settings.

    Returns:
        list[str]: the flags.
    """
    argv = []
    for flag in FLAGS:
        value = config_value(config, flag.target)
        name = flag.flags[0]
        if flag.action == "store_true":
            argv += [name] if value else []
        elif flag.action is argparse.BooleanOptionalAction:
            argv.append(name if value else f"--no-{name.lstrip('-')}")
        elif flag.action == "append":
            for key, item in value:
                argv += [name, f"{key}={item!r}"]
        elif flag.options.get("nargs") == 2:
            argv += [name, *(repr(float(item)) for item in value)]
        elif value is not None:
            argv += [name, text(value)]
    return argv
