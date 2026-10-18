"""Train or test a next-day fire prediction model.

Level 4 (cli). `python -m firepred.cli.train -m swinunetr3d -mode pred -b 8 -r 1
-lr 0.001 -nh 3 -ed 36 -ts 6 -it 1 --data-root <data>`; see --help. The flags
are those of the former run_spatial_temp_model_pred.py, plus --model-option.
Everything after parsing is firepred/training/run.py.
"""

import argparse

import wandb

from firepred.config.paths import DEFAULT_TEST_CSV
from firepred.config.training import (
    DEFAULT_EPOCHS,
    DEFAULT_LOAD_EPOCH,
    DEFAULT_NUM_WORKERS,
    DEFAULT_UNZIP_DIR,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    LAMBDA_CE,
    LAMBDA_DICE,
    LAMBDA_FOCAL,
    LOSSES,
    LR_DECAY_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    LR_SCHEDULES,
    PAPER_CE_WEIGHT,
    SEED,
    TESSERA_MIN_COVERAGE,
    WANDB_ENTITY,
)
from firepred.models.spec import parse_model_option
from firepred.training.accelerator import DEVICES
from firepred.training.run import run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m firepred.cli.train",
        description="Train or test spatial-temporal fire prediction models"
    )
    parser.add_argument("-m", required=True, help="Model name")
    parser.add_argument("-mode", required=True, help="Task name, e.g. pred")
    parser.add_argument("-b", type=int, required=True, help="Batch size")
    parser.add_argument("-r", type=int, required=True, help="Run number")
    parser.add_argument("-lr", type=float, required=True, help="Learning rate")
    parser.add_argument("-nh", type=int, required=True, help="Number of heads")
    parser.add_argument("-ed", type=int, required=True, help="Embedding size")
    parser.add_argument(
        "-nc",
        type=int,
        default=None,
        help="Model input channels after preprocessing: 43, or 172 with "
        "--tessera. Derived when omitted; checked when given.",
    )
    parser.add_argument("-ts", type=int, required=True, help="Time-series length")
    parser.add_argument(
        "-it",
        type=int,
        required=True,
        help="Generation interval; retained for experiment/checkpoint naming",
    )
    parser.add_argument(
        "-test",
        dest="run_test",
        action="store_true",
        help="Run test evaluation instead of training",
    )
    parser.add_argument("-seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="auto",
        help="auto (default): the GPU when visible, else the CPU. NEW: tpu trains on "
        "one TPU core/chip through PyTorch/XLA (torch_xla); results are close to, "
        "not identical to, a GPU run",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Directory containing train/, val/, and test/ NPZ folders",
    )
    parser.add_argument(
        "--unzip-dir",
        default=DEFAULT_UNZIP_DIR,
        help="Where the window archives are unzipped once, so each window is "
        "read without decompressing (same arrays). 'off' (default): read the "
        "archives; 'auto': <data-root>-npy beside the data root; or a path. "
        "Skipped when the data root is already unzipped or the disk lacks space.",
    )
    parser.add_argument(
        "--unetr-version",
        choices=("v0", "v1"),
        default="v1",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for -test; overrides the generated path",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow full unpickling of the -test checkpoint (it can run code stored "
        "in the file). Needed only for checkpoints with TESSERA statistics on "
        "torch < 2.4; use only for checkpoints you or your runs created",
    )
    parser.add_argument("--load-epoch", type=int, default=DEFAULT_LOAD_EPOCH)
    parser.add_argument(
        "--test-csv",
        default=str(DEFAULT_TEST_CSV),
    )
    parser.add_argument("--plot-dir", default="evaluation_plot")
    parser.add_argument(
        "--test-every",
        type=int,
        default=0,
        help="NEW, not in the paper: also score the test fires with the current "
        "weights every N epochs (0 = never). Logging only: no gradients, eval "
        "mode, and checkpoint choice still uses validation loss",
    )
    parser.add_argument(
        "--test-after-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, score the test fires with the lowest-validation-loss "
        "checkpoint and log test F1/IoU to the same W&B run",
    )
    parser.add_argument(
        "--tessera",
        action="store_true",
        help="Append per-fire TESSERA embeddings (128 bands + validity mask)",
    )
    parser.add_argument(
        "--tessera-dir",
        default=None,
        help="Folder of tessera/p<id>.npz files; default <data-root>/tessera",
    )
    parser.add_argument(
        "--loss",
        choices=LOSSES,
        default=None,
        help="dice: released TS-SatFire code. dicece: the paper's prediction "
        "loss, 1:1 Dice + CE with --ce-weight (use with -lr 0.001). "
        "dicefocal (NEW, not in the paper): --lambda-dice x Dice + "
        "--lambda-focal x focal loss on the fire channel, no CE. Default: "
        "the paper's choice per model, dice for attunet and dicece otherwise",
    )
    parser.add_argument(
        "--swin-temporal-downsample",
        action="store_true",
        help="swinunetr3d: also halve the days in patch merging, as the released "
        "code does. Default follows the paper: no temporal downsampling",
    )
    parser.add_argument(
        "--ce-weight",
        type=float,
        nargs=2,
        default=list(PAPER_CE_WEIGHT),
        metavar=("BACKGROUND", "FIRE"),
        help="dicece class weights (default: the paper's 1.0 446.7836)",
    )
    parser.add_argument("--lambda-dice", type=float, default=LAMBDA_DICE)
    parser.add_argument("--lambda-ce", type=float, default=LAMBDA_CE)
    parser.add_argument("--lambda-focal", type=float, default=LAMBDA_FOCAL,
                        help="dicefocal only: weight of the focal term (--lambda-dice weighs the Dice term)")
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA,
                        help="dicefocal only: focusing exponent; 0 makes the focal term plain binary cross entropy")
    parser.add_argument("--focal-alpha", type=float, default=FOCAL_ALPHA,
                        help="dicefocal only: weight of fire pixels in the focal term (e.g. 0.25); default off")
    # [ours] the TS-SatFire paper and the official code keep the rate constant.
    parser.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULES,
        default="none",
        help="NEW, not in the paper: learning-rate decay. none = constant rate "
        "(paper and official code); plateau = multiply by --lr-decay-factor "
        "after --lr-patience epochs without a lower validation loss; cosine = "
        "smooth decay to --lr-min over --epochs",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=LR_DECAY_FACTOR,
        help="NEW, plateau only: multiply the learning rate by this",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=LR_PATIENCE,
        help="NEW, plateau only: epochs without a lower validation loss before decaying",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=LR_MIN,
        help="NEW, plateau and cosine: the lowest learning rate",
    )
    parser.add_argument(
        "--upload-checkpoints",
        action="store_true",
        help="Also store each best checkpoint as a versioned W&B model Artifact",
    )
    parser.add_argument(
        "--tessera-min-coverage",
        type=float,
        default=TESSERA_MIN_COVERAGE,
        help="TESSERA pixels below this coverage are zeroed and masked",
    )
    parser.add_argument(
        "--model-option",
        action="append",
        type=parse_model_option,
        default=None,
        metavar="KEY=VALUE",
        help="Model-specific setting passed to the model builder as "
        "ModelSpec.options[KEY]; repeatable. VALUE is read as a Python literal "
        "when possible (3, 0.5, True, (1, 2)), else as text",
    )
    parser.add_argument("--wandb-user", default=WANDB_ENTITY)
    args = parser.parse_args(argv)
    # Repeated --model-option flags become one {KEY: VALUE} dict (or None).
    if args.model_option is not None:
        args.model_option = dict(args.model_option)
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
    wandb.finish()
