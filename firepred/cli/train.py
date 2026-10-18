"""Train or test a next-day fire prediction model.

Level 4 (cli). `python -m firepred.cli.train -m swinunetr3d -mode pred -b 8 -r 1
-lr 0.001 -nh 3 -ed 36 -ts 6 -it 1 --data-root <data>`; see --help. The flags
are those of the former run_spatial_temp_model_pred.py, plus --model-option.

Every flag is defined once, in firepred/training/flags.py: the parser, the
conversion to one frozen RunConfig (config_from_args) and back to a command
line (argv_for, so a notebook can build a RunConfig and launch the same run in
a fresh process) are all built from that table. Experiment(config).run() does
the rest (firepred/training/experiment.py); write_run_record saves what was run.
"""

import json
import sys
from pathlib import Path

from firepred.training.flags import argv_for, build_parser, config_from_args


def parse_args(argv=None):
    """Parse the training command line.

    Every flag is defined in training/flags.py. The parsed flags are never
    modified afterwards.

    Args:
        argv (list[str] | None): the arguments (default: sys.argv).

    Returns:
        argparse.Namespace: the parsed flags.
    """
    return build_parser().parse_args(argv)


def write_run_record(output_dir, config, code=None):
    """Save what a run is about to do: its command, its settings and the library versions.

    The files sit beside the checkpoints, so a run can be repeated later.
    Secrets and environment values are never written.

    Args:
        output_dir (str | Path): the folder to write into.
        config (RunConfig): the run's settings.
        code (dict | None): extra provenance (repo, branch, commit, data repo).

    Returns:
        list[Path]: the written files.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", "firepred.cli.train", *argv_for(config)]
    settings = {
        **(code or {}), "run_name": config.run_name, "model": config.model,
        "model_options": dict(config.extra_model_options), "model_channels": config.input_channels,
        "ts_length": config.ts_length, "ts_interval": config.ts_interval, "use_tessera": config.tessera,
        "learning_rate": config.learning_rate, "test_every": config.test_every,
        **config.loss.record(), **config.schedule.record(),
    }
    paths = {
        "training-command.json": json.dumps(command, indent=2),
        "code-version.json": json.dumps(settings, indent=2, default=str),
        "runtime.txt": f"Python: {sys.version}\nPyTorch: {torch.__version__}\n",
    }
    for name, text in paths.items():
        (output_dir / name).write_text(text)
    return [output_dir / name for name in paths]


def main(argv=None):
    """Train or test as the command line says.

    The flags become a RunConfig and Experiment(config).run() does the rest. The
    W&B run is finished when it returns.

    Args:
        argv (list[str] | None): the arguments (default: sys.argv).

    Returns:
        RunResult: the tested checkpoint and the final scores.
    """
    from firepred.training.experiment import Experiment

    return Experiment(config_from_args(parse_args(argv))).run()


if __name__ == "__main__":
    main()
