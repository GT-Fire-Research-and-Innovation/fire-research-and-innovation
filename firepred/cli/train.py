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
    """Parse the training command line (training/flags.py defines every flag)."""
    return build_parser().parse_args(argv)


def write_run_record(output_dir, config, code=None):
    """Save the training command, the resolved settings and the library versions in output_dir; return the paths.

    `code` is extra provenance (repo, branch, commit, data repo); no secrets or
    environment values are written.
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
    """Parse the flags, build the RunConfig and run the experiment; returns its RunResult."""
    from firepred.training.experiment import Experiment

    return Experiment(config_from_args(parse_args(argv))).run()


if __name__ == "__main__":
    main()
