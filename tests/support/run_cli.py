"""Run the training CLI in this process with a fake W&B, then dump its calls.

    python tests/support/run_cli.py <case dir> -- <train flags...>

The case dir becomes the working directory and the fake W&B run directory
(where checkpoints are written). The W&B call log goes to <case dir>/calls.json.
test_e2e.py runs this in a subprocess so the fake is installed before any
firepred module imports wandb.
"""

import json
import os
import sys
from pathlib import Path

SUPPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(SUPPORT.parents[1]))
sys.path.insert(0, str(SUPPORT))

import fake_wandb  # noqa: E402

sys.modules["wandb"] = fake_wandb


def main(argv):
    """Run `firepred.cli.train.main` on the flags after `--`; returns the call-log path."""
    case_dir = Path(argv[0]).resolve()
    flags = argv[argv.index("--") + 1:]
    case_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FAKE_WANDB_DIR"] = str(case_dir)
    os.chdir(case_dir)
    import torch

    torch.set_num_threads(4)
    import tiny_model  # noqa: F401  registers "tiny3d"
    from firepred.cli import train

    try:
        train.main(flags)
        if not any(call[0] == "finish" for call in fake_wandb.CALLS):
            fake_wandb.finish()  # cli.train finishes the run only under __main__
    finally:
        out = case_dir / "calls.json"
        out.write_text(json.dumps(fake_wandb.CALLS, default=str, allow_nan=True))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
