"""Compute golden groups and write them as JSON.

    python tests/golden/capture.py --out tests/golden/values            # every group
    python tests/golden/capture.py --group hub --out /tmp/fresh          # one group

Each group runs in a fresh subprocess (fakes are installed per process), on
the CPU with 4 threads, so a capture is repeatable. test_golden.py calls this
script and compares its output with the stored values. Overwriting the stored
values is a deliberate act: pass --reason, which is appended to _meta.json.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STORED = HERE / "values"
THREADS = 4


def environment():
    """Library versions and settings the stored values were computed with."""
    import monai
    import numpy
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "monai": monai.__version__,
        "threads": THREADS,
    }


def child_env():
    """Environment for a capture subprocess: CPU only, fixed threads and hashing."""
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        MPLBACKEND="Agg",
        OMP_NUM_THREADS=str(THREADS),
        MKL_NUM_THREADS=str(THREADS),
        PYTHONHASHSEED="0",
        WANDB_MODE="disabled",
        PYTHONWARNINGS="ignore",
    )
    return env


def run_group(name, out_dir):
    """Capture group `name` in a subprocess into out_dir/<name>.json and return that path."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.json"
    with tempfile.TemporaryDirectory(prefix=f"golden-{name}-") as work:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", name, "--out", str(target), "--work", work],
            check=True,
            env=child_env(),
        )
    return target


def child(name, out, work):
    """Inside the subprocess: compute one group and write its encoded values."""
    import torch

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parents[1] / "tests" / "support"))
    import fake_wandb

    sys.modules["wandb"] = fake_wandb  # training modules import wandb at module level
    torch.set_num_threads(THREADS)
    import cases
    from encode import encode

    work = Path(work).resolve()
    tree = cases.TREE

    def normalize(text):
        """Replace the scratch and repo paths, which differ between runs."""
        return text.replace(str(work), "<WORK>").replace(str(tree), "<TREE>")

    os.chdir(work)
    context = argparse.Namespace(work=work, tree=tree)
    values = cases.GROUPS[name](context)
    encoded = {key: encode(value, normalize) for key, value in values.items()}
    Path(out).write_text(json.dumps(encoded, indent=0, allow_nan=True) + "\n")
    return Path(out)


def main(argv=None):
    """Parse the command line and capture the requested groups; returns the written paths."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", help="Group to capture (repeatable; default: all)")
    parser.add_argument("--out", default=None, help="Output folder (default: a scratch folder)")
    parser.add_argument("--reason", default=None, help="Required to overwrite the stored values")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--work", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return [child(args.child, args.out, args.work)]

    sys.path.insert(0, str(HERE))
    import cases

    names = args.group or list(cases.GROUPS)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="golden-"))
    if out.resolve() == STORED.resolve() and any(STORED.glob("*.json")) and not args.reason:
        parser.error("refusing to overwrite the stored golden values without --reason")
    written = [run_group(name, out) for name in names]
    if out.resolve() == STORED.resolve():
        meta_path = STORED / "_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"environment": environment(), "updates": []}
        meta["updates"].append({"groups": names, "reason": args.reason or "initial capture"})
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for path in written:
        print(path)
    return written


if __name__ == "__main__":
    main()
