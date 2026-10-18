"""Prove a source tree computes exactly what a baseline tree computes.

    python tests/equivalence/compare.py --baseline <baseline checkout> [--tree .] \
        [--tier quick|full] [--work <scratch dir>] [--reuse-baseline] [--allow KEY_PREFIX ...]

Runs snapshot.py once per tree (separate processes, CPU only, fixed thread
count) and compares the two snapshots value by value: arrays by dtype, shape
and bytes, floats bit for bit, dicts in key order. Exit code 0 means
identical. --allow skips keys that differ on purpose (see README.md).
"""

import argparse
import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_snapshot(tree, tier, work, out, threads, foreign=None):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        MPLBACKEND="Agg",
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        PYTHONHASHSEED="0",
        WANDB_MODE="disabled",
        PYTHONWARNINGS="ignore",
    )
    command = [sys.executable, str(HERE / "snapshot.py"), "--tree", str(tree), "--tier", tier,
               "--work", str(work), "--out", str(out), "--threads", str(threads)]
    if foreign is not None:
        command += ["--foreign-checkpoint", str(foreign)]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode:
        sys.stderr.write(result.stdout[-4000:] + result.stderr[-8000:])
        raise SystemExit(f"snapshot failed for {tree}")
    # Snapshots are written by snapshot.py in this same run; never load others.
    with open(out, "rb") as handle:
        return pickle.load(handle)


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return struct.pack("<d", a) == struct.pack("<d", b)
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    return a == b


def first_difference(a, b, path=""):
    """Describe where two plain values first differ."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and type(a) is type(b):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for index, (x, y) in enumerate(zip(a, b)):
            if not same(x, y):
                return first_difference(x, y, f"{path}[{index}]")
    text_a, text_b = repr(a), repr(b)
    return f"{path}: {text_a[:300]} vs {text_b[:300]}"


def compare(baseline, current, allow):
    problems = []
    for key in sorted(set(baseline) | set(current)):
        if key == "layout" or key == "own_checkpoint_path" or any(key.startswith(prefix) for prefix in allow):
            continue
        if key not in baseline or key not in current:
            problems.append(f"{key}: only in {'current' if key in current else 'baseline'}")
        elif not same(baseline[key], current[key]):
            problems.append(f"{key}: {first_difference(baseline[key], current[key])}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, help="Checkout of the reference commit")
    parser.add_argument("--tree", default=str(HERE.parent.parent), help="Checkout under test")
    parser.add_argument("--tier", choices=("quick", "full", "hub"), default="quick")
    parser.add_argument("--work", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "firepred-equivalence"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reuse-baseline", action="store_true", help="Reuse a baseline snapshot from --work")
    parser.add_argument("--cross-test", action="store_true",
                        help="full tier: also test each tree with the other tree's checkpoint")
    parser.add_argument("--allow", nargs="*", default=[], help="Key prefixes allowed to differ")
    args = parser.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    baseline_file = work / f"baseline-{args.tier}.pkl"
    if args.reuse_baseline and baseline_file.exists():
        with open(baseline_file, "rb") as handle:  # written by an earlier run of this script
            baseline = pickle.load(handle)
    else:
        baseline = run_snapshot(args.baseline, args.tier, work / f"baseline-{args.tier}", baseline_file, args.threads)
    foreign = baseline.get("own_checkpoint_path") if args.cross_test else None
    current = run_snapshot(args.tree, args.tier, work / f"current-{args.tier}", work / f"current-{args.tier}.pkl",
                           args.threads, foreign)
    if args.cross_test:
        baseline = run_snapshot(args.baseline, args.tier, work / f"baseline-{args.tier}-x", baseline_file,
                                args.threads, current.get("own_checkpoint_path"))

    print(f"baseline layout: {baseline['layout']}; current layout: {current['layout']}; "
          f"{len(current)} keys compared")
    problems = compare(baseline, current, args.allow)
    for problem in problems:
        print("DIFF", problem)
    print("IDENTICAL" if not problems else f"{len(problems)} DIFFERENCES")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
