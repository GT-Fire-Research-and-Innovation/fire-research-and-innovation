"""Build the TS-SatFire next-day prediction dataset through Hugging Face.

Level 4 (cli). Each subcommand calls one master function (prep/dataset_master.py,
prep/tessera_master.py) and prints its report. The token
defaults to the HF_TOKEN environment variable. Defaults come from
firepred/config/data.py: -ts/-it (window length and stride) and the
dataset repos. --window-repo defaults to the repo for the chosen -ts/-it
(window_repo_for). The flags are those of the former dataset_gen_pred.py.

    python -m firepred.cli.dataset process -ts 6 -it 1
    python -m firepred.cli.dataset tessera [--only-fires 21890003 ...]
    python -m firepred.cli.dataset tessera-zip
    python -m firepred.cli.dataset pull-zip --data-root ./data && python -m firepred.cli.dataset pull-tessera --data-root ./data
    python -m firepred.cli.dataset download -ts 6 -it 1 --data-root ./data [--tessera]
    python -m firepred.cli.dataset upload-zip --data-root ./data [--zip-repo tessera --zip-name data_tessera.zip]
    python -m firepred.cli.dataset pull-zip --data-root ./data [--zip-repo window|tessera]
"""

import argparse
import os

from firepred.config.data import (
    SPLITS,
    SRC_REPO,
    TESSERA_REPO,
    TRAIN_VAL_LABEL_SEL,
    TS_INTERVAL,
    TS_LENGTH,
    window_repo_for,
)
from firepred.prep import dataset_master, tessera_master


ZIP_REPOS = ("window", "tessera")


def parse_args(argv=None):
    """Parse the dataset command line (global repo/folder flags, then one subcommand)."""
    parser = argparse.ArgumentParser(prog="python -m firepred.cli.dataset", description=__doc__.splitlines()[0])
    # Shared by every subcommand: credentials, the three Hugging Face repos (raw
    # source, windows, TESSERA) and the local working folders.
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (default: $HF_TOKEN)",
    )
    parser.add_argument("--src-repo", default=SRC_REPO, help="Raw GeoTIFFs + ROI CSVs")
    parser.add_argument(
        "--window-repo",
        dest="window_repo",
        default=None,
        help="Windowed NPZ per fire (default: the repo for -ts/-it)",
    )
    parser.add_argument("--tessera-repo", default=TESSERA_REPO, help="Prior-year TESSERA per fire")
    parser.add_argument("--roi-dir", default="hf_roi")
    parser.add_argument("--work-dir", default=".")
    # One subcommand per pipeline step; the module docstring lists them in run order.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Window settings, shared by every command that reads or writes windows.
    windows = argparse.ArgumentParser(add_help=False)
    windows.add_argument(
        "-ts",
        "--length",
        dest="ts",
        type=int,
        default=TS_LENGTH,
        help=f"Input days per window (default {TS_LENGTH})",
    )
    windows.add_argument(
        "-it",
        "--interval",
        dest="it",
        type=int,
        default=TS_INTERVAL,
        help=f"Days between window starts (default {TS_INTERVAL})",
    )

    # process (step 1): raw GeoTIFFs -> one windowed NPZ archive per fire in the
    # window repo. The label flags pick the burned-area mask (with or without NIFC).
    process = subparsers.add_parser(
        "process",
        parents=[windows],
        help="Stream raw fires from --src-repo into windowed NPZs in --window-repo",
    )
    process.add_argument(
        "-mode",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
    )
    process.add_argument("--batch-size", type=int, default=10)
    process.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild fires already in --window-repo and rewrite its window_config.json "
        "(window length/stride, label rules, crop, split) with the settings of this run",
    )
    process.add_argument("--debug", action="store_true")
    process.add_argument(
        "--ignore-roi-label-sel",
        action="store_true",
        help="Use --train-val-label-sel for test fires too instead of their ROI value",
    )
    process.add_argument(
        "--train-val-label-sel",
        type=int,
        choices=(0, 1),
        default=TRAIN_VAL_LABEL_SEL,
        help="Train/val burned-area mask: 0 = AF + NIFC perimeters (paper), 1 = AF only (released code)",
    )

    # Optional TESSERA steps: build per-fire embeddings (tessera), bundle them into
    # one zip (tessera-zip), unpack it beside the window data (pull-tessera).
    tessera = subparsers.add_parser(
        "tessera",
        help="Build prior-year TESSERA embeddings per fire into --tessera-repo",
    )
    tessera.add_argument(
        "-mode",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
    )
    tessera.add_argument("--batch-size", type=int, default=1)
    tessera.add_argument("--strip-rows", type=int, default=128)
    tessera.add_argument("--overwrite", action="store_true")
    tessera.add_argument("--only-fires", nargs="+", help="Restrict to these fire IDs (timing test)")

    subparsers.add_parser(
        "tessera-zip",
        help="Bundle --tessera-repo tessera/ and the drop list into tessera.zip there",
    )
    pull_tessera = subparsers.add_parser(
        "pull-tessera",
        help="Unpack tessera.zip into --data-root/tessera and drop 2017 window archives",
    )
    pull_tessera.add_argument("--data-root", default="./data")
    pull_tessera.add_argument("--keep-excluded", action="store_true")

    # Move window archives between the repo and a local data root: file by file
    # (download), or as one zip (upload-zip / pull-zip): one file to fetch instead
    # of one per fire.
    download = subparsers.add_parser(
        "download",
        parents=[windows],
        help="Download --window-repo NPZs into --data-root/{train,val,test}",
    )
    download.add_argument("--data-root", default="./data")
    download.add_argument(
        "--tessera",
        action="store_true",
        help="TESSERA mode: drop fires without prior-year TESSERA and download "
        "--tessera-repo embeddings into --data-root/tessera",
    )

    upload_zip = subparsers.add_parser(
        "upload-zip",
        parents=[windows],
        help="Zip --data-root and upload it to --zip-repo",
    )
    upload_zip.add_argument("--data-root", default="./data")
    upload_zip.add_argument("--zip-path", default="./data.zip")
    upload_zip.add_argument("--zip-repo", choices=ZIP_REPOS, default="window")
    upload_zip.add_argument("--zip-name", default="data.zip", help="data_tessera.zip for TESSERA mode")

    pull_zip = subparsers.add_parser(
        "pull-zip",
        parents=[windows],
        help="Download a zip from --zip-repo and extract it into --data-root",
    )
    pull_zip.add_argument("--data-root", default="./data")
    pull_zip.add_argument("--zip-repo", choices=ZIP_REPOS, default="window")
    pull_zip.add_argument("--zip-name", default="data.zip")

    return parser.parse_args(argv)


def window_repo_of(args):
    """--window-repo, or the repo named after -ts/-it (each window shape has its own repo)."""
    return args.window_repo or window_repo_for(args.ts, args.it)


def run(args):
    """Run the parsed subcommand through its master function and return the result."""
    if args.command == "process":
        return dataset_master.process_windows(
            args.token, src_repo=args.src_repo, window_repo=window_repo_of(args), batch_size=args.batch_size,
            length=args.ts, interval=args.it, splits=args.mode, roi_dir=args.roi_dir, work_dir=args.work_dir,
            overwrite=args.overwrite, debug=args.debug, use_roi_label_sel=not args.ignore_roi_label_sel,
            train_val_label_sel=args.train_val_label_sel,
        )
    if args.command == "tessera":
        return tessera_master.process_tessera(
            args.token, src_repo=args.src_repo, tessera_repo=args.tessera_repo, splits=args.mode,
            batch_size=args.batch_size, work_dir=args.work_dir, roi_dir=args.roi_dir, overwrite=args.overwrite,
            strip_rows=args.strip_rows, only_fires=args.only_fires,
        )
    if args.command == "tessera-zip":
        return tessera_master.upload_tessera_zip(
            args.token, src_repo=args.src_repo, tessera_repo=args.tessera_repo, work_dir=args.work_dir,
            roi_dir=args.roi_dir,
        )
    if args.command == "pull-tessera":
        return tessera_master.pull_tessera_zip(
            args.data_root, token=args.token, tessera_repo=args.tessera_repo, drop_excluded=not args.keep_excluded,
        )
    if args.command == "download":
        return dataset_master.download_windows(
            args.token, window_repo=window_repo_of(args), src_repo=args.src_repo, data_root=args.data_root,
            roi_dir=args.roi_dir, include_tessera=args.tessera, tessera_repo=args.tessera_repo,
        )
    # --zip-repo picks between the window repo and the TESSERA repo.
    zip_repo = {"window": window_repo_of(args), "tessera": args.tessera_repo}[args.zip_repo]
    if args.command == "upload-zip":
        return dataset_master.upload_data_zip(
            args.token, data_root=args.data_root, zip_path=args.zip_path, repo_id=zip_repo, zip_name=args.zip_name,
        )
    # pull-zip: the zips store their folders (train/, val/, test/, ...) at the root.
    return dataset_master.pull_data_zip(args.data_root, token=args.token, repo_id=zip_repo, zip_name=args.zip_name)


def main(argv=None):
    """Parse the command line and run the subcommand; returns its report."""
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
