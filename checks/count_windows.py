"""Count prepared dataset windows and batches per epoch without reading image arrays.

    python checks/count_windows.py --data-root ./data --batch-size 8
    python checks/count_windows.py --data-root ./data --splits train --per-fire
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firepred.config.dataset import SPLITS  # noqa: E402


def count_windows(data_root, splits=SPLITS):
    """Return {split: {fire_id: window_count}} using the trainer's archive index.

    Archives and extracted fire folders are discovered exactly as in FireDataset.
    Only metadata and member names are read, so image arrays stay compressed.
    """
    root = Path(data_root).expanduser()
    if not root.is_dir():
        raise ValueError(f"Data root is not a directory: {root}")
    from firepred.data.archive import find_fire_sources, fire_name, index_fires

    counts = {
        split: {
            fire_name(source).removeprefix("p"): len(index_fires([source]))
            for source in find_fire_sources(root / split)
        }
        for split in splits
    }
    if not any(counts.values()):
        raise ValueError(f"No fire archives or extracted fire folders found in {root} for {', '.join(splits)}")
    return counts


def main(argv=None):
    """Print split totals and optional per-fire counts for a prepared dataset."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing train/, val/, test/")
    parser.add_argument("--batch-size", type=int, default=8, help="Windows per batch (default: 8)")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--per-fire", action="store_true", help="Also list each fire's window count")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    try:
        counts = count_windows(args.data_root, args.splits)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Data root: {args.data_root.expanduser().resolve()}")
    print(f"Batch size: {args.batch_size}\n")
    print(f"{'Split':<8} {'Fires':>8} {'Windows':>10} {'Batches/epoch':>15}")
    for split, fires in counts.items():
        if not (args.data_root.expanduser() / split).is_dir():
            print(f"{split:<8} (missing directory)")
            continue
        windows = sum(fires.values())
        batches = (windows + args.batch_size - 1) // args.batch_size
        print(f"{split:<8} {len(fires):>8,} {windows:>10,} {batches:>15,}")
    print("\nEvery training window is visited once per epoch; a partial final batch is included.")
    print("Training batches are optimizer steps; validation/test batches only evaluate.")
    print("Test counts include all local fires; evaluation may filter them using the test CSV.")
    if args.per_fire:
        print(f"\n{'Split':<8} {'Fire ID':<40} {'Windows':>10}")
        for split, fires in counts.items():
            for fire_id, windows in fires.items():
                print(f"{split:<8} {fire_id:<40} {windows:>10,}")
    return counts


if __name__ == "__main__":
    main()
