"""Build the channel lab's cache: every train and val window, cropped, exactly as SwinUNETR sees it.

Run once from the repository root:

    python -m lab.cache --data-root R --tessera-dir R/tessera --out C

Writes C/data.npy (N, 43, T, P, P) fp16, C/static.npy (N, 129, P, P) fp16,
C/label.npy (N, P, P) uint8 and C/fire.npy (N,) int16 (each window's fire, for folds).
The test split is left out on purpose: it stays clean for confirming on SwinUNETR.
"""

import argparse
from pathlib import Path

import numpy as np

from firepred.data.archive import compute_tessera_stats, find_fire_sources, fire_name, load_tessera_maps
from firepred.data.fire_dataset import FireDataset
from firepred.data.stats import MissingStatistics, WindowConfig


def crop_corner(fire, size, crop, rng):
    """Top-left corner of a crop around a random fire pixel, jittered so the fire is not always central."""
    rows, cols = np.nonzero(fire)
    if len(rows) == 0:
        return (size - crop) // 2, (size - crop) // 2
    pick = rng.integers(len(rows))
    jitter = rng.integers(-crop // 2 + 1, crop // 2 + 1, size=2)
    return tuple(int(np.clip(center + shift - crop // 2, 0, size - crop))
                 for center, shift in zip((rows[pick], cols[pick]), jitter))


def main():
    """Read the train and val windows through FireDataset and write their crops to the cache."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tessera-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop", type=int, default=64)
    parser.add_argument("--ts-length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # Same statistics as training: window_config.json's, else the paper's (None).
    try:
        stats = WindowConfig.read(args.data_root).normalization(args.data_root / "train")[0]
        normalization = (stats.mean, stats.std)
    except MissingStatistics:
        normalization = None
    train_maps = load_tessera_maps(args.tessera_dir, find_fire_sources(args.data_root / "train"))
    tessera_stats = compute_tessera_stats(train_maps.values())
    datasets = [
        FireDataset(args.data_root / split, args.ts_length, target_is_single_day=True,
                    tessera_dir=args.tessera_dir, tessera_stats=tessera_stats,
                    tessera_maps=train_maps if split == "train" else None, normalization=normalization)
        for split in ("train", "val")
    ]

    n = sum(len(dataset) for dataset in datasets)
    channels, days, size, _ = datasets[0][0]["data"].shape
    p = args.crop
    args.out.mkdir(parents=True, exist_ok=True)
    shapes = {"data": ((n, channels, days, p, p), np.float16), "static": ((n, 129, p, p), np.float16),
              "label": ((n, p, p), np.uint8), "fire": ((n,), np.int16)}
    out = {name: np.lib.format.open_memmap(args.out / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
           for name, (shape, dtype) in shapes.items()}

    fires = {}
    i = 0
    for dataset in datasets:
        for index, (source, _) in enumerate(dataset.index):
            sample = dataset[index]
            fire = sample["labels"][1].numpy() > 0
            top, left = crop_corner(fire, size, p, rng)
            rows, cols = slice(top, top + p), slice(left, left + p)
            out["data"][i] = sample["data"][..., rows, cols].numpy()
            out["static"][i] = sample["static"][:, rows, cols].numpy()
            out["label"][i] = fire[rows, cols]
            out["fire"][i] = fires.setdefault(fire_name(source), len(fires))
            i += 1
            if i % 500 == 0:
                print(f"{i}/{n} windows", flush=True)
    for array in out.values():
        array.flush()
    size_gb = sum(array.nbytes for array in out.values()) / 1e9
    print(f"Cached {n} windows from {len(fires)} fires, crop {p}, {size_gb:.2f} GB in {args.out}")


if __name__ == "__main__":
    main()
