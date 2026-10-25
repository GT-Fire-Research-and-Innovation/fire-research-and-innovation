"""Train the channel lab's tiny model on one fire-fold and append its held-out score to results.csv.

    python -m lab.train --cache C --fold 0 [--drop G,..] [--keep G,..] [--random-bands K] [--gate LAMBDA] [--seed S]

Channels are named by group (GROUPS: "weather", "tessera", ...) or as t0..t127 for
single TESSERA bands. Dropped channels are zeroed in training and evaluation, which
is the training mean in standardized space, so a run with --drop G is
drop-and-retrain. --random-bands K keeps K random TESSERA bands (the control for a
selected subset). --gate puts a stochastic gate (Yamada et al. 2020) on every TESSERA
band; the bands left open are written to results.csv, ranked by gate value. On
synthetic data, 1e-4 to 1e-3 kept the planted band and closed the rest; 1e-2 closed
every gate before the model learned to use any.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from firepred.data.fire_dataset import batch_inputs

# Index ranges in the 172-channel input SwinUNETR sees (43 base, 128 TESSERA bands, 1 mask).
GROUPS = {
    "base": list(range(43)),
    "viirs_reflectance": [0, 1, 2, 5],
    "viirs_fire": [3, 4, 6, 7],
    "vegetation": [8, 9],
    "weather": list(range(10, 17)),
    "topography": [17, 18, 19],
    "drought": [20],
    "landcover": list(range(21, 38)),
    "forecast": list(range(38, 43)),
    "tessera": list(range(43, 171)),
    "tessera_mask": [171],
} | {f"t{band}": [43 + band] for band in range(128)}
CHANNELS = 172
BANDS = slice(43, 171)


class Tiny(nn.Module):
    """About 50k weights: mix channels per day, collapse the days, four dilated 3x3 layers, one fire logit."""

    def __init__(self, days, width=32):
        """Build the layers for `days` input days."""
        super().__init__()
        self.mix = nn.Conv3d(CHANNELS, width, 1)
        self.days = nn.Conv3d(width, width, (days, 1, 1))
        layers = []
        for dilation in (1, 2, 4, 8):
            layers += [nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation),
                       nn.BatchNorm2d(width), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        """(B, 172, T, H, W) input to (B, H, W) fire logits."""
        x = torch.relu(self.days(torch.relu(self.mix(x)))).squeeze(2)
        return self.head(self.body(x))[:, 0]


class Gates(nn.Module):
    """Stochastic gates: gate_b = clamp(mu_b + noise, 0, 1), penalized by P(gate_b > 0)."""

    SIGMA = 0.5

    def __init__(self, n):
        """One gate per band, half open to start (the paper's initialization)."""
        super().__init__()
        self.mu = nn.Parameter(torch.full((n,), 0.5))

    def forward(self):
        """The gates; noisy while training, deterministic at evaluation."""
        noise = self.SIGMA * torch.randn_like(self.mu) if self.training else 0.0
        return (self.mu + noise).clamp(0, 1)

    def penalty(self):
        """Expected number of open gates."""
        return torch.special.ndtr(self.mu / self.SIGMA).sum()


def channel_list(names):
    """Channel indices of comma-separated group or band names."""
    return sorted({channel for name in names.split(",") if name for channel in GROUPS[name]})


def main():
    """Train on all fires outside the fold, score the fold, append one row to results.csv."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--drop", default="")
    parser.add_argument("--keep", default="")
    parser.add_argument("--random-bands", type=int, default=0)
    parser.add_argument("--gate", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=None, help="CSV to append to (default: CACHE/results.csv)")
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    data = np.load(args.cache / "data.npy", mmap_mode="r")
    static = np.load(args.cache / "static.npy", mmap_mode="r")
    labels = np.load(args.cache / "label.npy")
    held_out = np.load(args.cache / "fire.npy") % args.folds == args.fold
    train_idx, val_idx = np.flatnonzero(~held_out), np.flatnonzero(held_out)

    keep = torch.ones(CHANNELS)
    if args.keep:
        keep.zero_()
        keep[channel_list(args.keep)] = 1
    if args.drop:
        keep[channel_list(args.drop)] = 0
    if args.random_bands:
        keep[BANDS] = 0
        keep[43 + rng.choice(128, args.random_bands, replace=False)] = 1
    keep = keep.to(device)

    model = Tiny(data.shape[2]).to(device)
    gates = Gates(128).to(device) if args.gate else None
    groups = [{"params": model.parameters()}]
    if gates is not None:
        groups.append({"params": gates.parameters(), "lr": 10 * args.lr})  # gates need to move further
    optimizer = torch.optim.Adam(groups, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    fire = labels[train_idx].mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - fire) / fire, device=device))

    def batch(idx):
        """Model input (with dropped channels zeroed and gates applied) and target of some windows."""
        idx = np.sort(idx)
        x = batch_inputs({"data": torch.from_numpy(data[idx]), "static": torch.from_numpy(static[idx])}, device)
        scale = keep if gates is None else torch.cat((keep[:43], keep[BANDS] * gates(), keep[171:]))
        return x.float() * scale.view(1, -1, 1, 1, 1), torch.from_numpy(labels[idx]).to(device).float()

    for epoch in range(args.epochs):
        model.train()
        if gates is not None:
            gates.train()
        total = 0.0
        order = rng.permutation(train_idx)
        for start in range(0, len(order), args.batch):
            x, y = batch(order[start:start + args.batch])
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(x).float(), y)
            if gates is not None:
                loss = loss + args.gate * gates.penalty()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / -(-len(order) // args.batch):.4f}")

    model.eval()
    if gates is not None:
        gates.eval()
    probabilities, targets = [], []
    with torch.no_grad(), torch.autocast(device.type, enabled=amp):
        for start in range(0, len(val_idx), args.batch):
            x, y = batch(val_idx[start:start + args.batch])
            probabilities.append(torch.sigmoid(model(x).float()).cpu().numpy().ravel())
            targets.append(y.cpu().numpy().ravel() > 0)
    probability, target = np.concatenate(probabilities), np.concatenate(targets)
    predicted = probability > 0.5
    ap = average_precision_score(target, probability)
    f1 = 2 * (predicted & target).sum() / max(1, predicted.sum() + target.sum())

    # TESSERA bands in use at evaluation: ranked by gate value when gated, else in index order.
    weight = (keep[BANDS] if gates is None else keep[BANDS] * gates()).detach().cpu()
    order = torch.argsort(weight, descending=True, stable=True) if gates is not None else range(128)
    bands = [int(band) for band in order if weight[band] > 0]
    row = {"fold": args.fold, "folds": args.folds, "seed": args.seed, "drop": args.drop, "keep": args.keep,
           "random_bands": args.random_bands, "gate": args.gate, "epochs": args.epochs,
           "val_ap": round(float(ap), 5), "val_f1": round(float(f1), 5),
           "seconds": round(time.time() - started, 1), "n_bands": len(bands), "bands": " ".join(map(str, bands))}
    results = args.results or args.cache / "results.csv"
    new = not results.exists()
    with results.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    print(f"fold {args.fold}: val AP {ap:.4f}, F1 {f1:.4f}, {len(bands)} TESSERA bands, "
          f"{row['seconds']} s -> {results}")


if __name__ == "__main__":
    main()
