"""Quick checks before a long run: a model, the data, one training step, load time.

Level 4 (cli).

    python -m firepred.cli.check model -m swinunetr3d [-nh 3 -ed 36 --size 256 --batch 2]
        Build the model, run random input through it, check the output shape
        contract, take one Adam step; print parameters, time and GPU memory.
    python -m firepred.cli.check data --data-root <root> [--tessera] [--train-step -m swinunetr3d ...]
        Read one training batch and one validation window and check their
        shapes; with --train-step also take one real training step.
    python -m firepred.cli.check timing --data-root <root> [--windows 10]
        Average time to load one training window.

Every check is disposable: nothing is saved, nothing is logged to W&B.
"""

import argparse
import time

import torch
from torch import nn

from firepred.config.data import CROP_SIZE, N_STORED_CHANNELS, TS_LENGTH, model_input_channels
from firepred.config.training import (
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    LAMBDA_CE,
    LAMBDA_DICE,
    LAMBDA_FOCAL,
    LOSSES,
    NUM_CLASSES,
    PAPER_CE_WEIGHT,
    SEED,
)
from firepred.data.archive import find_fire_sources
from firepred.data.dataset import (
    FireDataset,
    batch_inputs,
    compute_tessera_stats,
    load_tessera_maps,
    make_dataloader,
)
from firepred.models.registry import ModelSpec, available_models, build_model, forward_model, parse_model_option
from firepred.training.losses import build_criterion, compute_loss
from firepred.training.optim import build_grad_scaler, build_optimizer


# Device helpers: 'auto' picks the GPU when there is one; peak GPU memory is
# reset before, and printed after, the step being measured.
def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def reset_peak_memory(device):
    if device.type == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)


def report_peak_memory(device):
    if device.type != "cuda":
        return
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
        print(f"GPU {index} peak allocated memory: {torch.cuda.max_memory_allocated(index) / 1024**3:.2f} GiB")


def training_step(model, inputs, labels, args, device):
    """One forward/backward/Adam step; returns (logits, loss, components)."""
    model.train()
    # The same pieces the trainer uses (optimizer, scaler, loss), so a step that
    # passes here means the real loop's first step will run.
    optimizer = build_optimizer(model, args.lr)
    scaler = build_grad_scaler(device)
    logits = forward_model(model, inputs)
    criterion = build_criterion(args.loss, ce_weight=args.ce_weight, lambda_dice=args.lambda_dice,
                                lambda_ce=args.lambda_ce, lambda_focal=args.lambda_focal,
                                focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha).to(device)
    loss, parts = compute_loss(criterion, logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss in the training step.")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return logits, loss, parts


def make_model(args, channels, device):
    # Build through the registry with the same ModelSpec fields as training,
    # wrapped in DataParallel like the trainer.
    spec = ModelSpec(
        in_channels=channels,
        num_classes=NUM_CLASSES,
        ts_length=args.ts,
        num_heads=args.nh,
        feature_size=args.ed,
        image_size=(args.size, args.size) if hasattr(args, "size") else (CROP_SIZE, CROP_SIZE),
        options=dict(args.model_option or {}),
    )
    return nn.DataParallel(build_model(args.m, spec)).to(device)


def check_model(args):
    """Random input through a registered model: shapes, parameters, one step."""
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    channels = model_input_channels(args.tessera)
    model = make_model(args, channels, device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{args.m}: {parameters:,} parameters; input (B={args.batch}, C={channels}, "
          f"T={args.ts}, H={args.size}, W={args.size}) on {device}")

    # Random inputs (B, C, T, H, W) and random labels with ~2% fire pixels, one-hot
    # as (B, 2, H, W) like real targets: enough to exercise shapes and the loss.
    inputs = torch.randn(args.batch, channels, args.ts, args.size, args.size, device=device)
    fire = torch.rand(args.batch, args.size, args.size, device=device) > 0.98
    labels = torch.stack((~fire, fire), dim=1).float()

    reset_peak_memory(device)
    started = time.time()
    logits, loss, parts = training_step(model, inputs, labels, args, device)
    # The contract every model must meet: after forward_model (which averages a
    # per-day output over T), the logits are (B, 2, H, W).
    expected = (args.batch, NUM_CLASSES, args.size, args.size)
    if tuple(logits.shape) != expected:
        raise SystemExit(f"FAIL: forward_model returned {tuple(logits.shape)}; expected {expected} "
                         "(the model must return (B, K, T, H, W) or (B, K, H, W))")
    print(f"Output {tuple(logits.shape)} OK; one training step {time.time() - started:.2f} s; "
          f"loss {loss.item():.4f} {({k: round(v.item(), 4) for k, v in parts.items()})}")
    report_peak_memory(device)


def datasets(args):
    """Train (augmented) and validation FireDatasets for --data-root."""
    # Same dataset settings as training; TESSERA statistics come from the training
    # fires only, as in training.
    kwargs = dict(ts_length=args.ts, n_channel=N_STORED_CHANNELS, target_is_single_day=True)
    train_maps = None
    if args.tessera:
        tessera_dir = args.data_root / "tessera"
        train_maps = load_tessera_maps(tessera_dir, find_fire_sources(args.data_root / "train"))
        kwargs.update(tessera_dir=tessera_dir, tessera_stats=compute_tessera_stats(train_maps.values()))
    train = FireDataset(npz_dir=args.data_root / "train", use_augmentations=True, tessera_maps=train_maps, **kwargs)
    val = FireDataset(npz_dir=args.data_root / "val", use_augmentations=False, **kwargs)
    return train, val


def check_data(args):
    """One training batch and one validation window: shapes and channel counts."""
    channels = model_input_channels(args.tessera)
    train, val = datasets(args)
    batch = next(iter(make_dataloader(train, args.batch, shuffle=False, num_workers=0, pin_memory=False)))
    sample = val[0]
    # Model input width = time-varying channels + per-fire static channels (TESSERA,
    # when on). Both a training batch and a validation window must match it.
    batch_channels = batch["data"].shape[1] + (batch["static"].shape[1] if "static" in batch else 0)
    sample_channels = sample["data"].shape[0] + (sample["static"].shape[0] if "static" in sample else 0)
    assert batch_channels == channels, f"batch has {batch_channels} channels, expected {channels}"
    assert tuple(batch["data"].shape[2:]) == (args.ts, CROP_SIZE, CROP_SIZE), tuple(batch["data"].shape)
    assert tuple(batch["labels"].shape[1:]) == (2, CROP_SIZE, CROP_SIZE), tuple(batch["labels"].shape)
    assert sample_channels == channels, f"validation window has {sample_channels} channels, expected {channels}"
    assert tuple(sample["labels"].shape) == (2, CROP_SIZE, CROP_SIZE), tuple(sample["labels"].shape)
    print(f"Training windows: {len(train)} | validation windows: {len(val)}")
    print(f"Input: {tuple(batch['data'].shape)} | labels: {tuple(batch['labels'].shape)} | model channels: {channels}")

    # Optional: one real training step on this batch (catches loss or shape
    # problems with real targets before a long run).
    if not args.train_step:
        return
    if batch["data"].shape[0] != args.batch:
        raise RuntimeError("Too few training windows to test the configured full batch.")
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model = make_model(args, channels, device)
    reset_peak_memory(device)
    inputs = batch_inputs(batch, device)
    labels = batch["labels"].float().to(device)
    logits, loss, parts = training_step(model, inputs, labels, args, device)
    assert logits.shape == labels.shape, (tuple(logits.shape), tuple(labels.shape))
    report_peak_memory(device)
    print("One training step passed; loss:", loss.item(), {k: v.item() for k, v in parts.items()})


def check_timing(args):
    """Average seconds to load one augmented training window."""
    train = FireDataset(npz_dir=args.data_root / "train", ts_length=args.ts, n_channel=N_STORED_CHANNELS,
                        target_is_single_day=True, use_augmentations=True)
    count = min(args.windows, len(train))
    started = time.time()
    for index in range(count):
        train[index]
    print(f"{(time.time() - started) / count:.2f} s to load one window")


# Model and loss flags shared by `model` and `data`; same meaning as in cli.train.
def add_model_arguments(parser):
    parser.add_argument("-m", default="swinunetr3d", help=f"Registered model: {', '.join(available_models())}")
    parser.add_argument("-nh", type=int, default=3, help="Attention heads per stage")
    parser.add_argument("-ed", type=int, default=36, help="Base feature size")
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("--loss", choices=LOSSES, default="dicece")
    parser.add_argument("--ce-weight", type=float, nargs=2, default=list(PAPER_CE_WEIGHT))
    parser.add_argument("--lambda-dice", type=float, default=LAMBDA_DICE)
    parser.add_argument("--lambda-ce", type=float, default=LAMBDA_CE)
    parser.add_argument("--lambda-focal", type=float, default=LAMBDA_FOCAL,
                        help="dicefocal only: weight of the focal term (--lambda-dice weighs the Dice term)")
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA,
                        help="dicefocal only: focusing exponent; 0 makes the focal term plain binary cross entropy")
    parser.add_argument("--focal-alpha", type=float, default=FOCAL_ALPHA,
                        help="dicefocal only: weight of fire pixels in the focal term (e.g. 0.25); default off")
    parser.add_argument("--model-option", action="append", type=parse_model_option, default=None, metavar="KEY=VALUE")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("-seed", type=int, default=SEED)


def parse_args(argv=None):
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="python -m firepred.cli.check", description=__doc__.splitlines()[0])
    # Three independent checks: model (no data needed), data, and load timing.
    commands = parser.add_subparsers(dest="command", required=True)

    model = commands.add_parser("model", help="Random input through a model: shapes, parameters, one step")
    add_model_arguments(model)
    model.add_argument("--batch", type=int, default=2)
    model.add_argument("-ts", "--ts", dest="ts", type=int, default=TS_LENGTH)
    model.add_argument("--size", type=int, default=CROP_SIZE, help="H = W of the random input")
    model.add_argument("--tessera", action="store_true", help="Use the 172-channel TESSERA input width")

    data = commands.add_parser("data", help="One batch from --data-root: shapes; optionally one step")
    add_model_arguments(data)
    data.add_argument("--data-root", type=Path, required=True)
    data.add_argument("--batch", type=int, default=8)
    data.add_argument("-ts", "--ts", dest="ts", type=int, default=TS_LENGTH)
    data.add_argument("--tessera", action="store_true")
    data.add_argument("--train-step", action="store_true", help="Also take one training step on the batch")

    timing = commands.add_parser("timing", help="Average time to load one training window")
    timing.add_argument("--data-root", type=Path, required=True)
    timing.add_argument("-ts", "--ts", dest="ts", type=int, default=TS_LENGTH)
    timing.add_argument("--windows", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {"model": check_model, "data": check_data, "timing": check_timing}[args.command](args)


if __name__ == "__main__":
    main()
