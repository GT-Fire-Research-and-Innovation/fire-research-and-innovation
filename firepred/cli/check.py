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

from firepred.config.dataset import CROP_SIZE, N_STORED_CHANNELS, TS_LENGTH
from firepred.config.training import NUM_CLASSES
from firepred.data.fire_dataset import FireDataset, batch_inputs, make_dataloader
from firepred.models.registry import available_models
from firepred.training.accelerator import resolve_device
from firepred.training.config import LossConfig, RunConfig
from firepred.training.flags import add_flags
from firepred.training.experiment import FireData, prepare_model
from firepred.training.trainer import Trainer


def check_config(args):
    """A RunConfig from the check flags (the same settings training would use)."""
    return RunConfig(
        model=args.m,
        learning_rate=args.lr,
        num_heads=args.nh,
        feature_size=args.ed,
        ts_length=args.ts,
        seed=args.seed,
        device=args.device,
        data_root=getattr(args, "data_root", RunConfig.data_root),
        tessera=args.tessera,
        loss=LossConfig(args.loss, tuple(args.ce_weight), args.lambda_dice, args.lambda_ce,
                        args.lambda_focal, args.focal_gamma, args.focal_alpha),
        extra_model_options=tuple(dict(args.model_option or ()).items()),
        batch_size=args.batch,
    )


# Peak GPU memory is reset before, and printed after, the step being measured.
def reset_peak_memory(device):
    """Reset every GPU's peak-memory counter; returns the device."""
    if device.type == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    return device


def report_peak_memory(device):
    """Print and return every GPU's peak allocated memory in GiB ({} on other devices)."""
    peaks = {}
    if device.type == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
            peaks[index] = torch.cuda.max_memory_allocated(index) / 1024**3
            print(f"GPU {index} peak allocated memory: {peaks[index]:.2f} GiB")
    return peaks


def training_step(config, model, inputs, labels, device):
    """One forward/backward/Adam step with the trainer's own pieces; returns (logits, loss, parts).

    The same Trainer the real run uses, so a step that passes here means the
    real loop's first step will run.
    """
    logits, loss, parts = Trainer(config, model, device).step(inputs, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss in the training step.")
    return logits, loss, parts


def check_model(args):
    """Random input through a registered model: shapes, parameters, one step."""
    config = check_config(args)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    channels = config.input_channels
    model = prepare_model(config.model, config.model_spec(image_size=(args.size, args.size)), device)
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
    logits, loss, parts = training_step(config, model, inputs, labels, device)
    # The contract every model must meet: after forward_model (which averages a
    # per-day output over T), the logits are (B, 2, H, W).
    expected = (args.batch, NUM_CLASSES, args.size, args.size)
    if tuple(logits.shape) != expected:
        raise SystemExit(f"FAIL: forward_model returned {tuple(logits.shape)}; expected {expected} "
                         "(the model must return (B, K, T, H, W) or (B, K, H, W))")
    print(f"Output {tuple(logits.shape)} OK; one training step {time.time() - started:.2f} s; "
          f"loss {loss.item():.4f} {({k: round(v.item(), 4) for k, v in parts.items()})}")
    report_peak_memory(device)
    return logits


def datasets(config):
    """Train (augmented) and validation FireDatasets, built exactly as training builds them."""
    data = FireData(config, config.data_root, config.tessera_root)
    stats, maps = None, None
    if config.tessera:
        from firepred.data.archive import find_fire_sources
        from firepred.data.tessera_archive import compute_tessera_stats, load_tessera_maps

        maps = load_tessera_maps(config.tessera_root, find_fire_sources(config.data_root / "train"))
        stats = compute_tessera_stats(maps.values(), min_coverage=config.tessera_min_coverage)
    return (data.dataset("train", augment=True, tessera_stats=stats, tessera_maps=maps),
            data.dataset("val", augment=False, tessera_stats=stats))


def check_data(args):
    """One training batch and one validation window: shapes and channel counts."""
    config = check_config(args)
    channels = config.input_channels
    train, val = datasets(config)
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
        return batch
    if batch["data"].shape[0] != args.batch:
        raise RuntimeError("Too few training windows to test the configured full batch.")
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    model = prepare_model(config.model, config.model_spec(), device)
    reset_peak_memory(device)
    inputs = batch_inputs(batch, device)
    labels = batch["labels"].float().to(device)
    logits, loss, parts = training_step(config, model, inputs, labels, device)
    assert logits.shape == labels.shape, (tuple(logits.shape), tuple(labels.shape))
    report_peak_memory(device)
    print("One training step passed; loss:", loss.item(), {k: v.item() for k, v in parts.items()})
    return batch


def check_timing(args):
    """Average seconds to load one augmented training window (printed and returned)."""
    train = FireDataset(npz_dir=args.data_root / "train", ts_length=args.ts, n_channel=N_STORED_CHANNELS,
                        target_is_single_day=True, use_augmentations=True)
    count = min(args.windows, len(train))
    started = time.time()
    for index in range(count):
        train[index]
    seconds = (time.time() - started) / count
    print(f"{seconds:.2f} s to load one window")
    return seconds


# Model and loss flags shared by `model` and `data`: the training flags themselves
# (training/flags.py), with defaults where training requires a value.
CHECK_FLAGS = ("model", "num_heads", "feature_size", "learning_rate", "loss.name", "loss.ce_weight",
               "loss.lambda_dice", "loss.lambda_ce", "loss.lambda_focal", "loss.focal_gamma", "loss.focal_alpha",
               "extra_model_options", "device", "seed")


def add_model_arguments(parser):
    """Add the training flags `model` and `data` share (same names and help as cli.train); return the parser."""
    overrides = {
        "model": {"required": False, "default": "swinunetr3d",
                  "help": f"Registered model: {', '.join(available_models())}"},
        "num_heads": {"required": False},
        "feature_size": {"required": False},
        "learning_rate": {"required": False},
        "loss.name": {"default": LossConfig.name},
    }
    return add_flags(parser, CHECK_FLAGS, overrides)


def parse_args(argv=None):
    """Parse the check command line (model, data or timing)."""
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
    """Run the chosen check and return its result."""
    args = parse_args(argv)
    return {"model": check_model, "data": check_data, "timing": check_timing}[args.command](args)


if __name__ == "__main__":
    main()
