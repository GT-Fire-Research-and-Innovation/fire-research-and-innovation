"""Quick checks before a long run: a model, the data, one training step, load time.

Level 4 (cli).

    python -m firepred.cli.check model -m swinunetr3d [-nh 3 -ed 36 --size 256 --batch 2]
        Build the model, run random input through it, check the output shape
        contract, take one Adam step; print parameters, time and GPU memory.
    python -m firepred.cli.check data --data-root <root> [--tessera] [--train-step -m swinunetr3d ...]
        Read <root>/window_config.json as training does, then one training
        batch and one validation window, and check their shapes; with
        --train-step also take one real training step.
    python -m firepred.cli.check timing --data-root <root> [--windows 10]
        Average time to load one training window.
    python -m firepred.cli.check labels --data-root <root> [--split train]
        Exact background and fire pixel counts of a split, counted from the
        labels now, and the matching --ce-weight (`dataset count-pixels` records
        the same counts in window_config.json for --ce-weight auto).

Every check is disposable: nothing is saved, nothing is logged to W&B.
"""

import argparse
import time
from pathlib import Path

import torch

from firepred.config.dataset import BASE_MODEL_CHANNELS, CROP_SIZE, N_STORED_CHANNELS, TESSERA_CHANNELS, TS_LENGTH
from firepred.config.training import NUM_CLASSES
from firepred.data.fire_dataset import FireDataset, batch_inputs, make_dataloader
from firepred.data.stats import WindowConfig, count_labels
from firepred.models.registry import available_models
from firepred.training.accelerator import Device
from firepred.training.config import LossConfig, RunConfig
from firepred.training.flags import add_flags, ce_weight_value
from firepred.training.experiment import Experiment, FireData, with_window_label_counts, with_window_normalization
from firepred.training.trainer import Trainer


def check_config(args):
    """Build a RunConfig from the check flags, as training would.

    The checks use the same settings objects as a real run, so a check that
    passes means training will start. Settings the check flags lack keep their
    defaults. --ce-weight auto takes the pixel counts in the data root's
    window_config.json, as training does.

    Args:
        args (argparse.Namespace): the parsed check flags.

    Returns:
        RunConfig: the settings.
    """
    config = RunConfig(
        model=args.m,
        learning_rate=args.lr,
        num_heads=args.nh,
        feature_size=args.ed,
        ts_length=args.ts,
        seed=args.seed,
        device=args.device,
        data_root=getattr(args, "data_root", RunConfig.data_root),
        tessera=args.tessera,
        tessera_cells_per_pixel=args.tessera_cells_per_pixel,
        normalization=args.normalization,
        loss=LossConfig(args.loss, ce_weight_value(args.ce_weight), args.lambda_dice, args.lambda_ce,
                        args.lambda_focal, args.focal_gamma, args.focal_alpha),
        extra_model_options=tuple(dict(args.model_option or ()).items()),
        batch_size=args.batch,
    )
    if config.needs_label_counts:
        config = with_window_label_counts(config, WindowConfig.read(config.data_root),
                                          Path(config.data_root) / "train")
    return config


# Peak GPU memory is reset before, and printed after, the step being measured.
def reset_peak_memory(device):
    """Reset every GPU's peak-memory counter before a measured step.

    The peak after the step then shows that step's memory use. Other devices
    have no counter.

    Args:
        device (torch.device): the device.

    Returns:
        torch.device: device.
    """
    if device.type == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    return device


def report_peak_memory(device):
    """Print every GPU's peak memory since the last reset.

    This tells whether a batch size fits before a long run. Other devices
    report nothing.

    Args:
        device (torch.device): the device.

    Returns:
        dict[int, float]: {GPU index: peak GiB}.
    """
    peaks = {}
    if device.type == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
            peaks[index] = torch.cuda.max_memory_allocated(index) / 1024**3
            print(f"GPU {index} peak allocated memory: {peaks[index]:.2f} GiB")
    return peaks


def training_step(config, model, inputs, labels, device):
    """Take one training step with the trainer's own loss, optimizer and scaler.

    Using the real Trainer means a step that passes here will also pass in
    training. A non-finite loss fails the check.

    Args:
        config (RunConfig): the settings.
        model (torch.nn.Module): the model.
        inputs (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): input of shape
            (B, C, T, H, W), or (data, static) for a model with a TESSERA stem.
        labels (torch.Tensor): one-hot targets of shape (B, 2, H, W).
        device (torch.device): where it runs.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict]: (logits, loss, loss parts).

    Raises:
        RuntimeError: if the loss is not finite.
    """
    logits, loss, parts = Trainer(config, model, device).step(inputs, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss in the training step.")
    return logits, loss, parts


def check_model(args):
    """Check a model with random input: output shape, parameters, one step, time and memory.

    No data is needed. Random labels with about 2% fire pixels exercise the
    loss. The output must be (B, 2, H, W) after the day average.

    Args:
        args (argparse.Namespace): the parsed `model` flags.

    Returns:
        torch.Tensor: the logits.
    """
    config = check_config(args)
    torch.manual_seed(config.seed)
    device = Device.resolve(config.device).torch_device
    channels = config.input_channels
    model = Experiment(config).prepare_model(device, config.model_spec(image_size=(args.size, args.size)))
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{args.m}: {parameters:,} parameters; input (B={args.batch}, C={channels}, "
          f"T={args.ts}, H={args.size}, W={args.size}) on {device}")

    # Random inputs (B, C, T, H, W) and random labels with ~2% fire pixels, one-hot
    # as (B, 2, H, W) like real targets: enough to exercise shapes and the loss.
    cells = config.tessera_cells
    if cells > 1:
        # A finer TESSERA map goes beside the 43 channels, to the model's stem.
        side = args.size * cells
        inputs = (torch.randn(args.batch, BASE_MODEL_CHANNELS, args.ts, args.size, args.size, device=device),
                  torch.randn(args.batch, TESSERA_CHANNELS, side, side, device=device).half())
    else:
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
    """Build the training and validation datasets exactly as a run does.

    TESSERA statistics come from the training fires only. Training windows are
    augmented, validation windows are not.

    Args:
        config (RunConfig): the settings.

    Returns:
        tuple[FireDataset, FireDataset]: (train, val).
    """
    data = FireData(config, config.data_root, config.tessera_root)
    stats, maps = data.training_tessera()
    return (data.dataset("train", augment=True, tessera_stats=stats, tessera_maps=maps),
            data.dataset("val", augment=False, tessera_stats=stats))


def check_data(args):
    """Check one training batch and one validation window from a data root.

    The data root's window_config.json comes first, as in training: it must
    exist, match -ts, and (for --normalization window-config) hold the training
    statistics. Channel counts must match the model's input width, and shapes
    the window size. With --train-step one real training step runs on the batch.

    Args:
        args (argparse.Namespace): the parsed `data` flags.

    Returns:
        dict: the training batch.

    Raises:
        AssertionError: on a wrong shape or channel count.
        RuntimeError: if there are too few windows for one full batch.
    """
    config = check_config(args)
    window_config = WindowConfig.read(config.data_root).check_windows(config.ts_length)
    config = with_window_normalization(config, window_config, Path(config.data_root) / "train")
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
    device = Device.resolve(config.device).torch_device
    model = Experiment(config).prepare_model(device)
    reset_peak_memory(device)
    inputs = batch_inputs(batch, device)
    labels = batch["labels"].float().to(device)
    logits, loss, parts = training_step(config, model, inputs, labels, device)
    assert logits.shape == labels.shape, (tuple(logits.shape), tuple(labels.shape))
    report_peak_memory(device)
    print("One training step passed; loss:", loss.item(), {k: v.item() for k, v in parts.items()})
    return batch


def check_timing(args):
    """Time how long one augmented training window takes to load.

    Loading is rarely the bottleneck, but this shows it. The average over a few
    windows is printed.

    Args:
        args (argparse.Namespace): the parsed `timing` flags.

    Returns:
        float: seconds per window.
    """
    train = FireDataset(split_dir=args.data_root / "train", ts_length=args.ts, stored_channels=N_STORED_CHANNELS,
                        target_is_single_day=True, augment=True)
    count = min(args.windows, len(train))
    started = time.time()
    for index in range(count):
        train[index]
    seconds = (time.time() - started) / count
    print(f"{seconds:.2f} s to load one window")
    return seconds


def check_labels(args):
    """Count a split's background and fire pixels and print the matching --ce-weight.

    The counts are exact and always the same for the same data
    (data/stats.py). The printed weight is what --ce-weight auto trains
    with when the split is train.

    Args:
        args (argparse.Namespace): the parsed `labels` flags.

    Returns:
        LabelCounts: the split's counts.
    """
    started = time.time()
    counts = count_labels(args.data_root / args.split, workers=args.workers)
    print(counts.summary())
    if counts.fire:
        print(f"--ce-weight 1.0 {counts.background_per_fire!r}  (the paper's: 1.0 446.7836)")
    print(f"Counted in {time.time() - started:.1f} s")
    return counts


# Model and loss flags shared by `model` and `data`: the training flags themselves
# (training/flags.py), with defaults where training requires a value.
CHECK_FLAGS = ("model", "num_heads", "feature_size", "learning_rate", "loss.name", "loss.ce_weight",
               "loss.lambda_dice", "loss.lambda_ce", "loss.lambda_focal", "loss.focal_gamma", "loss.focal_alpha",
               "extra_model_options", "device", "seed", "tessera_cells_per_pixel", "normalization")


def add_model_arguments(parser):
    """Add the training flags the `model` and `data` checks share.

    They come from the training flag table, so names and help match cli.train.
    Flags training requires get defaults here.

    Args:
        parser (argparse.ArgumentParser): a subcommand's parser.

    Returns:
        argparse.ArgumentParser: parser.
    """
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
    """Parse the check command line: model, data or timing.

    Each check is a subcommand with its own flags. The model and data checks
    share the training flags.

    Args:
        argv (list[str] | None): the arguments (default: sys.argv).

    Returns:
        argparse.Namespace: the parsed flags.
    """
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

    labels = commands.add_parser("labels", help="Background and fire pixel counts of a split, and --ce-weight auto")
    labels.add_argument("--data-root", type=Path, required=True)
    labels.add_argument("--split", default="train", choices=("train", "val", "test"))
    labels.add_argument("--workers", type=int, default=None, help="Reading threads (default: up to 8)")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the chosen check.

    Checks save nothing and log nothing to W&B. They are safe to run at any time.

    Args:
        argv (list[str] | None): the arguments (default: sys.argv).

    Returns:
        object: the check's result.
    """
    args = parse_args(argv)
    checks = {"model": check_model, "data": check_data, "timing": check_timing, "labels": check_labels}
    return checks[args.command](args)


if __name__ == "__main__":
    main()
