"""One experiment from start to finish: run(args) with the parsed command line.

Level 3 (training). Called by cli/train.py.

The order of steps is part of the result: random numbers are drawn by model
construction, the DataLoader shuffle and augmentation, so moving a step
changes every number after it. The order is:

    1. seed numpy and torch
    2. W&B run (name, config, chart definitions)
    3. model (registry), wrapped in DataParallel
    4. loss, metrics, Adam, gradient scaler
    5. training: TESSERA statistics, datasets, shape check, loaders, epochs
    6. testing: load the best (or given) checkpoint, score the test fires
"""

from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn

from firepred.config.channels import N_STORED_CHANNELS, model_input_channels as count_model_channels
from firepred.config.paths import DEFAULT_DATA_ROOT
from firepred.config.training import NUM_CLASSES, TOP_N_CHECKPOINTS
from firepred.data.archive import find_fire_sources, fire_name, unzip_data_root
from firepred.data.dataset import FireDataset
from firepred.data.loader import make_dataloader, validate_dataset_output
from firepred.data.tessera import compute_tessera_stats
from firepred.models.registry import build_model
from firepred.models.spec import ModelSpec
from firepred.training import naming
from firepred.training.accelerator import checkpoint_map_location, is_xla, resolve_device, seed_device
from firepred.training.checkpoints import load_checkpoint
from firepred.training.evaluation import evaluate_model
from firepred.training.loop import train_model
from firepred.training.losses import build_criterion
from firepred.training.metrics import build_post_transform, build_validation_metrics
from firepred.training.optim import build_grad_scaler, build_lr_scheduler, build_optimizer
from firepred.training.tracking import configure_wandb, log_final_test


def default_loss(model_name):
    """Paper: Dice+CE for U-Net-3D, UNETR-3D and SwinUNETR-3D; the
    Attention-U-Net-3D "kept the original setup of using Dice loss". [paper]"""
    return "dice" if model_name == "attunet" else "dicece"


def model_options(args):
    """Model-specific settings for ModelSpec.options: the dedicated flags,
    then any --model-option KEY=VALUE pairs."""
    options = {
        "unetr_version": args.unetr_version,
        "swin_temporal_downsample": args.swin_temporal_downsample,
    }
    options.update(args.model_option or {})
    return options


def run(args):
    # Fix the random seeds first: everything random after this (initial weights,
    # shuffling, augmentation) is then reproducible.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Unpack the command line into named settings.
    model_name = args.m
    mode = args.mode
    if args.loss is None:
        args.loss = default_loss(model_name)
    batch_size = args.b
    num_heads = args.nh
    hidden_size = args.ed
    ts_length = args.ts
    learning_rate = args.lr
    # Logged to W&B only; Adam never receives it (docs/KNOWN_ISSUES.md).
    weight_decay = learning_rate / 10
    max_epochs = args.epochs
    top_n_checkpoints = TOP_N_CHECKPOINTS
    num_classes = NUM_CLASSES

    # The archives store 27 channels; FireDataset turns land cover into 17
    # one-hot channels (43). TESSERA mode appends 128 bands + 1 mask (172).
    stored_n_channels = N_STORED_CHANNELS
    use_tessera = args.tessera
    model_input_channels = count_model_channels(use_tessera)
    # -nc is optional; when given it must agree with what the data will produce.
    if args.nc is not None and args.nc != model_input_channels:
        raise ValueError(
            f"-nc {args.nc} does not match the preprocessing: expected "
            f"{model_input_channels} {'with' if use_tessera else 'without'} "
            "--tessera"
        )

    # ----- Where the data is ----- #

    # Locate train/, val/, test/ (and tessera/). With --unzip-dir, the archives are
    # unpacked once and read from the unpacked copy (same arrays, faster reads).
    processed_data_root = (
        Path(args.data_root).expanduser()
        if args.data_root is not None
        else DEFAULT_DATA_ROOT
    )
    tessera_dir = None
    if use_tessera:
        tessera_dir = (
            Path(args.tessera_dir).expanduser()
            if args.tessera_dir is not None
            else processed_data_root / "tessera"
        )
    # Unzipping copies only the window archives; TESSERA stays where it is.
    if args.unzip_dir != "off":
        resolved_root = processed_data_root.resolve()
        unzip_root = (
            resolved_root.with_name(resolved_root.name + "-npy")
            if args.unzip_dir == "auto"
            else Path(args.unzip_dir).expanduser()
        )
        processed_data_root = unzip_data_root(
            processed_data_root,
            unzip_root,
            splits=("test",) if args.run_test else ("train", "val", "test"),
        )
    train_npz_dir = processed_data_root / "train"
    val_npz_dir = processed_data_root / "val"
    test_npz_dir = processed_data_root / "test"

    # ----- Names and recorded settings ----- #

    loss_config = naming.loss_config(
        args.loss, args.ce_weight, args.lambda_dice, args.lambda_ce,
        args.lambda_focal, args.focal_gamma, args.focal_alpha,
    )
    lr_config = naming.lr_config(args.lr_schedule, args.lr_min, args.lr_decay_factor, args.lr_patience)
    lr_tag = naming.lr_tag(args.lr_schedule)
    run_tag = naming.run_tag(use_tessera, args.loss, args.lr_schedule)
    # Names and recorded settings: the loss and LR-schedule configs go into W&B and
    # into every checkpoint; the run tag goes into checkpoint file names.
    options = model_options(args)

    # Start the W&B run before building the model (keeps the original order of
    # random-number use, so results match earlier runs).
    configure_wandb(
        model_name=model_name,
        mode=mode,
        num_heads=num_heads,
        hidden_size=hidden_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        wandb_user_name=args.wandb_user,
        run_test=args.run_test,
        use_tessera=use_tessera,
        model_input_channels=model_input_channels,
        loss_config=loss_config,
    )

    # --device auto (default): the GPU when visible, else the CPU. --device tpu: one
    # TPU device through PyTorch/XLA, with its own random generator seeded here.
    device = resolve_device(args.device)
    seed_device(device, args.seed)
    if is_xla(device):
        wandb.config.update({"device": "tpu"})
    # No pinned host copies: with TESSERA a batch of 8 is about 0.8 GiB, and
    # pinning keeps a second copy of each in host RAM.
    pin_memory = False

    # ----- Model ----- #

    # Build the requested model from the registry, record model settings in W&B,
    # and wrap it in DataParallel (splits each batch across available GPUs).
    model = build_model(
        model_name,
        ModelSpec(
            in_channels=model_input_channels,
            num_classes=num_classes,
            ts_length=ts_length,
            num_heads=num_heads,
            feature_size=hidden_size,
            options=options,
        ),
    )
    if model_name == "swinunetr3d":
        wandb.config.update({"swin_temporal_downsample": args.swin_temporal_downsample})
    if args.model_option:
        wandb.config.update({"model_options": dict(args.model_option)})
    wandb.config.update(lr_config)
    # [ours] periodic test scoring during training (logging only).
    wandb.config.update({"test_every": args.test_every})
    if lr_tag:
        wandb.run.name += lr_tag
    model = nn.DataParallel(model)
    model.to(device)

    # ----- Loss, metrics, optimizer ----- #

    # Loss, metrics, optimizer and gradient scaler.
    criterion = build_criterion(
        args.loss,
        ce_weight=args.ce_weight,
        lambda_dice=args.lambda_dice,
        lambda_ce=args.lambda_ce,
        lambda_focal=args.lambda_focal,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    ).to(device)
    mean_iou, dice_metric = build_validation_metrics()
    post_trans = build_post_transform()
    optimizer = build_optimizer(model, learning_rate)
    scaler = build_grad_scaler(device)

    # Helpers shared by the periodic test check and the final test.
    def make_test_dataset(tessera_stats):
        return FireDataset(
            npz_dir=test_npz_dir,
            ts_length=ts_length,
            n_channel=stored_n_channels,
            target_is_single_day=True,
            use_augmentations=False,
            tessera_dir=tessera_dir,
            tessera_stats=tessera_stats,
        )

    def score(test_dataset, save_plots=True, verbose=True):
        return evaluate_model(
            model=model,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            post_trans=post_trans,
            device=device,
            test_csv_path=args.test_csv,
            plot_dir=args.plot_dir,
            num_heads=num_heads,
            hidden_size=hidden_size,
            model_input_channels=model_input_channels,
            save_plots=save_plots,
            verbose=verbose,
        )

    checkpoint_template = naming.checkpoint_template(
        model_name, mode, num_heads, hidden_size, batch_size, model_input_channels, ts_length, run_tag
    )

    # ----- Training ----- #

    if not args.run_test:
        # TESSERA normalization comes from the training fires only (no leak from val/test).
        tessera_stats = None
        if use_tessera:
            # Normalization comes from the training fires only.
            tessera_stats = compute_tessera_stats(
                [
                    tessera_dir / f"{fire_name(path)}.npz"
                    for path in find_fire_sources(train_npz_dir)
                ],
                min_coverage=args.tessera_min_coverage,
            )
        # Training windows are augmented; validation windows are not.
        train_dataset = FireDataset(
            npz_dir=train_npz_dir,
            ts_length=ts_length,
            n_channel=stored_n_channels,
            target_is_single_day=True,
            use_augmentations=True,
            tessera_dir=tessera_dir,
            tessera_stats=tessera_stats,
        )
        val_dataset = FireDataset(
            npz_dir=val_npz_dir,
            ts_length=ts_length,
            n_channel=stored_n_channels,
            target_is_single_day=True,
            use_augmentations=False,
            tessera_dir=tessera_dir,
            tessera_stats=tessera_stats,
        )

        # Use validation for the shape check so random augmentation is avoided.
        # Fail fast if the data's channel count does not match the model.
        validate_dataset_output(
            val_dataset,
            model_input_channels,
            "validation",
        )

        # Batches: shuffled for training, fixed order for validation.
        train_dataloader = make_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_dataloader = make_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        # [ours] optional test scoring every --test-every epochs with the
        # current weights, for logging only.
        score_test_fires = None
        if args.test_every > 0:
            if find_fire_sources(test_npz_dir):
                score_test_fires = partial(
                    score, make_test_dataset(tessera_stats), save_plots=False, verbose=False
                )
            else:
                print(f"No test archives in {test_npz_dir}; --test-every is ignored")

        # Train for --epochs, keeping the lowest-validation-loss checkpoint (with the
        # settings needed to reload it) and optionally uploading it to W&B.
        best_checkpoint = train_model(
            model=model,
            train_dataset=train_dataset,
            train_dataloader=train_dataloader,
            val_dataset=val_dataset,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            mean_iou=mean_iou,
            dice_metric=dice_metric,
            post_trans=post_trans,
            device=device,
            max_epochs=max_epochs,
            top_n_checkpoints=top_n_checkpoints,
            checkpoint_name=checkpoint_template,
            checkpoint_extra={
                "use_tessera": use_tessera,
                "model_input_channels": model_input_channels,
                "swin_temporal_downsample": args.swin_temporal_downsample,
                "tessera_stats": tessera_stats,
                "loss_config": loss_config,
                "lr_config": lr_config,
                **({"model_options": dict(args.model_option)} if args.model_option else {}),
            },
            artifact_name=(
                naming.artifact_name(model_name, run_tag, wandb.run.id)
                if args.upload_checkpoints
                else None
            ),
            lr_scheduler=build_lr_scheduler(
                optimizer,
                args.lr_schedule,
                max_epochs,
                args.lr_decay_factor,
                args.lr_patience,
                args.lr_min,
            ),
            test_every=args.test_every,
            score_test_fires=score_test_fires,
        )
        # Then test that best checkpoint on the test fires, unless asked not to.
        if not args.test_after_train:
            return
        if not find_fire_sources(test_npz_dir):
            print(f"No test archives in {test_npz_dir}; skipping the test step")
            return
        # The paper's metrics on the test fires, from the best-validation checkpoint.
        checkpoint_path = best_checkpoint
    else:
        # -test mode: skip training and test a given (or default-named) checkpoint.
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = naming.default_test_checkpoint(checkpoint_template, args.load_epoch)

    # ----- Testing ----- #

    # Load the checkpoint safely, refusing one trained with different inputs.
    checkpoint = load_checkpoint(
        checkpoint_path,
        checkpoint_map_location(device),
        model_name,
        use_tessera,
        args.swin_temporal_downsample,
        # The checkpoint this run just saved is trusted; a given one only on request.
        trusted=args.trust_checkpoint or not args.run_test,
    )
    test_dataset = make_test_dataset(checkpoint.get("tessera_stats"))
    validate_dataset_output(
        test_dataset,
        model_input_channels,
        "test",
    )

    # Restore the weights and score the test fires the paper's way (per fire, then
    # averaged), logging the final numbers to W&B.
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        f"Loaded checkpoint epoch={checkpoint['epoch']} "
        f"loss={checkpoint['loss']}"
    )

    log_final_test(score(test_dataset))
