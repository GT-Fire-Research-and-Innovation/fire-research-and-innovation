"""The epoch loop: train, validate, keep the best checkpoint, log to W&B.

Level 3 (training). Used by training/run.py.

One epoch is:
    train_one_epoch   forward, loss, backward, Adam step for every batch
    validate          loss and scores on the validation windows, no gradients
    log + checkpoint  W&B charts; save if the validation loss is among the best
    lr schedule       optional decay [ours]
    periodic test     optional test scores with the current weights [ours]
"""

import os
import random

import numpy as np
import torch
import wandb
from monai.data import decollate_batch
from tqdm import tqdm

from firepred.data.loader import batch_inputs
from firepred.models.registry import forward_model
from firepred.training.accelerator import mark_step
from firepred.training.checkpoints import BestCheckpoints, save_checkpoint
from firepred.training.losses import compute_loss, loss_name_of
from firepred.training.metrics import background_f1, fire_only_f1, fire_only_iou
from firepred.training.optim import step_lr_scheduler
from firepred.training.tracking import build_wandb_metric_labels


def train_one_epoch(model, train_dataset, train_dataloader, optimizer, criterion, scaler, post_trans, device,
                    epoch, max_epochs):
    """One pass over the training windows. Returns the epoch's mean scores."""
    # Training mode (BatchNorm updates its statistics) and running sums for the epoch averages.
    model.train()
    train_loss = 0.0
    train_components = {}
    train_fire_f1_sum = 0.0
    train_background_f1_sum = 0.0
    train_fire_iou_sum = 0.0
    train_bar = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        desc=f"Train epoch {epoch + 1}/{max_epochs}",
    )

    for batch_idx, batch in enumerate(train_bar):
        data_batch = batch_inputs(batch, device)
        labels_batch = batch["labels"].float().to(
            device,
            non_blocking=True,
        )

        # Forward pass: per-day logits averaged over the days -> (B, 2, H, W), then the loss.
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_model(model, data_batch)
        loss, components = compute_loss(criterion, outputs, labels_batch)
        mark_step(device)  # TPU only: run the forward graph once, before the scores read it.

        with torch.no_grad():
            predictions = torch.stack(
                # Training scores, for logging only: threshold the predictions and add up
                # per-window F1/IoU (no gradients, so they don't affect learning).
                [post_trans(item) for item in decollate_batch(outputs.detach())]
            )
            train_fire_f1_sum += fire_only_f1(predictions, labels_batch).sum().item()
            train_background_f1_sum += background_f1(predictions, labels_batch).sum().item()
            train_fire_iou_sum += fire_only_iou(predictions, labels_batch).sum().item()

        # Backward pass and one Adam step (the scaler only matters for mixed precision on GPU).
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        mark_step(device)  # TPU only: run the backward pass and the Adam update.

        # Accumulate the loss weighted by batch size, so the epoch value is a per-window mean.
        train_loss += loss.detach().item() * data_batch.size(0)
        for name, value in components.items():
            train_components[name] = (
                train_components.get(name, 0.0)
                + value.item() * data_batch.size(0)
            )
        running_loss = train_loss / (
            (batch_idx + 1) * data_batch.size(0)
        )
        train_bar.set_postfix(loss=f"{running_loss:.4f}")

        # Stop immediately if the loss blows up instead of training on garbage.
        if not np.isfinite(train_loss):
            raise RuntimeError(
                f"Training loss became non-finite at step {batch_idx}"
            )

    # Turn the running sums into per-window averages for this epoch.
    train_loss /= len(train_dataset)
    train_components = {
        f"train_{name}": total / len(train_dataset)
        for name, total in train_components.items()
    }
    return {
        "loss": train_loss,
        "components": train_components,
        "fire_f1": train_fire_f1_sum / len(train_dataset),
        "background_f1": train_background_f1_sum / len(train_dataset),
        "fire_iou": train_fire_iou_sum / len(train_dataset),
    }


def validate(model, val_dataset, val_dataloader, criterion, mean_iou, dice_metric, post_trans, device,
             epoch, max_epochs):
    """Loss and scores on the validation windows (eval mode, no gradients)."""
    # Evaluation mode: BatchNorm uses its stored statistics; no gradients below.
    model.eval()
    val_loss = 0.0
    val_components = {}
    iou_values = []
    dice_values = []
    val_fire_f1_sum = 0.0
    val_background_f1_sum = 0.0
    val_fire_iou_sum = 0.0
    val_bar = tqdm(
        val_dataloader,
        total=len(val_dataloader),
        desc=f"Validation epoch {epoch + 1}/{max_epochs}",
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_bar):
            data_batch = batch_inputs(batch, device)
            labels_batch = batch["labels"].float().to(
                device,
                non_blocking=True,
            )

            outputs = forward_model(model, data_batch)
            loss, components = compute_loss(criterion, outputs, labels_batch)
            mark_step(device)  # TPU only: run the forward graph once for loss and scores.
            # Same loss as training, on unaugmented validation windows.
            val_loss += loss.detach().item() * data_batch.size(0)
            for name, value in components.items():
                val_components[name] = (
                    val_components.get(name, 0.0)
                    + value.item() * data_batch.size(0)
                )

            # Threshold the predictions and score them: fire/background F1 and fire IoU per
            # window, plus MONAI's mean IoU / Dice over both classes.
            discrete_outputs = [
                post_trans(item)
                for item in decollate_batch(outputs)
            ]
            discrete_labels = decollate_batch(labels_batch)
            predictions = torch.stack(discrete_outputs)
            val_fire_f1_sum += fire_only_f1(predictions, labels_batch).sum().item()
            val_background_f1_sum += background_f1(predictions, labels_batch).sum().item()
            val_fire_iou_sum += fire_only_iou(predictions, labels_batch).sum().item()

            iou_values.append(
                mean_iou(
                    discrete_outputs,
                    discrete_labels,
                ).mean().item()
            )
            dice_values.append(
                dice_metric(
                    y_pred=discrete_outputs,
                    y=discrete_labels,
                ).mean().item()
            )

            running_val_loss = val_loss / (
                (batch_idx + 1) * data_batch.size(0)
            )
            val_bar.set_postfix(loss=f"{running_val_loss:.4f}")

    # Per-window averages for the epoch.
    val_loss /= len(val_dataset)
    val_components = {
        f"val_{name}": total / len(val_dataset)
        for name, total in val_components.items()
    }
    return {
        "loss": val_loss,
        "components": val_components,
        # Means of per-batch means (docs/KNOWN_ISSUES.md).
        "miou": float(np.mean(iou_values)),
        "mdice": float(np.mean(dice_values)),
        "fire_f1": val_fire_f1_sum / len(val_dataset),
        "background_f1": val_background_f1_sum / len(val_dataset),
        "fire_iou": val_fire_iou_sum / len(val_dataset),
    }


def score_test_fires_untouched(score_test_fires):
    """Run `score_test_fires()` without disturbing any random-number stream.

    Building test loaders draws random numbers; restoring every random state
    keeps the training run identical with or without these checks. [ours]
    """
    # Remember every random-number generator, score the test fires, then restore
    # them, so training continues exactly as if this check had not run.
    rng_states = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    test_scores = score_test_fires()
    random.setstate(rng_states[0])
    np.random.set_state(rng_states[1])
    torch.set_rng_state(rng_states[2])
    if rng_states[3] is not None:
        torch.cuda.set_rng_state_all(rng_states[3])
    return test_scores


def train_model(
    model,
    train_dataset,
    train_dataloader,
    val_dataset,
    val_dataloader,
    optimizer,
    criterion,
    scaler,
    mean_iou,
    dice_metric,
    post_trans,
    device,
    max_epochs,
    top_n_checkpoints,
    checkpoint_name,
    checkpoint_extra=None,
    artifact_name=None,
    lr_scheduler=None,
    test_every=0,
    score_test_fires=None,
):
    """Train and validate; return the checkpoint with the lowest validation loss.

    `checkpoint_name` holds a literal "{epoch}" (1-based). `lr_scheduler` is
    the optional decay [ours]; None keeps the paper's constant rate.

    [ours] With `test_every` > 0, `score_test_fires()` scores the test fires
    with the current weights every `test_every` epochs, for logging only. It
    runs in eval mode without gradients, so no weight or batch-norm statistic
    changes, and it never affects checkpoint choice.
    """
    metric_labels = build_wandb_metric_labels(loss_name_of(criterion))
    # One epoch = train, log, validate, log, keep the best checkpoint, adjust the
    # learning rate, optionally peek at the test fires.
    best_checkpoints = BestCheckpoints(top_n_checkpoints)

    for epoch in range(max_epochs):
        # 1. One pass over the training windows, then chart its averages.
        train = train_one_epoch(
            model, train_dataset, train_dataloader, optimizer, criterion, scaler, post_trans, device,
            epoch, max_epochs,
        )
        wandb.log(
            {
                metric_labels["epoch"]: epoch,
                metric_labels["train_loss"]: train["loss"],
                metric_labels["train_fire_f1"]: train["fire_f1"],
                metric_labels["train_background_f1"]: train["background_f1"],
                metric_labels["train_fire_iou"]: train["fire_iou"],
                **{metric_labels[name]: value for name, value in train["components"].items()},
            }
        )
        print(
            f"Epoch {epoch + 1}, Train Loss: {train['loss']:.4f}"
            + "".join(f", {k}: {v:.4f}" for k, v in train["components"].items())
        )

        # 2. Score the validation windows with the same weights, then chart them.
        val = validate(
            model, val_dataset, val_dataloader, criterion, mean_iou, dice_metric, post_trans, device,
            epoch, max_epochs,
        )
        wandb.log(
            {
                metric_labels["epoch"]: epoch,
                metric_labels["val_loss"]: val["loss"],
                metric_labels["miou"]: val["miou"],
                metric_labels["mdice"]: val["mdice"],
                metric_labels["val_fire_f1"]: val["fire_f1"],
                metric_labels["val_background_f1"]: val["background_f1"],
                metric_labels["val_fire_iou"]: val["fire_iou"],
                metric_labels["learning_rate"]: optimizer.param_groups[0]["lr"],
                **{metric_labels[name]: value for name, value in val["components"].items()},
            }
        )
        print(
            f"Epoch {epoch + 1}, Validation Loss: {val['loss']:.4f}, "
            + "".join(f"{k}: {v:.4f}, " for k, v in val["components"].items())
            + f"Mean IoU: {val['miou']:.4f}, "
            f"Mean Dice: {val['mdice']:.4f}, "
            f"Fire F1: {val['fire_f1']:.4f}, Fire IoU: {val['fire_iou']:.4f}, "
            f"Background F1: {val['background_f1']:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # With WANDB_MODE=disabled the run folder is not created for us.
        os.makedirs(wandb.run.dir, exist_ok=True)
        checkpoint_path = os.path.join(
            wandb.run.dir,
            checkpoint_name.format(epoch=epoch + 1),
        )
        # 3. Save the model if this epoch's validation loss is among the best so far.
        best_checkpoints.offer(
            val["loss"],
            checkpoint_path,
            lambda: save_checkpoint(
                model,
                optimizer,
                epoch,
                val["loss"],
                checkpoint_path,
                checkpoint_extra,
                artifact_name,
            ),
        )

        # 4. Optional learning-rate decay (none by default).
        step_lr_scheduler(lr_scheduler, val["loss"])

        # 5. Optional: test-fire scores with the current weights, for the charts only.
        if test_every and score_test_fires is not None and (epoch + 1) % test_every == 0:
            test_scores = score_test_fires_untouched(score_test_fires)
            wandb.log(
                {
                    metric_labels["epoch"]: epoch,
                    metric_labels["periodic_test_f1"]: test_scores["f1"],
                    metric_labels["periodic_test_iou"]: test_scores["iou"],
                }
            )
            print(
                f"Epoch {epoch + 1}, test fires (current weights, logging only): "
                f"F1 {test_scores['f1']:.4f}, IoU {test_scores['iou']:.4f}"
            )

    # The checkpoint with the lowest validation loss is the one that gets tested.
    best_checkpoints.print_summary()
    return best_checkpoints.best_path()
