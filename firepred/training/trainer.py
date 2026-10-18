"""Training: one Trainer owns the loss, optimizer, scaler and schedule, and runs the epochs.

Level 3 (training). Built by training/experiment.py (and cli/check.py for a
single step).

Information flows one way. The Trainer receives everything it works with:
its constructor takes the settings, the model and the device; fit() takes the
data, the Tracker it reports to, the CheckpointStore it saves into and an
optional Evaluator for periodic test scores. It sends results to those
objects and returns the best checkpoint; none of them calls back into it.
"""

import numpy as np
import torch
from monai.data import decollate_batch
from torch import optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from firepred.data.fire_dataset import batch_inputs
from firepred.models.registry import forward_model
from firepred.training.accelerator import mark_step
from firepred.training.objectives import (
    EpochScores,
    background_f1,
    build_criterion,
    build_post_transform,
    fire_only_f1,
    fire_only_iou,
    validation_dice,
    validation_iou,
)


def build_optimizer(model, learning_rate):
    """Adam at a constant `learning_rate`, no weight decay. [paper]

    (The run config logs weight_decay = lr / 10, but it never reached the
    optimizer; see docs/KNOWN_ISSUES.md.)
    """
    return optim.Adam(model.parameters(), lr=learning_rate)


def build_grad_scaler(device):
    """Loss scaling for CUDA; a no-op scaler on CPU."""
    return GradScaler(enabled=device.type == "cuda")


def build_lr_scheduler(optimizer, schedule, max_epochs):
    """Optional once-per-epoch learning-rate decay for a ScheduleConfig, or None. [ours]

    none:    constant learning rate, as in the paper and the official code.
    plateau: multiply the rate by `decay_factor` after `patience` epochs
             without a lower validation loss, never below `min_lr`.
    cosine:  decay smoothly from the starting rate to `min_lr` over
             `max_epochs` epochs.
    """
    if schedule.name == "none":
        return None
    if schedule.name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=schedule.decay_factor, patience=schedule.patience, min_lr=schedule.min_lr
        )
    if schedule.name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=schedule.min_lr)
    raise ValueError(f"Unsupported learning-rate schedule: {schedule.name}")


def step_lr_scheduler(lr_scheduler, val_loss):
    """Advance the schedule once per epoch (plateau needs the validation loss); return the scheduler."""
    if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler.step(val_loss)
    elif lr_scheduler is not None:
        lr_scheduler.step()
    return lr_scheduler


class Trainer:
    """Trains one model: steps, epochs and the epoch loop."""

    def __init__(self, config, model, device):
        """Build the loss, Adam, the gradient scaler, the LR schedule and the thresholding transform."""
        self.config = config
        self.model = model
        self.device = device
        self.criterion = build_criterion(config.loss).to(device)
        self.optimizer = build_optimizer(model, config.learning_rate)
        self.scaler = build_grad_scaler(device)
        self.scheduler = build_lr_scheduler(self.optimizer, config.schedule, config.epochs)
        self.post_transform = build_post_transform()

    @property
    def learning_rate(self):
        """The optimizer's current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def predictions(self, outputs):
        """Thresholded (B, 2, H, W) masks from logits."""
        return torch.stack([self.post_transform(item) for item in decollate_batch(outputs)])

    def step(self, inputs, labels):
        """One forward/backward/Adam step in train mode; return (logits, loss, loss parts)."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Per-day logits averaged over the days -> (B, 2, H, W), then the loss.
        outputs = forward_model(self.model, inputs)
        loss, parts = self.criterion(outputs, labels)
        mark_step(self.device)  # TPU only: run the forward graph once, before the scores read it.
        # The scaler only matters for mixed precision on GPU.
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        mark_step(self.device)  # TPU only: run the backward pass and the Adam update.
        return outputs, loss, parts

    def train_epoch(self, loader, epoch):
        """One pass over the training windows; returns the epoch's per-window means as EpochScores."""
        windows = len(loader.dataset)
        totals = {"loss": 0.0, "fire_f1": 0.0, "background_f1": 0.0, "fire_iou": 0.0}
        components = {}
        bar = tqdm(loader, total=len(loader), desc=f"Train epoch {epoch + 1}/{self.config.epochs}")
        for batch_idx, batch in enumerate(bar):
            inputs = batch_inputs(batch, self.device)
            labels = batch["labels"].float().to(self.device, non_blocking=True)
            outputs, loss, parts = self.step(inputs, labels)
            # Training scores, for logging only (no gradients, no effect on learning).
            with torch.no_grad():
                predictions = self.predictions(outputs.detach())
                totals["fire_f1"] += fire_only_f1(predictions, labels).sum().item()
                totals["background_f1"] += background_f1(predictions, labels).sum().item()
                totals["fire_iou"] += fire_only_iou(predictions, labels).sum().item()
            # Losses weighted by batch size, so the epoch value is a per-window mean.
            totals["loss"] += loss.detach().item() * inputs.size(0)
            for name, value in parts.items():
                components[name] = components.get(name, 0.0) + value.item() * inputs.size(0)
            bar.set_postfix(loss=f"{totals['loss'] / ((batch_idx + 1) * inputs.size(0)):.4f}")
            # Stop immediately if the loss blows up instead of training on garbage.
            if not np.isfinite(totals["loss"]):
                raise RuntimeError(f"Training loss became non-finite at step {batch_idx}")
        return EpochScores(
            split="train",
            loss=totals["loss"] / windows,
            components={f"train_{name}": total / windows for name, total in components.items()},
            fire_f1=totals["fire_f1"] / windows,
            background_f1=totals["background_f1"] / windows,
            fire_iou=totals["fire_iou"] / windows,
        )

    def validate(self, loader, epoch):
        """Loss and scores on the validation windows (eval mode, no gradients), as EpochScores.

        miou / mdice are means of per-batch means (docs/KNOWN_ISSUES.md).
        """
        self.model.eval()
        windows = len(loader.dataset)
        totals = {"loss": 0.0, "fire_f1": 0.0, "background_f1": 0.0, "fire_iou": 0.0}
        components, iou_values, dice_values = {}, [], []
        bar = tqdm(loader, total=len(loader), desc=f"Validation epoch {epoch + 1}/{self.config.epochs}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(bar):
                inputs = batch_inputs(batch, self.device)
                labels = batch["labels"].float().to(self.device, non_blocking=True)
                outputs = forward_model(self.model, inputs)
                loss, parts = self.criterion(outputs, labels)
                mark_step(self.device)  # TPU only: run the forward graph once for loss and scores.
                totals["loss"] += loss.detach().item() * inputs.size(0)
                for name, value in parts.items():
                    components[name] = components.get(name, 0.0) + value.item() * inputs.size(0)
                predictions = self.predictions(outputs)
                totals["fire_f1"] += fire_only_f1(predictions, labels).sum().item()
                totals["background_f1"] += background_f1(predictions, labels).sum().item()
                totals["fire_iou"] += fire_only_iou(predictions, labels).sum().item()
                iou_values.append(validation_iou(predictions, labels).mean().item())
                dice_values.append(validation_dice(predictions, labels).mean().item())
                bar.set_postfix(loss=f"{totals['loss'] / ((batch_idx + 1) * inputs.size(0)):.4f}")
        return EpochScores(
            split="val",
            loss=totals["loss"] / windows,
            components={f"val_{name}": total / windows for name, total in components.items()},
            fire_f1=totals["fire_f1"] / windows,
            background_f1=totals["background_f1"] / windows,
            fire_iou=totals["fire_iou"] / windows,
            miou=float(np.mean(iou_values)),
            mdice=float(np.mean(dice_values)),
        )

    def periodic_test_due(self, epoch, evaluator):
        """True when the test fires should be scored after `epoch` (0-based)."""
        every = self.config.test_every
        return evaluator is not None and every > 0 and (epoch + 1) % every == 0

    def fit(self, data, tracker, store, evaluator=None):
        """Train for config.epochs and return the path of the lowest-validation-loss checkpoint.

        Each epoch: train, validate, offer the checkpoint to `store`, step the
        learning-rate schedule, and (every config.test_every epochs, when an
        `evaluator` is given) score the test fires for the charts only.
        Every result goes to `tracker`; nothing here reads it back.
        """
        for epoch in range(self.config.epochs):
            # 1. Train, then chart the epoch's training scores.
            train = tracker.log_train(epoch, self.train_epoch(data.train_loader, epoch))
            # 2. Validate with the same weights, and chart it with this epoch's learning rate.
            val = tracker.log_validation(epoch, self.validate(data.val_loader, epoch), self.learning_rate)
            print(f"Epoch {epoch + 1}: train loss {train.loss:.4f}, validation loss {val.loss:.4f}, "
                  f"fire F1 {val.fire_f1:.4f}, fire IoU {val.fire_iou:.4f}, LR {self.learning_rate:.2e}")
            # 3. Keep the checkpoint if its validation loss is among the best so far. This
            #    comes before the schedule step, so the saved optimizer has this epoch's rate.
            saved = store.offer(epoch, val.loss, self.model, self.optimizer)
            if saved is not None and self.config.upload_checkpoints:
                tracker.log_checkpoint(saved, epoch, val.loss, store.settings)
            # 4. Optional learning-rate decay (plateau reads the validation loss).
            step_lr_scheduler(self.scheduler, val.loss)
            # 5. Optional test-fire scores with the current weights, for the charts only.
            if self.periodic_test_due(epoch, evaluator):
                tracker.log_periodic_test(epoch, evaluator.score(self.model))
        return store.best
