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
from torch import optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from firepred.data.fire_dataset import batch_inputs
from firepred.models.registry import forward_model
from firepred.training.accelerator import Device
from firepred.training.objectives import EpochScores, Scorer, TrainingLoss


class LearningRateSchedule:
    """Lower the learning rate once per epoch, or keep it constant.

    This is ours, not the paper's [ours]. "none" keeps the paper's constant
    rate. "plateau" multiplies the rate by decay_factor after patience epochs
    without a lower validation loss; "cosine" lowers it smoothly to min_lr over
    the run.

    Attributes:
        scheduler (torch.optim.lr_scheduler.LRScheduler | ReduceLROnPlateau | None):
            the torch scheduler, or None for "none".
    """

    def __init__(self, optimizer, schedule, max_epochs):
        """Build the torch scheduler for a ScheduleConfig.

        The scheduler changes the optimizer's rate in place. Nothing changes
        until step is called.

        Args:
            optimizer (torch.optim.Optimizer): the optimizer.
            schedule (ScheduleConfig): the schedule.
            max_epochs (int): the run's epochs (cosine's length).

        Raises:
            ValueError: on an unknown schedule.
        """
        if schedule.name == "none":
            self.scheduler = None
        elif schedule.name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=schedule.decay_factor, patience=schedule.patience,
                min_lr=schedule.min_lr,
            )
        elif schedule.name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=schedule.min_lr)
        else:
            raise ValueError(f"Unsupported learning-rate schedule: {schedule.name}")

    def step(self, val_loss):
        """Advance the schedule by one epoch.

        Plateau needs the validation loss; cosine does not. Without a schedule
        nothing happens.

        Args:
            val_loss (float): this epoch's validation loss.

        Returns:
            LearningRateSchedule: this schedule.
        """
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        elif self.scheduler is not None:
            self.scheduler.step()
        return self


class Trainer:
    """Train one model: single steps, whole epochs and the epoch loop.

    The trainer owns the loss, the optimizer, the gradient scaler and the
    schedule. Everything else (data, tracker, checkpoint store, evaluator) is
    handed to fit, and results only flow out to those objects.
    """

    def __init__(self, config, model, device):
        """Build the loss, Adam, the gradient scaler, the schedule and the scorer.

        The trainer holds no data, so cli.check can build one for a single step.
        Adam runs at a constant learning rate with no weight decay, as in the
        paper [paper]: the run config logs weight_decay = lr / 10, but it never
        reached the optimizer (docs/KNOWN_ISSUES.md). The gradient scaler is
        active on CUDA only.

        Args:
            config (RunConfig): the run's settings.
            model (torch.nn.Module): the model.
            device (Device | torch.device): where it runs.
        """
        self.config = config
        self.model = model
        self.device = Device.of(device)
        self.criterion = TrainingLoss(config.loss).to(self.device.torch_device)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        # The scaler protects small gradients in mixed precision; elsewhere it is a no-op.
        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        self.schedule = LearningRateSchedule(self.optimizer, config.schedule, config.epochs)
        self.scorer = Scorer()

    @property
    def learning_rate(self):
        """Give the optimizer's current learning rate.

        It changes only with a schedule. It is logged every epoch.

        Returns:
            float: the learning rate.
        """
        return self.optimizer.param_groups[0]["lr"]

    def step(self, inputs, labels):
        """Take one training step: forward, loss, backward and an Adam update.

        The model is put in train mode first. The logits are returned so the
        caller can score them.

        Args:
            inputs (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): input of shape
                (B, C, T, H, W), or (data, static) for a model with a TESSERA stem.
            labels (torch.Tensor): one-hot targets of shape (B, 2, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict]: (logits, loss, loss parts).
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Per-day logits averaged over the days -> (B, 2, H, W), then the loss.
        outputs = forward_model(self.model, inputs)
        loss, parts = self.criterion(outputs, labels)
        self.device.mark_step()  # TPU only: run the forward graph once, before the scores read it.
        # The scaler only matters for mixed precision on GPU.
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.device.mark_step()  # TPU only: run the backward pass and the Adam update.
        return outputs, loss, parts

    def train_epoch(self, loader, epoch):
        """Train for one pass over the training windows.

        Every batch takes one step, and fire and background F1 and IoU and the
        false-positive and false-negative counts are computed on its predictions
        for the charts. Losses are weighted by batch size, so the results are
        per-window means. A loss that becomes infinite stops the run.

        Args:
            loader (torch.utils.data.DataLoader): the training windows.
            epoch (int): the 0-based epoch (for the progress bar).

        Returns:
            EpochScores: the epoch's training scores.

        Raises:
            RuntimeError: if the loss becomes non-finite.
        """
        windows = len(loader.dataset)
        totals = {"loss": 0.0, "fire_f1": 0.0, "background_f1": 0.0, "fire_iou": 0.0, "background_iou": 0.0,
                  "fire_false_positives": 0.0, "fire_false_negatives": 0.0}
        components = {}
        bar = tqdm(loader, total=len(loader), desc=f"Train epoch {epoch + 1}/{self.config.epochs}")
        for batch_idx, batch in enumerate(bar):
            inputs = batch_inputs(batch, self.device.torch_device)
            labels = batch["labels"].float().to(self.device.torch_device, non_blocking=True)
            outputs, loss, parts = self.step(inputs, labels)
            # Training scores, for logging only (no gradients, no effect on learning).
            with torch.no_grad():
                predictions = self.scorer.predictions(outputs.detach())
                totals["fire_f1"] += self.scorer.fire_f1(predictions, labels).sum().item()
                totals["background_f1"] += self.scorer.background_f1(predictions, labels).sum().item()
                totals["fire_iou"] += self.scorer.fire_iou(predictions, labels).sum().item()
                totals["background_iou"] += self.scorer.background_iou(predictions, labels).sum().item()
                totals["fire_false_positives"] += self.scorer.fire_false_positives(predictions, labels).sum().item()
                totals["fire_false_negatives"] += self.scorer.fire_false_negatives(predictions, labels).sum().item()
            # Losses weighted by batch size, so the epoch value is a per-window mean.
            totals["loss"] += loss.detach().item() * labels.size(0)
            for name, value in parts.items():
                components[name] = components.get(name, 0.0) + value.item() * labels.size(0)
            bar.set_postfix(loss=f"{totals['loss'] / ((batch_idx + 1) * labels.size(0)):.4f}")
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
            background_iou=totals["background_iou"] / windows,
            fire_false_positives=totals["fire_false_positives"] / windows,
            fire_false_negatives=totals["fire_false_negatives"] / windows,
        )

    def validate(self, loader, epoch):
        """Score the validation windows with the current weights.

        The model is in eval mode and no gradients are computed. Like training,
        every score is a per-window mean.

        Args:
            loader (torch.utils.data.DataLoader): the validation windows.
            epoch (int): the 0-based epoch (for the progress bar).

        Returns:
            EpochScores: the epoch's validation scores.
        """
        self.model.eval()
        windows = len(loader.dataset)
        totals = {"loss": 0.0, "fire_f1": 0.0, "background_f1": 0.0, "fire_iou": 0.0, "background_iou": 0.0,
                  "fire_false_positives": 0.0, "fire_false_negatives": 0.0}
        components = {}
        bar = tqdm(loader, total=len(loader), desc=f"Validation epoch {epoch + 1}/{self.config.epochs}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(bar):
                inputs = batch_inputs(batch, self.device.torch_device)
                labels = batch["labels"].float().to(self.device.torch_device, non_blocking=True)
                outputs = forward_model(self.model, inputs)
                loss, parts = self.criterion(outputs, labels)
                self.device.mark_step()  # TPU only: run the forward graph once for loss and scores.
                totals["loss"] += loss.detach().item() * labels.size(0)
                for name, value in parts.items():
                    components[name] = components.get(name, 0.0) + value.item() * labels.size(0)
                predictions = self.scorer.predictions(outputs)
                totals["fire_f1"] += self.scorer.fire_f1(predictions, labels).sum().item()
                totals["background_f1"] += self.scorer.background_f1(predictions, labels).sum().item()
                totals["fire_iou"] += self.scorer.fire_iou(predictions, labels).sum().item()
                totals["background_iou"] += self.scorer.background_iou(predictions, labels).sum().item()
                totals["fire_false_positives"] += self.scorer.fire_false_positives(predictions, labels).sum().item()
                totals["fire_false_negatives"] += self.scorer.fire_false_negatives(predictions, labels).sum().item()
                bar.set_postfix(loss=f"{totals['loss'] / ((batch_idx + 1) * labels.size(0)):.4f}")
        return EpochScores(
            split="val",
            loss=totals["loss"] / windows,
            components={f"val_{name}": total / windows for name, total in components.items()},
            fire_f1=totals["fire_f1"] / windows,
            background_f1=totals["background_f1"] / windows,
            fire_iou=totals["fire_iou"] / windows,
            background_iou=totals["background_iou"] / windows,
            fire_false_positives=totals["fire_false_positives"] / windows,
            fire_false_negatives=totals["fire_false_negatives"] / windows,
        )

    def periodic_test_due(self, epoch, evaluator):
        """Say whether to score the test fires after this epoch.

        It is due every --test-every epochs, when an evaluator exists. 0 turns it
        off.

        Args:
            epoch (int): the 0-based epoch.
            evaluator (Evaluator | None): the test-fire evaluator.

        Returns:
            bool: True when due.
        """
        every = self.config.test_every
        return evaluator is not None and every > 0 and (epoch + 1) % every == 0

    def fit(self, data, tracker, store, evaluator=None):
        """Train for every epoch and return the checkpoint with the lowest validation loss.

        Each epoch: train, validate, offer the checkpoint to the store (uploading
        it when asked), step the schedule, and every --test-every epochs score the
        test fires for the charts. The checkpoint is saved before the schedule
        steps, so it keeps that epoch's learning rate. Results only flow out.

        Args:
            data (TrainingData): the training and validation loaders.
            tracker (Tracker): where results are logged.
            store (CheckpointStore): where checkpoints are kept.
            evaluator (Evaluator | None): scores the test fires, or None.

        Returns:
            Path: the best checkpoint.
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
            self.schedule.step(val.loss)
            # 5. Optional test-fire scores with the current weights, for the charts only.
            if self.periodic_test_due(epoch, evaluator):
                tracker.log_periodic_test(epoch, evaluator.score(self.model))
        return store.best
