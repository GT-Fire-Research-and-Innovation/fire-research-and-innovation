"""Optimizer, mixed-precision gradient scaler and learning-rate schedule.

Level 3 (training). Used by training/run.py and training/loop.py.
"""

from torch import optim
from torch.cuda.amp import GradScaler


def build_optimizer(model, learning_rate):
    """Adam at a constant `learning_rate`, no weight decay. [paper]

    (The run config logs weight_decay = lr / 10, but it never reached the
    optimizer; see docs/KNOWN_ISSUES.md.)
    """
    return optim.Adam(model.parameters(), lr=learning_rate)


def build_grad_scaler(device):
    """Loss scaling for CUDA; a no-op scaler on CPU."""
    return GradScaler(enabled=device.type == "cuda")


def build_lr_scheduler(optimizer, schedule, max_epochs, decay_factor, patience, min_lr):
    """Optional once-per-epoch learning-rate decay. [ours]

    none:    constant learning rate, as in the paper and the official code.
    plateau: multiply the rate by `decay_factor` after `patience` epochs
             without a lower validation loss, never below `min_lr`.
    cosine:  decay smoothly from the starting rate to `min_lr` over
             `max_epochs` epochs.
    """
    if schedule == "none":
        return None
    if schedule == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=decay_factor,
            patience=patience,
            min_lr=min_lr,
        )
    if schedule == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=min_lr,
        )
    raise ValueError(f"Unsupported learning-rate schedule: {schedule}")


def step_lr_scheduler(lr_scheduler, val_loss):
    """Advance the schedule once per epoch (plateau needs the validation loss)."""
    if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler.step(val_loss)
    elif lr_scheduler is not None:
        lr_scheduler.step()
