"""Weights & Biases: run setup, chart names, and the final test log.

Level 3 (training). Used by training/run.py, training/loop.py and
training/checkpoints.py. Chart names are what the dashboards are built on;
they never change.
"""

import wandb

from firepred.config.training import WANDB_PROJECT
from firepred.training.naming import run_name


def build_wandb_metric_labels(loss_name):
    """Describe the task, loss, and classes included in each graph."""
    loss_title = {
        "dice": "Dice Loss",
        "dicece": "Weighted Dice + Cross-Entropy Loss",
        "dicefocal": "Dice + Focal Loss",
    }[loss_name]
    labels = {
        "epoch": "Training Epoch (Starting at 0)",
        "train_loss": f"Training {loss_title} — Next-Day Fire Growth",
        "val_loss": f"Validation {loss_title} — Next-Day Fire Growth",
        "miou": "Validation Mean IoU — Background & Fire Growth",
        "mdice": "Validation Mean Dice (F1) — Background & Fire Growth",
        "train_fire_f1": "Training F1 — Next-Day Fire Growth",
        "train_background_f1": "Training F1 — Background",
        "val_fire_f1": "Validation F1 — Next-Day Fire Growth",
        "val_background_f1": "Validation F1 — Background",
        # The paper's IoU (Equation 2) for the fire class; logged, never trained on.
        "train_fire_iou": "Training IoU — Next-Day Fire Growth",
        "val_fire_iou": "Validation IoU — Next-Day Fire Growth",
        # [ours] test fires scored with the current weights every --test-every
        # epochs (logging only).
        "periodic_test_f1": "Test F1 — Next-Day Fire Growth (periodic, current weights)",
        "periodic_test_iou": "Test IoU — Next-Day Fire Growth (periodic, current weights)",
        # Constant unless the [ours] --lr-schedule is used.
        "learning_rate": "Learning Rate",
    }
    if loss_name == "dicece":
        for prefix, split in (("train", "Training"), ("val", "Validation")):
            labels[f"{prefix}_dice_loss"] = (
                f"{split} Weighted Dice Loss — Next-Day Fire Growth"
            )
            labels[f"{prefix}_ce_loss"] = (
                f"{split} Weighted Cross-Entropy Loss — Next-Day Fire Growth"
            )
    if loss_name == "dicefocal":
        for prefix, split in (("train", "Training"), ("val", "Validation")):
            labels[f"{prefix}_dice_loss"] = f"{split} Dice Loss — Next-Day Fire Growth"
            labels[f"{prefix}_focal_loss"] = f"{split} Focal Loss — Next-Day Fire Growth"
    return labels


def configure_wandb(
    model_name,
    mode,
    num_heads,
    hidden_size,
    batch_size,
    learning_rate,
    weight_decay,
    max_epochs,
    wandb_user_name,
    run_test,
    use_tessera,
    model_input_channels,
    loss_config,
):
    """Initialize one W&B run for either training or testing."""
    # Start the W&B run; training runs also declare every chart against the epoch axis.
    wandb.login()
    wandb.init(project=WANDB_PROJECT, entity=wandb_user_name)
    if not run_test:
        metric_labels = build_wandb_metric_labels(loss_config["loss"])
        epoch_metric = metric_labels["epoch"]
        wandb.define_metric(epoch_metric)
        for key, label in metric_labels.items():
            if key != "epoch":
                wandb.define_metric(label, step_metric=epoch_metric)
    # Name the run from its settings, and record the hyperparameters in its config.
    wandb.run.name = run_name(
        run_test, mode, model_name, num_heads, hidden_size, batch_size, use_tessera, loss_config["loss"]
    )
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": max_epochs,
            "batch_size": batch_size,
            "mode": mode,
            "model": model_name,
            "tessera": use_tessera,
            "input_channels": model_input_channels,
            **loss_config,
        }
    )


def log_final_test(scores):
    """Log the test scores of the best-validation checkpoint (the paper's metrics)."""
    # Both short keys (test_f1) and readable chart names, for the paper's two metrics.
    test_metrics = {
        "test_f1": scores["f1"],
        "test_iou": scores["iou"],
        "Test F1 — Next-Day Fire Growth": scores["f1"],
        "Test IoU — Next-Day Fire Growth": scores["iou"],
        "Test F1 — Background": scores["background_f1"],
        "test_fires": scores["fires"],
    }
    wandb.log(test_metrics)
    # One value per run: the summary shows it in the run overview and table.
    wandb.run.summary.update(test_metrics)
