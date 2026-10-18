"""Every name a run produces: tags, W&B run name, checkpoint and artifact names.

Level 3 (training). Used by training/run.py and training/tracking.py.

These strings are how old runs and checkpoints are found again, so they never
change. A run with the paper's Dice+CE loss and TESSERA inputs, for example:
    run name    train_pred_swinunetr3d_num_heads_3_hidden_size_36_batchsize_8_tessera_dicece
    checkpoint  model_swinunetr3d_mode_pred_num_heads_3_hidden_size_36_batchsize_8_
                checkpoint_epoch_12_nc_172_ts_6_tessera_dicece.pth
"""

from pathlib import Path


def loss_config(loss, ce_weight, lambda_dice, lambda_ce, lambda_focal=None, focal_gamma=None, focal_alpha=None):
    """The loss settings recorded in W&B and in every checkpoint."""
    config = {"loss": loss}
    if loss == "dicece":
        config.update(
            ce_weight=list(ce_weight),
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    if loss == "dicefocal":
        config.update(
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
    return config


def lr_config(schedule, lr_min, decay_factor, patience):
    """The learning-rate schedule recorded in W&B and in every checkpoint, so
    decayed runs are never mistaken for paper runs. [ours]"""
    config = {"lr_schedule": schedule}
    if schedule != "none":
        config["lr_min"] = lr_min
    if schedule == "plateau":
        config.update(
            lr_decay_factor=decay_factor,
            lr_patience=patience,
        )
    return config


def lr_tag(schedule):
    """'' for a constant rate, else '_lr-<schedule>'."""
    return "" if schedule == "none" else f"_lr-{schedule}"


def run_tag(use_tessera, loss, schedule):
    """Suffix for checkpoint names; the released-code Dice loss keeps the old names."""
    tessera_tag = "_tessera" if use_tessera else ""
    return tessera_tag + ("" if loss == "dice" else f"_{loss}") + lr_tag(schedule)


def run_name(run_test, mode, model_name, num_heads, hidden_size, batch_size, use_tessera, loss):
    """The W&B run name, before an lr tag is appended."""
    return (
        f"{'test' if run_test else 'train'}_{mode}_{model_name}_"
        f"num_heads_{num_heads}_hidden_size_{hidden_size}_"
        f"batchsize_{batch_size}{'_tessera' if use_tessera else ''}"
        f"{'' if loss == 'dice' else '_' + loss}"
    )


def checkpoint_template(model_name, mode, num_heads, hidden_size, batch_size, model_input_channels, ts_length, tag):
    """Checkpoint file name with a literal '{epoch}' placeholder (1-based epoch)."""
    return (
        f"model_{model_name}_mode_{mode}_num_heads_{num_heads}_"
        f"hidden_size_{hidden_size}_batchsize_{batch_size}_"
        "checkpoint_epoch_{epoch}_"
        f"nc_{model_input_channels}_ts_{ts_length}{tag}.pth"
    )


def default_test_checkpoint(template, load_epoch):
    """Where -test looks without --checkpoint. [released]

    Training saves into the W&B run directory instead (docs/KNOWN_ISSUES.md).
    """
    return Path("saved_models") / template.format(epoch=load_epoch)


def artifact_name(model_name, tag, run_id):
    """W&B model-artifact name for a run's best checkpoints."""
    return f"{model_name}{tag.replace('_', '-')}-checkpoints-{run_id}"
