"""Weights & Biases: the one place a run talks to W&B.

Level 3 (training). Used by training/experiment.py (it opens the Tracker) and
training/trainer.py (it reports to it). Nothing else imports wandb.

    with Tracker(config, device) as tracker:   # login, init, charts, name, config
        tracker.checkpoint_dir                 # where this run's checkpoints go
        tracker.log_train / log_validation / log_periodic_test / log_checkpoint
        tracker.log_final(scores)
    # leaving the block finishes the run

Chart names are what the dashboards are built on; they never change.
"""

import os
from pathlib import Path

from firepred.config.training import WANDB_PROJECT


def chart_labels(loss_name):
    """{internal name: W&B chart name} for a run with `loss_name`; the names never change."""
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


class Tracker:
    """One W&B run: set up on entry, finished on exit; every log goes through it."""

    def __init__(self, config, device):
        """Remember the run's settings; nothing is sent until the block is entered."""
        self.config = config
        self.device = device
        self.labels = chart_labels(config.loss.name)
        self.wandb = None

    def __enter__(self):
        """Log in, start the run, declare the charts (training only), name it, record its config."""
        import wandb

        self.wandb = wandb
        wandb.login()
        wandb.init(project=WANDB_PROJECT, entity=self.config.wandb_entity)
        if not self.config.test_only:
            epoch = self.labels["epoch"]
            wandb.define_metric(epoch)
            for key, label in self.labels.items():
                if key != "epoch":
                    wandb.define_metric(label, step_metric=epoch)
        wandb.run.name = self.config.run_name
        wandb.config.update(self.config.wandb_config(on_tpu=getattr(self.device, "type", "") == "xla"))
        return self

    def __exit__(self, *exc_info):
        """Finish the run (also after an error); errors are not swallowed."""
        self.wandb.finish()
        return False

    @property
    def run_id(self):
        """The W&B run id."""
        return self.wandb.run.id

    @property
    def checkpoint_dir(self):
        """This run's folder (the W&B run directory), created if needed."""
        # With WANDB_MODE=disabled the run folder is not created for us.
        os.makedirs(self.wandb.run.dir, exist_ok=True)
        return Path(self.wandb.run.dir)

    def log_train(self, epoch, scores):
        """Chart one epoch's training EpochScores; returns them unchanged."""
        labels = self.labels
        self.wandb.log({
            labels["epoch"]: epoch,
            labels["train_loss"]: scores.loss,
            labels["train_fire_f1"]: scores.fire_f1,
            labels["train_background_f1"]: scores.background_f1,
            labels["train_fire_iou"]: scores.fire_iou,
            **{labels[name]: value for name, value in scores.components.items()},
        })
        return scores

    def log_validation(self, epoch, scores, learning_rate):
        """Chart one epoch's validation EpochScores and the learning rate; returns the scores unchanged."""
        labels = self.labels
        self.wandb.log({
            labels["epoch"]: epoch,
            labels["val_loss"]: scores.loss,
            labels["miou"]: scores.miou,
            labels["mdice"]: scores.mdice,
            labels["val_fire_f1"]: scores.fire_f1,
            labels["val_background_f1"]: scores.background_f1,
            labels["val_fire_iou"]: scores.fire_iou,
            labels["learning_rate"]: learning_rate,
            **{labels[name]: value for name, value in scores.components.items()},
        })
        return scores

    def log_periodic_test(self, epoch, scores):
        """Chart test-fire scores with the current weights (logging only). [ours]"""
        self.wandb.log({
            self.labels["epoch"]: epoch,
            self.labels["periodic_test_f1"]: scores.f1,
            self.labels["periodic_test_iou"]: scores.iou,
        })
        return scores

    def log_checkpoint(self, path, epoch, val_loss, settings):
        """Store a saved checkpoint as a new version of the run's W&B model Artifact; return its name.

        Online, wait for the upload so a later best checkpoint cannot replace
        the file first; offline runs log it locally for a later `wandb sync`.
        """
        metadata = {"epoch": epoch + 1, "validation_loss": float(val_loss)}
        for key in ("use_tessera", "model_input_channels"):
            if key in settings:
                metadata[key] = settings[key]
        name = self.config.artifact_name(self.run_id)
        artifact = self.wandb.Artifact(name=name, type="model", metadata=metadata)
        artifact.add_file(str(path), name=Path(path).name)
        uploaded = self.wandb.run.log_artifact(artifact, aliases=["best", "latest", f"epoch-{epoch + 1}"])
        if not self.wandb.run.offline:
            uploaded.wait()
        print(f"Checkpoint logged to W&B Artifacts: {name}")
        return name

    def log_final(self, scores):
        """Log the test scores of the tested checkpoint (the paper's metrics) and put them in the run summary."""
        metrics = {
            "test_f1": scores.f1,
            "test_iou": scores.iou,
            "Test F1 — Next-Day Fire Growth": scores.f1,
            "Test IoU — Next-Day Fire Growth": scores.iou,
            "Test F1 — Background": scores.background_f1,
            "test_fires": scores.fires,
        }
        self.wandb.log(metrics)
        self.wandb.run.summary.update(metrics)
        return metrics


def check_wandb_access(entity, project, name="permission-check", attempts=5):
    """Prove the logged-in key can write to entity/project: log one tiny run, then delete it.

    For notebooks, before a long run. Returns the check run's URL. A run that
    cannot be deleted right away (the API can lag) is left for the W&B UI and
    reported in a printed note.
    """
    import time

    import wandb

    with wandb.init(entity=entity, project=project, name=name, mode="online") as run:
        run.log({"permission_check": 1})
        url = run.url
    path = f"{entity}/{project}/{run.id}"
    for attempt in range(attempts):
        try:
            check_run = wandb.Api().run(path)
            if check_run.name != name:
                raise RuntimeError(f"{path} is not the permission-check run; not deleting it")
            check_run.delete()
            break
        except RuntimeError:
            raise
        except Exception as error:  # a just-finished run can take a few seconds to appear in the API
            if attempt == attempts - 1:
                print(f"Could not delete {path} ({error}); delete it in the W&B UI if needed.")
            else:
                time.sleep(3)
    return url
