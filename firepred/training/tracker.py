"""Weights & Biases: the one place a run talks to W&B.

Level 3 (training). Used by training/experiment.py (it opens the Tracker) and
training/trainer.py (it reports to it). Nothing else imports wandb.

    with Tracker(config, device) as tracker:   # login, init, charts, name, config
        tracker.checkpoint_dir                 # where this run's checkpoints go
        tracker.log_train / log_validation / log_periodic_test / log_checkpoint
        tracker.log_final(scores)
    # leaving the block finishes the run

Chart names (Tracker.chart_labels) are what the dashboards are built on; they
never change. Tracker.check_access checks a W&B key before a long run.
"""

import os
from pathlib import Path

from firepred.config.training import WANDB_PROJECT
from firepred.training.accelerator import Device


class Tracker:
    """One W&B run: set up when the `with` block starts, finished when it ends.

    Every log of the run goes through this class, and no other module imports
    W&B. It also decides where checkpoints are written (the run folder).
    """

    @staticmethod
    def chart_labels(loss_name):
        """Give the W&B chart name of every logged value for a loss.

        Dashboards are built on these names, so they never change. Dice+CE and
        Dice+focal also chart their two parts.

        Args:
            loss_name (str): "dice", "dicece" or "dicefocal".

        Returns:
            dict[str, str]: {internal name: chart name}.
        """
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
            # [ours] the two kinds of error behind IoU: false alarms and missed fire
            # pixels, as mean pixels per window (a window has 256 x 256 pixels).
            "train_fire_false_positives": "Training False Positives — Next-Day Fire Growth (pixels per window)",
            "train_fire_false_negatives": "Training False Negatives — Next-Day Fire Growth (pixels per window)",
            "val_fire_false_positives": "Validation False Positives — Next-Day Fire Growth (pixels per window)",
            "val_fire_false_negatives": "Validation False Negatives — Next-Day Fire Growth (pixels per window)",
            # [ours] test fires scored with the current weights every --test-every
            # epochs (logging only).
            "periodic_test_f1": "Test F1 — Next-Day Fire Growth (periodic, current weights)",
            "periodic_test_iou": "Test IoU — Next-Day Fire Growth (periodic, current weights)",
            "periodic_test_false_positives": (
                "Test False Positives — Next-Day Fire Growth (pixels per window, periodic, current weights)"
            ),
            "periodic_test_false_negatives": (
                "Test False Negatives — Next-Day Fire Growth (pixels per window, periodic, current weights)"
            ),
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

    def __init__(self, config, device):
        """Remember the run's settings; nothing is sent to W&B yet.

        The chart names follow from the loss. W&B is imported on entering the block.

        Args:
            config (RunConfig): the run's settings.
            device (Device | torch.device): the training device.
        """
        self.config = config
        self.device = Device.of(device)
        self.labels = self.chart_labels(config.loss.name)
        self.wandb = None

    def __enter__(self):
        """Log in, start the run, declare the charts, name the run and record its config.

        Test runs declare no charts. The run name and config come from the
        RunConfig.

        Returns:
            Tracker: this tracker.
        """
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
        wandb.config.update(self.config.wandb_config(on_tpu=self.device.is_xla))
        return self

    def __exit__(self, *exc_info):
        """Finish the W&B run, also after an error.

        The run is closed whether the block ended normally or with an error.
        Errors raised inside the block are not swallowed.

        Args:
            *exc_info: the exception details, if any.

        Returns:
            bool: False, so exceptions propagate.
        """
        self.wandb.finish()
        return False

    @property
    def run_id(self):
        """Give the W&B run id.

        The artifact name includes it. It is set once the run starts.

        Returns:
            str: the run id.
        """
        return self.wandb.run.id

    @property
    def checkpoint_dir(self):
        """Give the run's folder, where checkpoints are written.

        It is the W&B run directory. It is created if needed, because W&B does not
        create it when disabled.

        Returns:
            Path: the folder.
        """
        # With WANDB_MODE=disabled the run folder is not created for us.
        os.makedirs(self.wandb.run.dir, exist_ok=True)
        return Path(self.wandb.run.dir)

    def log_train(self, epoch, scores):
        """Chart one epoch's training scores.

        The scores are returned unchanged. The trainer can therefore log and keep
        them in one line.

        Args:
            epoch (int): the 0-based epoch.
            scores (EpochScores): the training scores.

        Returns:
            EpochScores: scores.
        """
        labels = self.labels
        self.wandb.log({
            labels["epoch"]: epoch,
            labels["train_loss"]: scores.loss,
            labels["train_fire_f1"]: scores.fire_f1,
            labels["train_background_f1"]: scores.background_f1,
            labels["train_fire_iou"]: scores.fire_iou,
            labels["train_fire_false_positives"]: scores.fire_false_positives,
            labels["train_fire_false_negatives"]: scores.fire_false_negatives,
            **{labels[name]: value for name, value in scores.components.items()},
        })
        return scores

    def log_validation(self, epoch, scores, learning_rate):
        """Chart one epoch's validation scores and the learning rate.

        The scores are returned unchanged. The learning rate shows the schedule's
        effect.

        Args:
            epoch (int): the 0-based epoch.
            scores (EpochScores): the validation scores.
            learning_rate (float): the optimizer's current rate.

        Returns:
            EpochScores: scores.
        """
        labels = self.labels
        self.wandb.log({
            labels["epoch"]: epoch,
            labels["val_loss"]: scores.loss,
            labels["miou"]: scores.miou,
            labels["mdice"]: scores.mdice,
            labels["val_fire_f1"]: scores.fire_f1,
            labels["val_background_f1"]: scores.background_f1,
            labels["val_fire_iou"]: scores.fire_iou,
            labels["val_fire_false_positives"]: scores.fire_false_positives,
            labels["val_fire_false_negatives"]: scores.fire_false_negatives,
            labels["learning_rate"]: learning_rate,
            **{labels[name]: value for name, value in scores.components.items()},
        })
        return scores

    def log_periodic_test(self, epoch, scores):
        """Chart test-fire scores computed with the current weights.

        This is ours, not the paper's [ours]. These are for watching progress only; the kept checkpoint is still chosen
        by validation loss. The scores are returned unchanged.

        Args:
            epoch (int): the 0-based epoch.
            scores (TestScores): the test scores.

        Returns:
            TestScores: scores.
        """
        self.wandb.log({
            self.labels["epoch"]: epoch,
            self.labels["periodic_test_f1"]: scores.f1,
            self.labels["periodic_test_iou"]: scores.iou,
            self.labels["periodic_test_false_positives"]: scores.false_positives,
            self.labels["periodic_test_false_negatives"]: scores.false_negatives,
        })
        return scores

    def log_checkpoint(self, path, epoch, val_loss, settings):
        """Upload a saved checkpoint as a new version of the run's W&B model artifact.

        The version is tagged best, latest and epoch-N. Online, the upload is
        waited for, so a later checkpoint cannot replace the file first.

        Args:
            path (str | Path): the checkpoint file.
            epoch (int): the 0-based epoch.
            val_loss (float): its validation loss.
            settings (dict): the run settings stored in it.

        Returns:
            str: the artifact name.
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
        """Log the final test scores and put them in the run summary.

        These are the paper's metrics for the tested checkpoint. They appear both
        as charts and in the run overview.

        Args:
            scores (TestScores): the final test scores.

        Returns:
            dict: the logged values.
        """
        metrics = {
            "test_f1": scores.f1,
            "test_iou": scores.iou,
            "Test F1 — Next-Day Fire Growth": scores.f1,
            "Test IoU — Next-Day Fire Growth": scores.iou,
            "Test F1 — Background": scores.background_f1,
            "test_fires": scores.fires,
            # [ours] mean over fires of each fire's per-window error counts.
            "test_false_positives": scores.false_positives,
            "test_false_negatives": scores.false_negatives,
            "Test False Positives — Next-Day Fire Growth (pixels per window)": scores.false_positives,
            "Test False Negatives — Next-Day Fire Growth (pixels per window)": scores.false_negatives,
        }
        self.wandb.log(metrics)
        self.wandb.run.summary.update(metrics)
        return metrics

    @staticmethod
    def check_access(entity, project, name="permission-check", attempts=5):
        """Check that the W&B key can write to a project: log a tiny run, then delete it.

        Notebooks run this before a long training run, so a permission problem
        shows at once. A run that cannot be deleted right away (the API can lag) is
        reported and left for the W&B UI.

        Args:
            entity (str): the W&B team.
            project (str): the W&B project.
            name (str): the check run's name.
            attempts (int): delete attempts, 3 seconds apart.

        Returns:
            str: the check run's URL.

        Raises:
            RuntimeError: if the run found is not the check run.
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
