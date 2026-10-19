"""Test-set scoring the paper's way: per fire, then averaged over fires.

Level 3 (training). Used by training/experiment.py (the final test) and
training/trainer.py (periodic test scoring, logging only).

Each test fire's score is the mean over its windows; the model's score is the
mean over fires, so a long fire counts as much as a short one. Optionally saves
one overlay image per window: true positives (autumn), false positives
(summer), false negatives (brg) over VIIRS I4 of the last input day.

Scoring never disturbs training: the model is put in eval mode, nothing is
augmented, and the loaders draw from their own random generator, so the
global random streams the training loop uses are left untouched.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from firepred.config.dataset import EXCLUDED_TEST_FIRES
from firepred.data.fire_dataset import batch_inputs, make_dataloader
from firepred.models.registry import forward_model
from firepred.training.accelerator import Device
from firepred.training.objectives import Scorer


@dataclass(frozen=True)
class FireScore:
    """Hold one test fire's scores, each a mean over its windows.

    The paper scores every fire on its own and then averages the fires. These
    are the per-fire numbers.

    Attributes:
        fire_id (str): the fire id.
        f1 (float): fire-class F1.
        iou (float): fire-class IoU (the paper's Equation 2).
        background_f1 (float): background-class F1.
        windows (int): the fire's windows.
        false_positives (float): false-alarm pixels per window. [ours]
        false_negatives (float): missed fire pixels per window. [ours]
    """

    fire_id: str
    f1: float
    iou: float
    background_f1: float
    windows: int
    false_positives: float = 0.0
    false_negatives: float = 0.0


@dataclass(frozen=True)
class TestScores:
    """Hold a model's test scores: the means over fires, plus each fire's own.

    A long fire counts as much as a short one, because every fire contributes
    one score. These are the numbers the paper reports.

    Attributes:
        f1 (float): mean fire-class F1 over fires.
        iou (float): mean fire-class IoU over fires.
        background_f1 (float): mean background F1 over fires.
        fires (int): fires scored.
        per_fire (tuple[FireScore, ...]): each fire's scores.
        false_positives (float): mean over fires of false-alarm pixels per window. [ours]
        false_negatives (float): mean over fires of missed fire pixels per window. [ours]
    """

    f1: float
    iou: float
    background_f1: float
    fires: int
    per_fire: tuple = ()
    false_positives: float = 0.0
    false_negatives: float = 0.0

    def as_dict(self):
        """Give the four headline scores as a plain dict.

        The golden tests and older code compare these keys. Per-fire scores and
        the false-positive and false-negative counts are left out.

        Returns:
            dict: {"f1", "iou", "background_f1", "fires"}.
        """
        return {"f1": self.f1, "iou": self.iou, "background_f1": self.background_f1, "fires": self.fires}

    def summary(self):
        """Describe the scores: one block per fire, then the model's means.

        The experiment prints this after the final test. The format follows the
        released code's printout.

        Returns:
            str: the summary, one score per line.
        """
        lines = [f"ID {fire.fire_id} IoU: {fire.iou}\nID {fire.fire_id} F1: {fire.f1}\n"
                 f"ID {fire.fire_id} Background F1: {fire.background_f1}" for fire in self.per_fire]
        lines.append(f"Model F1: {self.f1}; background F1: {self.background_f1}; model IoU: {self.iou}")
        return "\n".join(lines)


# Channel drawn under the overlays: VIIRS I4 (channel 3).
PLOT_BACKGROUND_CHANNEL = 3


class Evaluator:
    """Score a model on the test fires, one fire at a time.

    Each fire's score is the mean over its windows, and the model's score is
    the mean over fires, as in the paper. Scoring runs in eval mode with its
    own random generator, so it never changes training.
    """

    def __init__(self, config, device, dataset):
        """Group the test windows by fire and fix the scoring order.

        The order is the test CSV's. The fires must have archives.

        Args:
            config (RunConfig): batch size, workers, test CSV, names.
            device (Device | torch.device): where the model runs.
            dataset (FireDataset): the test windows.

        Raises:
            RuntimeError: if no test fire has an archive.
        """
        self.config = config
        self.device = Device.of(device)
        self.dataset = dataset
        self.scorer = Scorer()
        self.windows_by_fire = dataset.windows_by_fire()
        self.fire_ids = self.load_test_ids(config.test_csv, set(self.windows_by_fire))
        if not self.fire_ids:
            raise RuntimeError("No test fires match the available NPZ archives")

    @staticmethod
    def normalize_for_plot(array):
        """Scale an image band to the range 0..1 for plotting.

        The overlay draws the band in gray. A constant band becomes all zeros.

        Args:
            array (np.ndarray): the band.

        Returns:
            np.ndarray: the scaled band.
        """
        minimum = float(array.min())
        maximum = float(array.max())
        if maximum == minimum:
            return np.zeros_like(array, dtype=np.float32)
        return (array - minimum) / (maximum - minimum)


    @staticmethod
    def load_test_ids(test_csv_path, available_ids):
        """List the test fires to score, in the test CSV's order.

        Only fires that have an archive are kept, and the excluded test fire is
        left out. Without the CSV every available fire is scored, sorted.

        Args:
            test_csv_path (str | Path): the test CSV.
            available_ids (set[str]): fire ids that have archives.

        Returns:
            list[str]: the fire ids to score.
        """
        test_csv_path = Path(test_csv_path).expanduser()
        if not test_csv_path.exists():
            print(
                f"Test CSV not found at {test_csv_path}; using NPZ filenames"
            )
            return sorted(available_ids)

        # Test fires in the CSV's order, minus the excluded ones, keeping only fires
        # that have an archive.
        dataframe = pd.read_csv(test_csv_path, dtype={"Id": str})
        ids = dataframe["Id"].astype(str)
        ids = ids[~ids.isin(EXCLUDED_TEST_FIRES)]
        return [fire_id for fire_id in ids if fire_id in available_ids]


    @staticmethod
    def save_prediction_plot(background, prediction, label, path):
        """Save an overlay picture of one window's prediction.

        The background band is drawn in gray; correct fire pixels, false alarms and
        missed pixels are drawn in three colours on top. One picture per test
        window helps see where the model is wrong.

        Args:
            background (np.ndarray): (H, W) band to draw underneath.
            prediction (np.ndarray): (H, W) predicted fire mask.
            label (np.ndarray): (H, W) true fire mask.
            path (str | Path): the image file.

        Returns:
            str | Path: path.
        """
        plt.imshow(
            Evaluator.normalize_for_plot(background),
            cmap="gray",
        )

        true_positive = np.where(
            (prediction == 1) & (label == 1),
            1.0,
            np.nan,
        )
        false_positive = np.where(
            (prediction == 1) & (label == 0),
            1.0,
            np.nan,
        )
        false_negative = np.where(
            (prediction == 0) & (label == 1),
            1.0,
            np.nan,
        )

        plt.imshow(
            true_positive,
            cmap="autumn",
            interpolation="nearest",
        )
        plt.imshow(
            false_positive,
            cmap="summer",
            interpolation="nearest",
        )
        plt.imshow(
            false_negative,
            cmap="brg",
            interpolation="nearest",
        )
        plt.axis("off")
        plt.savefig(
            path,
            bbox_inches="tight",
        )
        plt.close()
        return path

    def loader(self, fire_id):
        """Make a loader over one fire's windows.

        The windows stay in order. The loader has its own random generator, so it
        never draws from the training run's random numbers.

        Args:
            fire_id (str): the fire.

        Returns:
            torch.utils.data.DataLoader: the loader.
        """
        return make_dataloader(
            Subset(self.dataset, self.windows_by_fire[fire_id]),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def plot_name(self, fire_id, fire_number, batch_number, sample):
        """Name one window's overlay picture.

        The name records the fire, the model settings and the window's position
        [released]. It is unique within a run.

        Args:
            fire_id (str): the fire.
            fire_number (int): the fire's position in the scoring order.
            batch_number (int): the batch within the fire.
            sample (int): the window within the batch.

        Returns:
            str: the file name.
        """
        return (
            f"id_{fire_id}_nhead_{self.config.num_heads}_hidden_{self.config.feature_size}_"
            f"batch_{batch_number}_sample_{sample}_fire_{fire_number}_nc_{self.config.input_channels}.png"
        )

    def score_fire(self, model, fire_id, fire_number, plot_dir=None):
        """Score one fire and optionally save its overlay pictures.

        Predictions are thresholded and compared with the targets window by
        window. The fire's scores are the means over its windows.

        Args:
            model (torch.nn.Module): the model, in eval mode.
            fire_id (str): the fire.
            fire_number (int): its position in the scoring order.
            plot_dir (str | Path | None): where to save pictures, or None.

        Returns:
            FireScore | None: the fire's scores, or None when it has no windows.
        """
        f1 = background = iou = false_positives = false_negatives = 0.0
        windows = 0
        for batch_number, batch in enumerate(self.loader(fire_id)):
            data, labels = batch["data"], batch["labels"]
            with torch.no_grad():
                outputs = forward_model(model, batch_inputs(batch, self.device.torch_device))
                predictions = self.scorer.predictions(outputs).cpu()
            f1 += self.scorer.fire_f1(predictions, labels).sum().item()
            background += self.scorer.background_f1(predictions, labels).sum().item()
            iou += self.scorer.fire_iou(predictions, labels).sum().item()
            false_positives += self.scorer.fire_false_positives(predictions, labels).sum().item()
            false_negatives += self.scorer.fire_false_negatives(predictions, labels).sum().item()
            windows += data.shape[0]
            if plot_dir is None:
                continue
            for sample in range(data.shape[0]):
                self.save_prediction_plot(
                    background=data[sample, PLOT_BACKGROUND_CHANNEL, -1].cpu().numpy(),
                    prediction=predictions.numpy()[sample, 1],
                    label=(labels[sample, 1] > 0).cpu().numpy(),
                    path=Path(plot_dir) / self.plot_name(fire_id, fire_number, batch_number, sample),
                )
        if windows == 0:
            return None
        return FireScore(fire_id, f1 / windows, iou / windows, background / windows, windows,
                         false_positives / windows, false_negatives / windows)

    def score(self, model, plot_dir=None):
        """Score the model on every test fire and average over fires.

        The model is put in eval mode and no gradients are computed. Fires without
        windows are skipped.

        Args:
            model (torch.nn.Module): the model.
            plot_dir (str | Path | None): where to save overlay pictures, or None.

        Returns:
            TestScores: the means over fires and each fire's scores.

        Raises:
            RuntimeError: if no fire could be scored.
        """
        if plot_dir is not None:
            Path(plot_dir).mkdir(parents=True, exist_ok=True)
        model.eval()
        fires = []
        for fire_number, fire_id in enumerate(self.fire_ids):
            fire = self.score_fire(model, fire_id, fire_number, plot_dir)
            if fire is None:
                print(f"Skipping {fire_id}: no windows")
                continue
            fires.append(fire)
        if not fires:
            raise RuntimeError("No test fires were evaluated")
        count = len(fires)
        # Accumulate in fire order, as sums of fire means, then divide once.
        f1 = iou = background = false_positives = false_negatives = 0.0
        for fire in fires:
            f1 += fire.f1
            background += fire.background_f1
            iou += fire.iou
            false_positives += fire.false_positives
            false_negatives += fire.false_negatives
        return TestScores(f1 / count, iou / count, background / count, count, tuple(fires),
                          false_positives / count, false_negatives / count)
