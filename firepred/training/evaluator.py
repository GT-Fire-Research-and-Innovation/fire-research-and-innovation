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
from monai.data import decollate_batch
from torch.utils.data import Subset

from firepred.config.dataset import EXCLUDED_TEST_FIRES
from firepred.data.fire_dataset import batch_inputs, make_dataloader
from firepred.models.registry import forward_model
from firepred.training.objectives import background_f1, build_post_transform, fire_only_f1, fire_only_iou


@dataclass(frozen=True)
class FireScore:
    """One test fire's scores: means over its windows."""

    fire_id: str
    f1: float
    iou: float
    background_f1: float
    windows: int


@dataclass(frozen=True)
class TestScores:
    """The model's test scores (means over fires) and each fire's own."""

    f1: float
    iou: float
    background_f1: float
    fires: int
    per_fire: tuple = ()

    def as_dict(self):
        """{"f1", "iou", "background_f1", "fires"}."""
        return {"f1": self.f1, "iou": self.iou, "background_f1": self.background_f1, "fires": self.fires}

    def summary(self):
        """One line per fire, then the model's scores."""
        lines = [f"ID {fire.fire_id} IoU: {fire.iou}\nID {fire.fire_id} F1: {fire.f1}\n"
                 f"ID {fire.fire_id} Background F1: {fire.background_f1}" for fire in self.per_fire]
        lines.append(f"Model F1: {self.f1}; background F1: {self.background_f1}; model IoU: {self.iou}")
        return "\n".join(lines)


# Channel drawn under the overlays: VIIRS I4 (channel 3).
PLOT_BACKGROUND_CHANNEL = 3


def normalize_for_plot(array):
    """Min-max scale `array` to [0, 1] (zeros when it is constant)."""
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum == minimum:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / (maximum - minimum)


def load_test_ids(test_csv_path, available_ids):
    """Use CSV order when available; otherwise test every available archive."""
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


def save_prediction_plot(background, prediction, label, path):
    """Overlay TP/FP/FN pixels of one window on its background band and save it to `path`; return path."""
    plt.imshow(
        normalize_for_plot(background),
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


class Evaluator:
    """Scores a model on one test FireDataset, fire by fire."""

    def __init__(self, config, device, dataset):
        """Group the dataset's windows by fire and fix the scoring order (CSV order)."""
        self.config = config
        self.device = device
        self.dataset = dataset
        self.post_transform = build_post_transform()
        self.windows_by_fire = dataset.windows_by_fire()
        self.fire_ids = load_test_ids(config.test_csv, set(self.windows_by_fire))
        if not self.fire_ids:
            raise RuntimeError("No test fires match the available NPZ archives")

    def loader(self, fire_id):
        """An unshuffled loader over one fire's windows, on a private random generator."""
        return make_dataloader(
            Subset(self.dataset, self.windows_by_fire[fire_id]),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def plot_name(self, fire_id, fire_number, batch_number, sample):
        """File name of one window's overlay image. [released]"""
        return (
            f"id_{fire_id}_nhead_{self.config.num_heads}_hidden_{self.config.feature_size}_"
            f"batch_{batch_number}_sample_{sample}_fire_{fire_number}_nc_{self.config.input_channels}.png"
        )

    def score_fire(self, model, fire_id, fire_number, plot_dir=None):
        """One fire's FireScore (None when it has no windows); saves overlays into plot_dir if given."""
        f1 = background = iou = 0.0
        windows = 0
        for batch_number, batch in enumerate(self.loader(fire_id)):
            data, labels = batch["data"], batch["labels"]
            with torch.no_grad():
                outputs = forward_model(model, batch_inputs(batch, self.device))
                predictions = torch.stack([self.post_transform(item) for item in decollate_batch(outputs)]).cpu()
            f1 += fire_only_f1(predictions, labels).sum().item()
            background += background_f1(predictions, labels).sum().item()
            iou += fire_only_iou(predictions, labels).sum().item()
            windows += data.shape[0]
            if plot_dir is None:
                continue
            for sample in range(data.shape[0]):
                save_prediction_plot(
                    background=data[sample, PLOT_BACKGROUND_CHANNEL, -1].cpu().numpy(),
                    prediction=predictions.numpy()[sample, 1],
                    label=(labels[sample, 1] > 0).cpu().numpy(),
                    path=Path(plot_dir) / self.plot_name(fire_id, fire_number, batch_number, sample),
                )
        if windows == 0:
            return None
        return FireScore(fire_id, f1 / windows, iou / windows, background / windows, windows)

    def score(self, model, plot_dir=None):
        """TestScores of `model` over every test fire (eval mode, no gradients)."""
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
        f1 = iou = background = 0.0
        for fire in fires:
            f1 += fire.f1
            background += fire.background_f1
            iou += fire.iou
        return TestScores(f1 / count, iou / count, background / count, count, tuple(fires))
