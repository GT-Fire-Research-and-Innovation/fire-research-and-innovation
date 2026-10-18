"""Test-set scoring the paper's way: per fire, then averaged over fires.

Level 3 (training). Used by training/run.py.

Each test fire's score is the mean over its windows; the model's score is the
mean over fires, so a long fire counts as much as a short one. Optionally saves
one overlay image per window: true positives (autumn), false positives
(summer), false negatives (brg) over VIIRS I4 of the last input day.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.data import decollate_batch
from torch.utils.data import Subset

from firepred.config.data import EXCLUDED_TEST_FIRES
from firepred.data.archive import fire_name
from firepred.data.dataset import batch_inputs, make_dataloader
from firepred.models.registry import forward_model
from firepred.training.metrics import background_f1, fire_only_f1, fire_only_iou

# Channel drawn under the overlays: VIIRS I4 (channel 3).
PLOT_BACKGROUND_CHANNEL = 3


def normalize_for_plot(array):
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
    """Overlay TP/FP/FN pixels of one window on its background band."""
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


def evaluate_model(
    model,
    test_dataset,
    batch_size,
    num_workers,
    pin_memory,
    post_trans,
    device,
    test_csv_path,
    plot_dir,
    num_heads,
    hidden_size,
    model_input_channels,
    save_plots=True,
    verbose=True,
):
    """Score the test fires one at a time and return the paper's metrics.

    Returns {"f1", "iou", "background_f1", "fires"}.
    """
    # Group the test windows by fire, so each fire can be scored on its own.
    indices_by_fire = defaultdict(list)
    for dataset_idx, (archive_path, _) in enumerate(test_dataset.index):
        fire_id = fire_name(archive_path)
        if fire_id.startswith("p"):
            fire_id = fire_id[1:]
        indices_by_fire[fire_id].append(dataset_idx)

    test_ids = load_test_ids(test_csv_path, set(indices_by_fire))
    if not test_ids:
        raise RuntimeError("No test fires match the available NPZ archives")

    plot_dir = Path(plot_dir)
    if save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Running totals over fires; each fire contributes one score (its window average).
    model.eval()
    f1_all = 0.0
    background_f1_all = 0.0
    iou_all = 0.0
    evaluated_fire_count = 0

    for fire_number, fire_id in enumerate(test_ids):
        # One fire at a time: a loader over just its windows.
        fire_subset = Subset(test_dataset, indices_by_fire[fire_id])
        test_dataloader = make_dataloader(
            fire_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        fire_f1 = 0.0
        fire_background_f1 = 0.0
        fire_iou = 0.0
        window_count = 0

        for batch_number, batch in enumerate(test_dataloader):
            data_batch = batch["data"]
            labels_batch = batch["labels"]

            with torch.no_grad():
                outputs = forward_model(model, batch_inputs(batch, device))
                discrete_outputs = [
                    post_trans(item)
                    for item in decollate_batch(outputs)
                ]

            # Score each window of this fire (thresholded predictions vs. targets).
            predictions = torch.stack(discrete_outputs).cpu()
            fire_f1 += fire_only_f1(predictions, labels_batch).sum().item()
            fire_background_f1 += background_f1(predictions, labels_batch).sum().item()
            fire_iou += fire_only_iou(predictions, labels_batch).sum().item()
            window_count += data_batch.shape[0]
            if not save_plots:
                continue

            # Optional pictures: one overlay per window (correct, false alarm, missed).
            outputs_numpy = predictions.numpy()
            for sample_idx in range(data_batch.shape[0]):
                plot_name = (
                    f"id_{fire_id}_nhead_{num_heads}_"
                    f"hidden_{hidden_size}_batch_{batch_number}_"
                    f"sample_{sample_idx}_fire_{fire_number}_"
                    f"nc_{model_input_channels}.png"
                )
                save_prediction_plot(
                    background=data_batch[sample_idx, PLOT_BACKGROUND_CHANNEL, -1].cpu().numpy(),
                    prediction=outputs_numpy[sample_idx, 1],
                    label=(labels_batch[sample_idx, 1] > 0).cpu().numpy(),
                    path=plot_dir / plot_name,
                )

        if window_count == 0:
            print(f"Skipping {fire_id}: no windows")
            continue

        # This fire's score = mean over its windows; add it to the across-fire totals.
        mean_fire_f1 = fire_f1 / window_count
        mean_fire_iou = fire_iou / window_count
        f1_all += mean_fire_f1
        background_f1_all += fire_background_f1 / window_count
        iou_all += mean_fire_iou
        evaluated_fire_count += 1

        if verbose:
            print(f"ID {fire_id} IoU: {mean_fire_iou}")
            print(f"ID {fire_id} F1: {mean_fire_f1}")
            print(f"ID {fire_id} Background F1: {fire_background_f1 / window_count}")

    if evaluated_fire_count == 0:
        raise RuntimeError("No test fires were evaluated")

    # The model's score = mean over fires, so a long fire counts as much as a short one.
    model_f1 = f1_all / evaluated_fire_count
    model_iou = iou_all / evaluated_fire_count
    model_background_f1 = background_f1_all / evaluated_fire_count
    if verbose:
        print(
            f"Model F1: {model_f1}; background F1: {model_background_f1}; "
            f"model IoU: {model_iou}"
        )
    return {
        "f1": model_f1,
        "iou": model_iou,
        "background_f1": model_background_f1,
        "fires": evaluated_fire_count,
    }
