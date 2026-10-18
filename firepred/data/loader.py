"""From Dataset to model input: DataLoaders and batch assembly.

Level 1 (data). Used by training/run.py, training/evaluation.py and cli/check.py.
"""

import torch
from torch.utils.data import DataLoader

from firepred.config.channels import BASE_MODEL_CHANNELS, TESSERA_CHANNELS


def make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    """Create a conservative loader that can be tuned after RAM is measured."""
    # Batches of windows; training shuffles, validation and test keep a fixed order.
    loader_options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    # Worker prefetching can hold multiple decompressed windows in RAM.
    # Keep one prefetched batch per worker rather than PyTorch's default two.
    if num_workers > 0:
        loader_options["prefetch_factor"] = 1
        loader_options["persistent_workers"] = True

    return DataLoader(**loader_options)


def validate_dataset_output(dataset, expected_model_channels, dataset_name):
    """Fail before model construction if the Dataset/model shapes disagree."""
    if len(dataset) == 0:
        raise RuntimeError(f"{dataset_name} dataset contains no windows")

    # Load one window and check its channel count matches what the model expects,
    # so a data/model mismatch fails here, before any training time is spent.
    sample = dataset[0]
    data_shape = tuple(sample["data"].shape)
    label_shape = tuple(sample["labels"].shape)
    static_channels = sample["static"].shape[0] if "static" in sample else 0

    if len(data_shape) != 4:
        raise ValueError(
            f"{dataset_name} sample must be (C,T,H,W), found {data_shape}"
        )

    actual_channels = data_shape[0] + static_channels
    if actual_channels != expected_model_channels:
        raise ValueError(
            f"{dataset_name} FireDataset produces {actual_channels} channels "
            f"after preprocessing, but the model is configured for "
            f"{expected_model_channels}. The 27-channel archive plus 17-class "
            f"land-cover expansion gives {BASE_MODEL_CHANNELS} channels; "
            f"--tessera adds {TESSERA_CHANNELS} more."
        )

    print(
        f"{dataset_name} windows={len(dataset)}, "
        f"data shape={data_shape}, label shape={label_shape}"
        + (f", static TESSERA channels={static_channels}" if static_channels else "")
    )


def batch_inputs(batch, device):
    """Model input on `device`: the 43 stored channels, then (TESSERA mode)
    the static (129, H, W) map repeated across the T days.

    The repeat happens here on the GPU, not in the loader, so host RAM holds
    the TESSERA map once per sample instead of once per day. The channel
    order matches appending TESSERA after the base channels.
    """
    # Move the batch to the GPU; in TESSERA mode, copy the per-fire map onto every
    # day there and append it after the 43 regular channels.
    data = batch["data"].to(device, non_blocking=True)
    if "static" in batch:
        static = batch["static"].to(device, non_blocking=True)
        static = static.unsqueeze(2).expand(-1, -1, data.shape[2], -1, -1)
        data = torch.cat((data, static), dim=1)
    return data
