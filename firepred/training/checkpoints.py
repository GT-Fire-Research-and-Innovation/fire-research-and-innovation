"""Checkpoints: save, keep the best N, load for testing.

Level 3 (training). Used by training/trainer.py (CheckpointStore) and
training/experiment.py (load_checkpoint). Uploading a checkpoint to W&B is
the Tracker's job (training/tracker.py), so this module never touches W&B.

A checkpoint is a plain dict (no repo classes are pickled, so moving code never
breaks loading):
    epoch                 0-based (the file name uses epoch + 1)
    model_state_dict      keys start with "module." (nn.DataParallel)
    optimizer_state_dict
    loss                  validation loss
    use_tessera, model_input_channels, swin_temporal_downsample,
    tessera_stats, loss_config, lr_config[, model_options]
                          the run settings needed to rebuild identical inputs
                          (RunConfig.checkpoint_settings)
"""

import heapq
from pathlib import Path

import torch

from firepred.training import accelerator


def save_checkpoint(model, optimizer, epoch, val_loss, save_path, extra=None):
    """Write weights, optimizer state and the run settings `extra` to `save_path`; return the path.

    accelerator.save is torch.save; TPU tensors are copied to the CPU first.
    """
    accelerator.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            **(extra or {}),
        },
        save_path,
        device=next(model.parameters()).device,
    )
    return Path(save_path)


class CheckpointStore:
    """The `keep` lowest-validation-loss checkpoints of one run, on disk in `directory`.

    offer() saves the model when its validation loss is among the best so far
    and deletes the checkpoint it displaces. File names come from `template`
    (RunConfig.checkpoint_template, with a 1-based {epoch}); every file also
    holds `settings` (RunConfig.checkpoint_settings).
    """

    def __init__(self, directory, template, settings, keep=1):
        """Remember where and how to save; nothing is written yet."""
        self.directory = Path(directory)
        self.template = template
        self.settings = dict(settings)
        self.keep = keep
        self._heap = []  # (-loss, path): heap[0] is the worst checkpoint kept

    def offer(self, epoch, val_loss, model, optimizer):
        """Save this epoch if it qualifies; return the saved path, or None when it does not."""
        path = self.directory / self.template.format(epoch=epoch + 1)
        if len(self._heap) >= self.keep:
            if val_loss >= -self._heap[0][0]:
                return None
            _, displaced = heapq.heappop(self._heap)
            Path(displaced).unlink(missing_ok=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, optimizer, epoch, val_loss, path, self.settings)
        heapq.heappush(self._heap, (-val_loss, str(path)))
        return path

    def ranking(self):
        """[(validation loss, path)] of the kept checkpoints, best first."""
        return [(-loss, Path(path)) for loss, path in sorted(self._heap, reverse=True)]

    @property
    def best(self):
        """Path of the lowest-validation-loss checkpoint."""
        return self.ranking()[0][1]


def _numpy_array_globals():
    """The globals numpy needs to rebuild a pickled ndarray (TESSERA statistics).

    torch's restricted unpickler matches globals by "module.name". numpy 1.x
    pickles name numpy.core.multiarray._reconstruct, numpy 2.x name
    numpy._core.multiarray._reconstruct; registering the one real function
    under both names lets either numpy read checkpoints written by the other.
    """
    import importlib

    import numpy as np

    reconstruct = None
    for module_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            reconstruct = importlib.import_module(module_name)._reconstruct
            break
        except (ImportError, AttributeError):
            continue

    def alias(module_name):
        """numpy's _reconstruct, registered under `module_name`."""

        def _reconstruct(*args, **kwargs):
            """Rebuild a pickled ndarray (delegates to numpy)."""
            return reconstruct(*args, **kwargs)

        _reconstruct.__module__ = module_name
        return _reconstruct

    allowed = [np.ndarray, np.dtype]
    if reconstruct is not None:
        allowed += [alias("numpy._core.multiarray"), alias("numpy.core.multiarray")]
    # numpy >= 1.25 pickles a dtype through its concrete class, e.g. Float32DType.
    dtypes = getattr(np, "dtypes", None)
    if dtypes is not None:
        allowed += [getattr(dtypes, name) for name in dir(dtypes) if name.endswith("DType")]
    return allowed


def read_checkpoint(checkpoint_path, device, trusted=False):
    """torch.load without running arbitrary pickled code, unless trusted.

    1. weights_only=True: tensors and plain containers only. Every checkpoint
       without TESSERA statistics loads here.
    2. On torch >= 2.4, again with numpy's array-rebuilding functions allowed,
       for the TESSERA statistics (numpy arrays).
    3. Full unpickling (weights_only=False), which can execute code from the
       file, only when `trusted`: a checkpoint this run just wrote, or
       --trust-checkpoint. All three return the same objects.
    """
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as error:  # torch raises UnpicklingError or RuntimeError by version
        first_error = error
    # Second try (torch >= 2.4): also allow numpy's array-rebuilding functions,
    # which TESSERA checkpoints need for their normalization statistics.
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is not None:
        try:
            with safe_globals(_numpy_array_globals()):
                return torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception as error:
            first_error = error
    # Last resort, only for trusted files: full unpickling, which can run code stored in the file.
    if not trusted:
        raise RuntimeError(
            f"{checkpoint_path} holds objects that a restricted load refuses "
            f"({type(first_error).__name__}). Loading it fully can run code stored "
            "in the file, so it needs --trust-checkpoint (only for checkpoints "
            "you or your own runs created)."
        ) from first_error
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def load_checkpoint(checkpoint_path, device, config, trusted=False):
    """Load a checkpoint (see read_checkpoint) and refuse one trained with other inputs than `config`'s.

    A TESSERA mismatch, a different input width, or (swinunetr3d) a different
    temporal-downsampling setting would silently give wrong results.
    """
    checkpoint = read_checkpoint(checkpoint_path, device, trusted)
    # Checkpoints from before TESSERA support carry no flag: they are 43-channel.
    trained_with_tessera = bool(checkpoint.get("use_tessera", False))
    if trained_with_tessera != config.tessera:
        raise ValueError(
            f"{checkpoint_path} was trained "
            f"{'with' if trained_with_tessera else 'without'} TESSERA; "
            f"{'add' if trained_with_tessera else 'remove'} --tessera"
        )
    trained_channels = checkpoint.get("model_input_channels", config.input_channels)
    if trained_channels != config.input_channels:
        raise ValueError(
            f"{checkpoint_path} was trained on {trained_channels} input channels; this run has "
            f"{config.input_channels}"
        )
    # Checkpoints from before this flag used the released code's merging.
    trained_temporal_downsample = bool(checkpoint.get("swin_temporal_downsample", True))
    if config.model == "swinunetr3d" and trained_temporal_downsample != config.swin_temporal_downsample:
        raise ValueError(
            f"{checkpoint_path} was trained "
            f"{'with' if trained_temporal_downsample else 'without'} temporal "
            f"downsampling; {'add' if trained_temporal_downsample else 'remove'} "
            "--swin-temporal-downsample"
        )
    return checkpoint
