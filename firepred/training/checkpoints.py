"""Checkpoints: save, upload to W&B, keep the best N, load for testing.

Level 3 (training). Used by training/loop.py and training/run.py.

A checkpoint is a plain dict (no repo classes are pickled, so moving code never
breaks loading):
    epoch                 0-based (the file name uses epoch + 1)
    model_state_dict      keys start with "module." (nn.DataParallel)
    optimizer_state_dict
    loss                  validation loss
    use_tessera, model_input_channels, swin_temporal_downsample,
    tessera_stats, loss_config, lr_config      the run settings needed to
                                               rebuild identical inputs
"""

import heapq
import os
from pathlib import Path

import torch
import wandb

from firepred.training import accelerator


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    save_path,
    extra=None,
    artifact_name=None,
):
    # Weights, optimizer state and the settings needed to rebuild identical inputs.
    # (accelerator.save is torch.save; TPU tensors are copied to the CPU first.)
    accelerator.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            # Run config (e.g. TESSERA mode and its normalization) needed to
            # rebuild identical inputs at test time.
            **(extra or {}),
        },
        save_path,
        device=next(model.parameters()).device,
    )

    if artifact_name is not None:
        upload_checkpoint(save_path, epoch, val_loss, artifact_name, extra)


def upload_checkpoint(save_path, epoch, val_loss, artifact_name, extra=None):
    """Store a versioned checkpoint as a W&B model Artifact.

    Online, wait for the upload so a later best checkpoint cannot replace the
    file first; offline runs log it locally for a later `wandb sync`.
    """
    if wandb.run is None:
        raise RuntimeError("A W&B run is required to upload model checkpoints")
    # Upload the file as a new version of the run's model artifact, tagged best/latest/epoch-N.
    metadata = {"epoch": epoch + 1, "validation_loss": float(val_loss)}
    for key in ("use_tessera", "model_input_channels"):
        if extra and key in extra:
            metadata[key] = extra[key]
    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    artifact.add_file(str(save_path), name=Path(save_path).name)
    uploaded = wandb.run.log_artifact(
        artifact, aliases=["best", "latest", f"epoch-{epoch + 1}"]
    )
    if not wandb.run.offline:
        uploaded.wait()
    print(f"Checkpoint logged to W&B Artifacts: {artifact_name}")


class BestCheckpoints:
    """Keep the `top_n` lowest-validation-loss checkpoints on disk.

    offer() saves the current model when it is among the best so far and
    deletes the checkpoint it displaces. The heap stores (-loss, path), so
    heap[0] is the worst checkpoint kept.
    """

    def __init__(self, top_n):
        self.top_n = top_n
        self.heap = []

    def offer(self, val_loss, checkpoint_path, save):
        """Call `save()` (which writes checkpoint_path) if val_loss qualifies."""
        if len(self.heap) < self.top_n:
            save()
            heapq.heappush(
                self.heap,
                (-val_loss, checkpoint_path),
            )
        else:
            worst_loss = -self.heap[0][0]
            if val_loss < worst_loss:
                _, removed_path = heapq.heapreplace(
                    self.heap,
                    (-val_loss, checkpoint_path),
                )
                if os.path.exists(removed_path):
                    os.remove(removed_path)
                save()

    def print_summary(self):
        print("Best checkpoints:")
        for negative_loss, checkpoint_path in sorted(
            self.heap,
            reverse=True,
        ):
            print(f"loss={-negative_loss:.6f}: {checkpoint_path}")

    def best_path(self):
        """Path of the lowest-validation-loss checkpoint."""
        # Heap entries are (-loss, path), so the largest is the lowest loss.
        return max(self.heap)[1]


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
        def _reconstruct(*args, **kwargs):
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


def load_checkpoint(checkpoint_path, device, model_name, use_tessera, swin_temporal_downsample, trusted=False):
    """Load a checkpoint (see read_checkpoint) and refuse one trained with different inputs."""
    checkpoint = read_checkpoint(checkpoint_path, device, trusted)
    # Checkpoints from before TESSERA support carry no flag: they are 43-channel.
    # Refuse a checkpoint whose inputs differ from this run's: a TESSERA mismatch
    # or a different temporal-downsampling setting would silently give wrong results.
    trained_with_tessera = bool(checkpoint.get("use_tessera", False))
    if trained_with_tessera != use_tessera:
        raise ValueError(
            f"{checkpoint_path} was trained "
            f"{'with' if trained_with_tessera else 'without'} TESSERA; "
            f"{'add' if trained_with_tessera else 'remove'} --tessera"
        )
    # Checkpoints from before this flag used the released code's merging.
    trained_temporal_downsample = bool(checkpoint.get("swin_temporal_downsample", True))
    if model_name == "swinunetr3d" and trained_temporal_downsample != swin_temporal_downsample:
        raise ValueError(
            f"{checkpoint_path} was trained "
            f"{'with' if trained_temporal_downsample else 'without'} temporal "
            f"downsampling; {'add' if trained_temporal_downsample else 'remove'} "
            "--swin-temporal-downsample"
        )
    return checkpoint
