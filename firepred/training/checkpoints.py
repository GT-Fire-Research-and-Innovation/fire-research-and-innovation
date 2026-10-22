"""Checkpoints: save, keep the best N, load for testing.

Level 3 (training). CheckpointStore saves during training (training/trainer.py);
find_test_checkpoint finds one for -test and CheckpointLoader reads it back
(training/experiment.py). Uploading a checkpoint to W&B is
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

import glob
import heapq
import os
from pathlib import Path

import torch

from firepred.training.accelerator import Device


class CheckpointStore:
    """Keep the best few checkpoints of a run on disk, by validation loss.

    Every epoch offers a checkpoint. It is saved when its validation loss is
    among the lowest so far, and the checkpoint it pushes out is deleted, so at
    most `keep` files exist at any time.
    """

    def __init__(self, directory, template, settings, keep=1):
        """Remember where and how to save; nothing is written yet.

        File names come from the template with a 1-based epoch. Every file also
        holds the run settings.

        Args:
            directory (str | Path): where checkpoints go (the W&B run folder).
            template (str): RunConfig.checkpoint_template, with an {epoch} placeholder.
            settings (dict): RunConfig.checkpoint_settings, stored in every file.
            keep (int): how many checkpoints to keep.
        """
        self.directory = Path(directory)
        self.template = template
        self.settings = dict(settings)
        self.keep = keep
        self._heap = []  # (-loss, path): heap[0] is the worst checkpoint kept

    def offer(self, epoch, val_loss, model, optimizer):
        """Save this epoch's checkpoint if its validation loss is among the best.

        When the store is full, the worst kept checkpoint is deleted first. A
        checkpoint that is not better than the worst kept one is not saved.

        Args:
            epoch (int): the 0-based epoch.
            val_loss (float): this epoch's validation loss.
            model (torch.nn.Module): the model.
            optimizer (torch.optim.Optimizer): the optimizer.

        Returns:
            Path | None: the saved file, or None when the epoch was not good enough.
        """
        path = self.directory / self.template.format(epoch=epoch + 1)
        if len(self._heap) >= self.keep:
            if val_loss >= -self._heap[0][0]:
                return None
            _, displaced = heapq.heappop(self._heap)
            Path(displaced).unlink(missing_ok=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.save(path, epoch, val_loss, model, optimizer)
        heapq.heappush(self._heap, (-val_loss, str(path)))
        return path

    def save(self, path, epoch, val_loss, model, optimizer):
        """Write one checkpoint file: the weights, the optimizer state and the run settings.

        The file is a plain dict, so loading never needs repo classes. The settings
        (TESSERA statistics, loss, schedule) let a later test run rebuild exactly
        the same inputs. offer calls this; it does not decide what is kept.

        Args:
            path (str | Path): the file to write.
            epoch (int): the 0-based epoch.
            val_loss (float): this epoch's validation loss.
            model (torch.nn.Module): the model.
            optimizer (torch.optim.Optimizer): the optimizer.

        Returns:
            Path: path.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            **self.settings,
        }
        return Device.of(next(model.parameters()).device).save(checkpoint, path)

    def ranking(self):
        """List the kept checkpoints from best to worst.

        Best means the lowest validation loss. The experiment prints this list
        after training.

        Returns:
            list[tuple[float, Path]]: (validation loss, file) pairs.
        """
        return [(-loss, Path(path)) for loss, path in sorted(self._heap, reverse=True)]

    @property
    def best(self):
        """Give the checkpoint with the lowest validation loss.

        This is the checkpoint the run tests at the end. At least one epoch must
        have been offered.

        Returns:
            Path: the best checkpoint file.
        """
        return self.ranking()[0][1]


def find_test_checkpoint(pattern, root="."):
    """Find the checkpoint -test uses when no --checkpoint is given: the newest match.

    Training writes its checkpoints into its W&B run folder,
    <WANDB_DIR or root>/wandb/<run>/files/, so every run folder there is searched,
    plus saved_models/ where the released code looked [released]. Of the files
    whose name matches the pattern, the most recently written one is used.

    Args:
        pattern (str): the file name to match (RunConfig.test_checkpoint_pattern).
        root (str | Path): the folder the runs were started from.

    Returns:
        Path: the newest matching checkpoint.

    Raises:
        FileNotFoundError: if no file matches; the message names the places searched.
    """
    root = Path(root)
    places = [Path(os.environ.get("WANDB_DIR", root)) / "wandb" / "*" / "files", root / "saved_models"]
    # wandb/latest-run links to a run folder; resolving counts each file once.
    found = {Path(path).resolve() for place in places for path in glob.glob(str(place / pattern))}
    if not found:
        searched = ", ".join(str(place / pattern) for place in places)
        raise FileNotFoundError(f"No checkpoint for -test in {searched}; pass --checkpoint")
    return max(found, key=lambda path: path.stat().st_mtime)


class CheckpointLoader:
    """Read checkpoints back safely, and refuse one trained with other inputs than this run.

    A pickled file can contain code that runs when it is loaded, so reading is
    restricted unless the file is trusted. Tensors are placed where the device
    wants them (the CPU for a TPU run).

    Attributes:
        device (Device): where the tensors go.
        config (RunConfig | None): this run's settings, needed by load.
    """

    def __init__(self, device, config=None):
        """Remember where tensors go and which settings a checkpoint must match.

        Nothing is read yet. read needs only the device; load also needs the
        config.

        Args:
            device (Device | torch.device | str): where to place the tensors.
            config (RunConfig | None): this run's settings.
        """
        self.device = Device.of(device)
        self.config = config

    @staticmethod
    def numpy_array_globals():
        """List the numpy functions a restricted checkpoint load may use to rebuild arrays.

        TESSERA checkpoints store numpy arrays, which torch's safe loader refuses
        by default. numpy 1.x and 2.x name the rebuild function differently, so it
        is registered under both names and either numpy can read the other's files.

        Returns:
            list: the allowed numpy classes and functions.
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
            """Make numpy's array-rebuild function under a given module name.

            The restricted loader matches functions by their module and name. This
            gives the one real function the name a checkpoint expects.

            Args:
                module_name (str): "numpy._core.multiarray" or "numpy.core.multiarray".

            Returns:
                Callable: the renamed rebuild function.
            """

            def _reconstruct(*args, **kwargs):
                """Rebuild a pickled numpy array by calling numpy's own function.

                This wrapper exists only to carry another module name. It passes every
                argument through unchanged.

                Args:
                    *args: the pickled array's rebuild arguments.
                    **kwargs: the pickled array's rebuild keyword arguments.

                Returns:
                    np.ndarray: the rebuilt array.
                """
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

    def read(self, checkpoint_path, trusted=False):
        """Load a checkpoint without running any code stored in it, unless it is trusted.

        A pickled file can contain code that runs when it is loaded. The first try
        allows only tensors and plain containers; the second (torch 2.4 or newer)
        also allows numpy arrays for the TESSERA statistics. Full loading happens
        only for a trusted file: a checkpoint this run wrote, or --trust-checkpoint.

        Args:
            checkpoint_path (str | Path): the checkpoint file.
            trusted (bool): allow full unpickling as a last resort.

        Returns:
            dict: the checkpoint.

        Raises:
            RuntimeError: if a restricted load fails and the file is not trusted.
        """
        try:
            return torch.load(checkpoint_path, map_location=self.device.map_location, weights_only=True)
        except Exception as error:  # torch raises UnpicklingError or RuntimeError by version
            first_error = error
        # Second try (torch >= 2.4): also allow numpy's array-rebuilding functions,
        # which TESSERA checkpoints need for their normalization statistics.
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is not None:
            try:
                with safe_globals(self.numpy_array_globals()):
                    return torch.load(checkpoint_path, map_location=self.device.map_location, weights_only=True)
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
        return torch.load(checkpoint_path, map_location=self.device.map_location, weights_only=False)

    def load(self, checkpoint_path, trusted=False):
        """Load a checkpoint and refuse it if it was trained with other inputs than this run.

        A TESSERA mismatch (on or off, or another number of cells per pixel), a
        different input width or (SwinUNETR-3D and its variants) a different
        temporal-merging setting would silently give wrong results. Loading
        follows read's safety rules.

        Args:
            checkpoint_path (str | Path): the checkpoint file.
            trusted (bool): allow full unpickling as a last resort.

        Returns:
            dict: the checkpoint.

        Raises:
            ValueError: if the checkpoint's inputs differ from this run's, or the loader
                has no config to compare with.
            RuntimeError: if a restricted load fails and the file is not trusted.
        """
        config = self.config
        if config is None:
            raise ValueError("CheckpointLoader.load compares a checkpoint with this run's settings; "
                             "create the loader with a config, or use read() for an unchecked load")
        checkpoint = self.read(checkpoint_path, trusted)
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
        trained_cells = checkpoint.get("tessera_cells_per_pixel", 1)  # older checkpoints: one cell per pixel
        if config.tessera and trained_cells != config.tessera_cells:
            raise ValueError(
                f"{checkpoint_path} was trained with {trained_cells} TESSERA cells per pixel; "
                f"use --tessera-cells-per-pixel {trained_cells}"
            )
        # Checkpoints from before this flag used the released code's merging.
        trained_temporal_downsample = bool(checkpoint.get("swin_temporal_downsample", True))
        if config.is_swinunetr3d and trained_temporal_downsample != config.swin_temporal_downsample:
            raise ValueError(
                f"{checkpoint_path} was trained "
                f"{'with' if trained_temporal_downsample else 'without'} temporal "
                f"downsampling; {'add' if trained_temporal_downsample else 'remove'} "
                "--swin-temporal-downsample"
            )
        return checkpoint
