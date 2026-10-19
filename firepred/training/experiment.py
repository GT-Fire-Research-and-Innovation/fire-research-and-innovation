"""One experiment from start to finish: Experiment(config).run().

Level 3 (training). Called by cli/train.py.

The run is a straight line. Each step builds one object from the RunConfig
and what the steps before it produced, and hands it on; nothing later calls
back into an earlier step:

    1. the Device, then seed numpy, torch, Python's random (and XLA on a TPU)
    2. FireData: where the data is (unzipped once if asked)
    3. the Tracker (the W&B run)
    4. the model (registry), wrapped in DataParallel
    5. training only: datasets and loaders -> Trainer.fit -> best checkpoint
    6. testing: CheckpointLoader.load that (or the given) checkpoint, Evaluator.score,
       Tracker.log_final
"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from firepred.config.dataset import N_STORED_CHANNELS
from firepred.data.archive import find_fire_sources, unzip_data_root
from firepred.data.fire_dataset import FireDataset, make_dataloader, validate_dataset_output
from firepred.data.tessera_archive import compute_tessera_stats, iter_tessera_maps, load_tessera_maps
from firepred.models.registry import build_model
from firepred.training.accelerator import Device
from firepred.training.checkpoints import CheckpointLoader, CheckpointStore
from firepred.training.evaluator import Evaluator
from firepred.training.tracker import Tracker
from firepred.training.trainer import Trainer


@dataclass(frozen=True)
class TrainingData:
    """Hold what Trainer.fit reads: the two loaders and the TESSERA statistics.

    The loaders are built by FireData.training. The statistics are stored
    with every checkpoint, so a later test run can normalize TESSERA exactly
    as training did.

    Attributes:
        train_loader (torch.utils.data.DataLoader): augmented, shuffled training windows.
        val_loader (torch.utils.data.DataLoader): validation windows in a fixed order.
        tessera_stats (dict | None): the training TESSERA statistics.
    """

    train_loader: object
    val_loader: object
    tessera_stats: dict = None


@dataclass(frozen=True)
class FireData:
    """Know where one run's window data is, and make its datasets and loaders.

    Every dataset of the run is built the same way from here. The training
    TESSERA statistics are computed from the training fires only.

    Attributes:
        config (RunConfig): the run's settings.
        root (Path): the data root actually read (possibly the unzipped copy).
        tessera_dir (Path | None): the TESSERA folder.
    """

    config: object
    root: Path
    tessera_dir: Path = None

    @classmethod
    def locate(cls, config):
        """Find the run's data, unzipping the archives once when asked.

        With --unzip-dir the archives are unpacked into folders, which are faster to
        read and hold identical arrays. The TESSERA folder stays where it is.

        Args:
            config (RunConfig): the run's settings.

        Returns:
            FireData: where the data is.
        """
        root = Path(config.data_root).expanduser()
        tessera_dir = config.tessera_root
        if config.unzip_dir != "off":
            resolved = root.resolve()
            unzip_root = (
                resolved.with_name(resolved.name + "-npy") if config.unzip_dir == "auto"
                else Path(config.unzip_dir).expanduser()
            )
            splits = ("test",) if config.test_only else ("train", "val", "test")
            root = unzip_data_root(root, unzip_root, splits=splits)
        return cls(config, Path(root), tessera_dir)

    def has_test(self):
        """Say whether the test split holds any fire.

        Without test fires, periodic and final testing are skipped. The training
        runs anyway.

        Returns:
            bool: True when there are test archives.
        """
        return bool(find_fire_sources(self.root / "test"))

    def dataset(self, split, augment, tessera_stats=None, tessera_maps=None):
        """Build the FireDataset of one split with the run's settings.

        Every split is built with the same settings. TESSERA statistics and
        maps are passed only in TESSERA mode.

        Args:
            split (str): "train", "val" or "test".
            augment (bool): random flips and rotation.
            tessera_stats (dict | None): the training TESSERA statistics.
            tessera_maps (dict | None): the split's TESSERA maps, when already read.

        Returns:
            FireDataset: the dataset.
        """
        return FireDataset(
            split_dir=self.root / split,
            ts_length=self.config.ts_length,
            stored_channels=N_STORED_CHANNELS,
            target_is_single_day=True,
            augment=augment,
            tessera_dir=self.tessera_dir,
            tessera_stats=tessera_stats,
            tessera_maps=tessera_maps,
            cells_per_pixel=self.config.tessera_cells,
        )

    def loader(self, dataset, shuffle, seed_offset):
        """Make a loader for a dataset with its own random generator.

        The generator is seeded from the run's seed plus an offset, so each loader
        has a separate, repeatable random stream. Pinned memory is off to save host
        RAM.

        Args:
            dataset (FireDataset): the windows.
            shuffle (bool): reshuffle every epoch.
            seed_offset (int): added to the run's seed for this loader.

        Returns:
            torch.utils.data.DataLoader: the loader.
        """
        return make_dataloader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            # No pinned host copies: with TESSERA a batch of 8 is about 0.8 GiB,
            # and pinning keeps a second copy of each in host RAM.
            pin_memory=False,
            generator=torch.Generator().manual_seed(self.config.seed + seed_offset),
        )

    def training_tessera(self):
        """Read the training fires' TESSERA and compute its statistics.

        The statistics come from the training fires only, so nothing leaks from
        held-out fires. With one cell per pixel the maps are read once and kept
        for the training dataset; with k > 1 cells they are streamed one fire at
        a time and not kept (the dataset reads them per window).

        Returns:
            tuple[dict | None, dict | None]: (statistics, kept maps); (None, None)
            without TESSERA.
        """
        if self.tessera_dir is None:
            return None, None
        cells = self.config.tessera_cells
        train_fires = find_fire_sources(self.root / "train")
        maps = None
        if cells > 1:
            maps_for_stats = iter_tessera_maps(self.tessera_dir, train_fires, cells)
        else:
            maps = load_tessera_maps(self.tessera_dir, train_fires, cells)
            maps_for_stats = maps.values()
        return compute_tessera_stats(maps_for_stats, min_coverage=self.config.tessera_min_coverage), maps

    def training(self):
        """Build the training and validation data.

        TESSERA statistics come from the training fires only, so nothing leaks
        from held-out fires (training_tessera). The validation set is
        shape-checked before training.

        Returns:
            TrainingData: the two loaders and the TESSERA statistics.
        """
        stats, maps = self.training_tessera()
        train = self.dataset("train", augment=True, tessera_stats=stats, tessera_maps=maps)
        val = self.dataset("val", augment=False, tessera_stats=stats)
        # Validation for the shape check, so no random augmentation is drawn.
        validate_dataset_output(val, self.config.input_channels, "validation")
        return TrainingData(self.loader(train, shuffle=True, seed_offset=0),
                            self.loader(val, shuffle=False, seed_offset=1), stats)

    def test_dataset(self, tessera_stats=None):
        """Build the test dataset and check it fits the model.

        Test windows are never augmented. In TESSERA mode the given statistics
        (normally the checkpoint's) normalize them.

        Args:
            tessera_stats (dict | None): the TESSERA statistics to use.

        Returns:
            FireDataset: the test windows.
        """
        dataset = self.dataset("test", augment=False, tessera_stats=tessera_stats)
        validate_dataset_output(dataset, self.config.input_channels, "test")
        return dataset


@dataclass(frozen=True)
class RunResult:
    """Hold what a run produced.

    The checkpoint is the one that was tested (the best one, after training).
    The scores are missing when no test ran.

    Attributes:
        checkpoint (Path): the tested or best checkpoint.
        scores (TestScores | None): the final test scores.
    """

    checkpoint: Path
    scores: object = None


class Experiment:
    """Run one training or test run of a RunConfig, start to finish, in a straight line.

    Each step builds one object from the settings and the steps before it and
    hands it on: data, device, W&B tracker, model, then training and testing.
    Nothing later calls back into an earlier step.
    """

    def __init__(self, config):
        """Remember the settings; run() does the work.

        Nothing is built yet. The experiment can be inspected before it runs.

        Args:
            config (RunConfig): the run's settings.
        """
        self.config = config

    def seed(self, device):
        """Seed every random number generator the run uses.

        numpy, torch (CPU and every GPU), Python's random and, on a TPU, XLA's
        generator are seeded with the run's seed. Everything random afterwards
        (weights, shuffling, augmentation) is then repeatable.

        Args:
            device (Device): the training device.

        Returns:
            int: the seed.
        """
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return device.seed(seed)

    def prepare_model(self, device, spec=None):
        """Build the run's registered model, wrap it in DataParallel and move it to the device.

        DataParallel splits each batch across the visible GPUs. The wrapping adds
        "module." to every weight name in checkpoints. cli.check passes its own
        spec to try another window size.

        Args:
            device (Device | torch.device): where it runs.
            spec (ModelSpec | None): what the model is built for (default: config.model_spec()).

        Returns:
            torch.nn.DataParallel: the wrapped model.
        """
        model = nn.DataParallel(build_model(self.config.model, spec or self.config.model_spec()))
        model.to(Device.of(device).torch_device)
        return model

    def train(self, data, model, device, tracker):
        """Train the model and return its best checkpoint.

        The data, an optional evaluator for periodic test scores, and the
        checkpoint store are built here and handed to Trainer.fit. The kept
        checkpoints are printed afterwards.

        Args:
            data (FireData): where the data is.
            model (torch.nn.Module): the model.
            device (Device): where it runs.
            tracker (Tracker): the W&B run.

        Returns:
            Path: the best checkpoint.
        """
        config = self.config
        training = data.training()
        # [ours] optional test scoring every --test-every epochs with the current weights (logging only).
        evaluator = None
        if config.test_every > 0:
            if data.has_test():
                evaluator = Evaluator(config, device, data.test_dataset(training.tessera_stats))
            else:
                print(f"No test archives in {data.root / 'test'}; --test-every is ignored")
        store = CheckpointStore(tracker.checkpoint_dir, config.checkpoint_template,
                                config.checkpoint_settings(training.tessera_stats), keep=config.keep_checkpoints)
        best = Trainer(config, model, device).fit(training, tracker, store, evaluator)
        for loss, path in store.ranking():
            print(f"Best checkpoint: loss={loss:.6f}: {path}")
        return best

    def test(self, data, model, device, tracker, checkpoint_path):
        """Load a checkpoint into the model, score the test fires and log the scores.

        The checkpoint must match this run's inputs. Its TESSERA statistics
        normalize the test data exactly as in training.

        Args:
            data (FireData): where the data is.
            model (torch.nn.Module): the model.
            device (Device): where it runs.
            tracker (Tracker): the W&B run.
            checkpoint_path (str | Path): the checkpoint to test.

        Returns:
            TestScores: the scores.
        """
        config = self.config
        checkpoint = CheckpointLoader(device, config).load(
            checkpoint_path,
            # The checkpoint this run just saved is trusted; a given one only on request.
            trusted=config.trust_checkpoint or not config.test_only,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint epoch={checkpoint['epoch']} loss={checkpoint['loss']}")
        scores = Evaluator(config, device, data.test_dataset(checkpoint.get("tessera_stats"))).score(
            model, config.plot_dir
        )
        print(scores.summary())
        tracker.log_final(scores)
        return scores

    def run(self):
        """Run the experiment: seed, find the data, start W&B, build the model, train, test.

        With -test only the testing runs, and no loss or optimizer is built. The
        W&B run is finished when this returns, even after an error.

        Returns:
            RunResult: the tested checkpoint and the final scores.
        """
        config = self.config
        device = Device.resolve(config.device)
        self.seed(device)
        data = FireData.locate(config)
        with Tracker(config, device) as tracker:
            model = self.prepare_model(device)
            if config.test_only:
                checkpoint = config.checkpoint or config.test_checkpoint
            else:
                checkpoint = self.train(data, model, device, tracker)
                if not config.test_after_train:
                    return RunResult(checkpoint)
                if not data.has_test():
                    print(f"No test archives in {data.root / 'test'}; skipping the test step")
                    return RunResult(checkpoint)
            return RunResult(checkpoint, self.test(data, model, device, tracker, checkpoint))
