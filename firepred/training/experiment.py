"""One experiment from start to finish: Experiment(config).run().

Level 3 (training). Called by cli/train.py.

The run is a straight line. Each step builds one object from the RunConfig
and what the steps before it produced, and hands it on; nothing later calls
back into an earlier step:

    1. seed numpy, torch and Python's random
    2. FireData: where the data is (unzipped once if asked)
    3. the device, then the Tracker (the W&B run)
    4. the model (registry), wrapped in DataParallel
    5. training only: datasets and loaders -> Trainer.fit -> best checkpoint
    6. testing: load that (or the given) checkpoint, Evaluator.score, Tracker.log_final
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
from firepred.data.tessera_archive import compute_tessera_stats, load_tessera_maps
from firepred.models.registry import build_model
from firepred.training.accelerator import checkpoint_map_location, resolve_device, seed_device
from firepred.training.checkpoints import CheckpointStore, load_checkpoint
from firepred.training.evaluator import Evaluator
from firepred.training.tracker import Tracker
from firepred.training.trainer import Trainer


def seed_everything(seed):
    """Seed numpy, torch (CPU and every GPU) and Python's random; return the seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def prepare_model(name, spec, device):
    """Build the registered model `name` for `spec`, wrapped in DataParallel and moved to `device`."""
    model = nn.DataParallel(build_model(name, spec))
    model.to(device)
    return model


@dataclass(frozen=True)
class TrainingData:
    """What Trainer.fit reads: the two loaders, plus the TESSERA statistics saved with checkpoints."""

    train_loader: object
    val_loader: object
    tessera_stats: dict = None


@dataclass(frozen=True)
class FireData:
    """Where one run's window data is, and how its datasets and loaders are made."""

    config: object
    root: Path
    tessera_dir: Path = None

    @classmethod
    def locate(cls, config):
        """Find train/, val/, test/ under config.data_root; with --unzip-dir, unpack the archives once.

        Only the window archives are unpacked (the same arrays, faster reads);
        the TESSERA folder stays where it is.
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
        """True when the test split holds any fire."""
        return bool(find_fire_sources(self.root / "test"))

    def dataset(self, split, augment, tessera_stats=None, tessera_maps=None):
        """The FireDataset of one split (TESSERA maps/statistics only in TESSERA mode)."""
        return FireDataset(
            npz_dir=self.root / split,
            ts_length=self.config.ts_length,
            n_channel=N_STORED_CHANNELS,
            target_is_single_day=True,
            use_augmentations=augment,
            tessera_dir=self.tessera_dir,
            tessera_stats=tessera_stats,
            tessera_maps=tessera_maps,
        )

    def loader(self, dataset, shuffle, seed_offset):
        """A DataLoader on its own random generator (seeded from config.seed)."""
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

    def training(self):
        """TrainingData: augmented, shuffled training windows and fixed validation windows.

        TESSERA normalization comes from the training fires only (no leak from
        validation or test); each training fire's TESSERA file is read once.
        """
        stats, maps = None, None
        if self.tessera_dir is not None:
            maps = load_tessera_maps(self.tessera_dir, find_fire_sources(self.root / "train"))
            stats = compute_tessera_stats(maps.values(), min_coverage=self.config.tessera_min_coverage)
        train = self.dataset("train", augment=True, tessera_stats=stats, tessera_maps=maps)
        val = self.dataset("val", augment=False, tessera_stats=stats)
        # Validation for the shape check, so no random augmentation is drawn.
        validate_dataset_output(val, self.config.input_channels, "validation")
        return TrainingData(self.loader(train, shuffle=True, seed_offset=0),
                            self.loader(val, shuffle=False, seed_offset=1), stats)

    def test_dataset(self, tessera_stats=None):
        """The unaugmented test FireDataset, checked against the model's input width."""
        dataset = self.dataset("test", augment=False, tessera_stats=tessera_stats)
        validate_dataset_output(dataset, self.config.input_channels, "test")
        return dataset


@dataclass(frozen=True)
class RunResult:
    """What a run produced: the tested (or best) checkpoint and the final test scores (or None)."""

    checkpoint: Path
    scores: object = None


class Experiment:
    """One training or test run of a RunConfig."""

    def __init__(self, config):
        """Remember the settings; run() does the work."""
        self.config = config

    def train(self, data, model, device, tracker):
        """Fit the model and return the path of its best checkpoint."""
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
        """Load `checkpoint_path` into the model, score the test fires and log them; return the TestScores."""
        config = self.config
        checkpoint = load_checkpoint(
            checkpoint_path, checkpoint_map_location(device), config,
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
        """Train (unless -test) and test; returns a RunResult."""
        config = self.config
        seed_everything(config.seed)
        data = FireData.locate(config)
        device = resolve_device(config.device)
        seed_device(device, config.seed)
        with Tracker(config, device) as tracker:
            model = prepare_model(config.model, config.model_spec(), device)
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
