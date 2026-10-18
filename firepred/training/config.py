"""The settings of one run, resolved once: RunConfig and every name derived from it.

Level 3 (training). Built by cli/train.py (config_from_args) and read by every
other training module. Nothing here imports torch, so a notebook can build a
RunConfig to compose a training command cheaply.

Information flows one way: the command line becomes a frozen RunConfig, and
each class of the run (Tracker, FireData, Trainer, Evaluator) reads the
settings it needs from it. Nothing changes a setting after this point, and
every derived value (loss record, tags, run and checkpoint names) has exactly
one definition, below.

These names are how old runs and checkpoints are found again, so they never
change. A run with the paper's Dice+CE loss and TESSERA inputs, for example:
    run name    train_pred_swinunetr3d_num_heads_3_hidden_size_36_batchsize_8_tessera_dicece
    checkpoint  model_swinunetr3d_mode_pred_num_heads_3_hidden_size_36_batchsize_8_
                checkpoint_epoch_12_nc_172_ts_6_tessera_dicece.pth
"""

from dataclasses import dataclass, field
from pathlib import Path

from firepred.config.dataset import DEFAULT_DATA_ROOT, DEFAULT_TEST_CSV, TS_INTERVAL, TS_LENGTH, model_input_channels
from firepred.config.training import (
    DEFAULT_EPOCHS,
    DEFAULT_LOAD_EPOCH,
    DEFAULT_NUM_WORKERS,
    DEFAULT_UNZIP_DIR,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    LAMBDA_CE,
    LAMBDA_DICE,
    LAMBDA_FOCAL,
    LOSSES,
    LR_DECAY_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    LR_SCHEDULES,
    NUM_CLASSES,
    PAPER_CE_WEIGHT,
    SEED,
    TESSERA_MIN_COVERAGE,
    TOP_N_CHECKPOINTS,
    WANDB_ENTITY,
)


def default_loss(model_name):
    """Paper: Dice+CE for U-Net-3D, UNETR-3D and SwinUNETR-3D; the
    Attention-U-Net-3D "kept the original setup of using Dice loss". [paper]"""
    return "dice" if model_name == "attunet" else "dicece"


@dataclass(frozen=True)
class LossConfig:
    """Which loss, with its weights (see training/objectives.build_criterion)."""

    name: str = "dicece"
    ce_weight: tuple = PAPER_CE_WEIGHT
    lambda_dice: float = LAMBDA_DICE
    lambda_ce: float = LAMBDA_CE
    lambda_focal: float = LAMBDA_FOCAL
    focal_gamma: float = FOCAL_GAMMA
    focal_alpha: float = FOCAL_ALPHA

    def record(self):
        """The loss settings recorded in W&B and in every checkpoint."""
        config = {"loss": self.name}
        if self.name == "dicece":
            config.update(ce_weight=list(self.ce_weight), lambda_dice=self.lambda_dice, lambda_ce=self.lambda_ce)
        if self.name == "dicefocal":
            config.update(
                lambda_dice=self.lambda_dice,
                lambda_focal=self.lambda_focal,
                focal_gamma=self.focal_gamma,
                focal_alpha=self.focal_alpha,
            )
        return config


@dataclass(frozen=True)
class ScheduleConfig:
    """The optional learning-rate decay. [ours]"""

    name: str = "none"
    decay_factor: float = LR_DECAY_FACTOR
    patience: int = LR_PATIENCE
    min_lr: float = LR_MIN

    def record(self):
        """The schedule recorded in W&B and in every checkpoint, so decayed runs
        are never mistaken for paper runs."""
        config = {"lr_schedule": self.name}
        if self.name != "none":
            config["lr_min"] = self.min_lr
        if self.name == "plateau":
            config.update(lr_decay_factor=self.decay_factor, lr_patience=self.patience)
        return config

    @property
    def tag(self):
        """'' for a constant rate, else '_lr-<schedule>'."""
        return "" if self.name == "none" else f"_lr-{self.name}"


@dataclass(frozen=True)
class RunConfig:
    """Every setting of one training or test run (one field per training flag)."""

    model: str
    mode: str = "pred"
    batch_size: int = 8
    learning_rate: float = 0.001
    num_heads: int = 3
    feature_size: int = 36
    ts_length: int = TS_LENGTH
    ts_interval: int = TS_INTERVAL     # -it: how the windows were built; recorded only
    run_number: int = 1                # -r: recorded only [released]
    seed: int = SEED
    epochs: int = DEFAULT_EPOCHS
    num_workers: int = DEFAULT_NUM_WORKERS
    device: str = "auto"
    data_root: Path = DEFAULT_DATA_ROOT
    unzip_dir: str = DEFAULT_UNZIP_DIR
    tessera: bool = False
    tessera_dir: Path = None
    tessera_min_coverage: float = TESSERA_MIN_COVERAGE
    loss: LossConfig = None            # None -> the paper's loss for `model` (default_loss)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    swin_temporal_downsample: bool = False
    unetr_version: str = "v1"
    extra_model_options: tuple = ()    # (KEY, VALUE) pairs from --model-option
    test_only: bool = False
    checkpoint: Path = None
    trust_checkpoint: bool = False
    load_epoch: int = DEFAULT_LOAD_EPOCH
    test_csv: Path = DEFAULT_TEST_CSV
    plot_dir: Path = Path("evaluation_plot")
    test_every: int = 0
    test_after_train: bool = True
    upload_checkpoints: bool = False
    wandb_entity: str = WANDB_ENTITY
    expected_channels: int = None      # -nc: checked against input_channels when given
    keep_checkpoints: int = TOP_N_CHECKPOINTS

    def __post_init__(self):
        """Fill the default loss and validate settings that combine."""
        if self.loss is None:
            object.__setattr__(self, "loss", LossConfig(default_loss(self.model)))
        if self.loss.name not in LOSSES:
            raise ValueError(f"Unsupported loss: {self.loss.name}")
        if self.schedule.name not in LR_SCHEDULES:
            raise ValueError(f"Unsupported learning-rate schedule: {self.schedule.name}")
        if self.expected_channels is not None and self.expected_channels != self.input_channels:
            raise ValueError(
                f"-nc {self.expected_channels} does not match the preprocessing: expected "
                f"{self.input_channels} {'with' if self.tessera else 'without'} --tessera"
            )

    # ----- data ----- #

    @property
    def input_channels(self):
        """Model input channels after preprocessing: 43, or 172 with TESSERA."""
        return model_input_channels(self.tessera)

    @property
    def tessera_root(self):
        """Folder of tessera/p<id>.npz files (None without TESSERA); default <data-root>/tessera."""
        if not self.tessera:
            return None
        return Path(self.tessera_dir).expanduser() if self.tessera_dir is not None else Path(self.data_root) / "tessera"

    # ----- model ----- #

    @property
    def model_options(self):
        """ModelSpec.options: the dedicated flags, then any --model-option pairs."""
        return {
            "unetr_version": self.unetr_version,
            "swin_temporal_downsample": self.swin_temporal_downsample,
            **dict(self.extra_model_options),
        }

    def model_spec(self, image_size=None):
        """The ModelSpec handed to the model builder (models/registry.py)."""
        from firepred.models.registry import ModelSpec

        sizes = {} if image_size is None else {"image_size": tuple(image_size)}
        return ModelSpec(
            in_channels=self.input_channels,
            num_classes=NUM_CLASSES,
            ts_length=self.ts_length,
            num_heads=self.num_heads,
            feature_size=self.feature_size,
            options=self.model_options,
            **sizes,
        )

    # ----- names ----- #

    @property
    def run_tag(self):
        """Suffix for checkpoint names; the released-code Dice loss keeps the old names."""
        tessera_tag = "_tessera" if self.tessera else ""
        return tessera_tag + ("" if self.loss.name == "dice" else f"_{self.loss.name}") + self.schedule.tag

    @property
    def run_name(self):
        """The W&B run name."""
        return (
            f"{'test' if self.test_only else 'train'}_{self.mode}_{self.model}_"
            f"num_heads_{self.num_heads}_hidden_size_{self.feature_size}_"
            f"batchsize_{self.batch_size}{'_tessera' if self.tessera else ''}"
            f"{'' if self.loss.name == 'dice' else '_' + self.loss.name}"
            f"{self.schedule.tag}"
        )

    @property
    def checkpoint_template(self):
        """Checkpoint file name with a literal '{epoch}' placeholder (1-based epoch)."""
        return (
            f"model_{self.model}_mode_{self.mode}_num_heads_{self.num_heads}_"
            f"hidden_size_{self.feature_size}_batchsize_{self.batch_size}_"
            "checkpoint_epoch_{epoch}_"
            f"nc_{self.input_channels}_ts_{self.ts_length}{self.run_tag}.pth"
        )

    @property
    def test_checkpoint(self):
        """Where -test looks without --checkpoint. [released]

        Training saves into the W&B run directory instead (docs/KNOWN_ISSUES.md).
        """
        return Path("saved_models") / self.checkpoint_template.format(epoch=self.load_epoch)

    def artifact_name(self, run_id):
        """W&B model-artifact name for a run's best checkpoints."""
        return f"{self.model}{self.run_tag.replace('_', '-')}-checkpoints-{run_id}"

    # ----- what gets recorded ----- #

    def checkpoint_settings(self, tessera_stats):
        """Settings saved in every checkpoint, needed to rebuild identical inputs at test time."""
        settings = {
            "use_tessera": self.tessera,
            "model_input_channels": self.input_channels,
            "swin_temporal_downsample": self.swin_temporal_downsample,
            "tessera_stats": tessera_stats,
            "loss_config": self.loss.record(),
            "lr_config": self.schedule.record(),
        }
        if self.extra_model_options:
            settings["model_options"] = dict(self.extra_model_options)
        return settings

    def wandb_config(self, on_tpu=False):
        """The run's W&B config: hyperparameters, loss, schedule and model options."""
        config = {
            "learning_rate": self.learning_rate,
            # Logged only; Adam never receives it (docs/KNOWN_ISSUES.md).
            "weight_decay": self.learning_rate / 10,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "mode": self.mode,
            "model": self.model,
            "tessera": self.tessera,
            "input_channels": self.input_channels,
            **self.loss.record(),
        }
        if on_tpu:
            config["device"] = "tpu"
        if self.model == "swinunetr3d":
            config["swin_temporal_downsample"] = self.swin_temporal_downsample
        if self.extra_model_options:
            config["model_options"] = dict(self.extra_model_options)
        config.update(self.schedule.record())
        config["test_every"] = self.test_every  # [ours] periodic test scoring (logging only)
        return config
