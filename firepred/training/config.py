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


@dataclass(frozen=True)
class LossConfig:
    """Hold the training loss and its weights.

    The loss is built from this by objectives.TrainingLoss. The weights that
    do not apply to the chosen loss are kept but ignored.

    Attributes:
        name (str): "dice", "dicece" or "dicefocal".
        ce_weight (tuple[float, float]): class weights (background, fire) for dicece.
        lambda_dice (float): weight of the Dice term.
        lambda_ce (float): weight of the cross-entropy term (dicece).
        lambda_focal (float): weight of the focal term (dicefocal).
        focal_gamma (float): focal exponent (dicefocal).
        focal_alpha (float | None): weight of fire pixels in the focal term (dicefocal).
    """

    name: str = "dicece"
    ce_weight: tuple = PAPER_CE_WEIGHT
    lambda_dice: float = LAMBDA_DICE
    lambda_ce: float = LAMBDA_CE
    lambda_focal: float = LAMBDA_FOCAL
    focal_gamma: float = FOCAL_GAMMA
    focal_alpha: float = FOCAL_ALPHA

    @staticmethod
    def paper_loss(model_name):
        """Pick the paper's loss for a model.

        The paper trains U-Net-3D, UNETR-3D and SwinUNETR-3D with Dice + cross
        entropy. Attention-U-Net-3D "kept the original setup of using Dice loss".

        Args:
            model_name (str): the model name.

        Returns:
            str: "dice" for attunet, "dicece" otherwise.
        """
        return "dice" if model_name == "attunet" else "dicece"

    def record(self):
        """List the loss settings recorded in W&B and in every checkpoint.

        Only the weights the chosen loss uses are listed. The record is how a run's
        loss is found again later.

        Returns:
            dict: e.g. {"loss": "dicece", "ce_weight": [...], "lambda_dice": 1.0, "lambda_ce": 1.0}.
        """
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
    """Hold the optional learning-rate decay.

    This is ours, not the paper's [ours]. The paper keeps the rate constant ("none"). "plateau" lowers it when the
    validation loss stops improving; "cosine" lowers it smoothly to min_lr.

    Attributes:
        name (str): "none", "plateau" or "cosine".
        decay_factor (float): plateau: multiply the rate by this.
        patience (int): plateau: epochs without improvement before decaying.
        min_lr (float): the lowest rate (plateau and cosine).
    """

    name: str = "none"
    decay_factor: float = LR_DECAY_FACTOR
    patience: int = LR_PATIENCE
    min_lr: float = LR_MIN

    def record(self):
        """List the schedule settings recorded in W&B and in every checkpoint.

        Recording the schedule keeps decayed runs from being mistaken for paper
        runs. Only the settings the schedule uses are listed.

        Returns:
            dict: e.g. {"lr_schedule": "plateau", "lr_min": 1e-06, ...}.
        """
        config = {"lr_schedule": self.name}
        if self.name != "none":
            config["lr_min"] = self.min_lr
        if self.name == "plateau":
            config.update(lr_decay_factor=self.decay_factor, lr_patience=self.patience)
        return config

    @property
    def tag(self):
        """Give the schedule's suffix for run and checkpoint names.

        A constant rate adds nothing, so paper runs keep their old names. A decay
        adds "_lr-<schedule>".

        Returns:
            str: "" or "_lr-<schedule>".
        """
        return "" if self.name == "none" else f"_lr-{self.name}"


@dataclass(frozen=True)
class RunConfig:
    """Hold every setting of one training or test run, fixed once at the start.

    The command line becomes one RunConfig, and every part of the run reads its
    settings from it. Because it is frozen, no step can change a setting
    another step relies on. Every derived value (channels, names, recorded
    settings) has exactly one definition here.

    Attributes:
        model (str): the registered model name.
        mode (str): the task label, e.g. "pred" (part of every name).
        batch_size (int): windows per batch.
        learning_rate (float): Adam's learning rate.
        num_heads (int): attention heads per stage.
        feature_size (int): the model's base feature width.
        ts_length (int): input days per window.
        ts_interval (int): days between window starts (recorded only).
        run_number (int): the run number (recorded only).
        seed (int): the random seed.
        epochs (int): training passes over the data.
        num_workers (int): data-loading processes.
        device (str): "auto", "cpu", "cuda" or "tpu".
        data_root (Path): the folder with train/, val/ and test/.
        unzip_dir (str): "off", "auto" or a folder to unzip the archives into.
        tessera (bool): append TESSERA embeddings.
        tessera_dir (Path | None): the TESSERA folder (default <data_root>/tessera).
        tessera_min_coverage (float): TESSERA pixels below this coverage are masked.
        loss (LossConfig): the training loss.
        schedule (ScheduleConfig): the learning-rate schedule.
        swin_temporal_downsample (bool): SwinUNETR-3D also merges days.
        unetr_version (str): UNETR-3D sizes, "v1" (paper) or "v0".
        extra_model_options (tuple[tuple[str, object], ...]): --model-option pairs.
        test_only (bool): test a checkpoint instead of training.
        checkpoint (Path | None): the checkpoint to test.
        trust_checkpoint (bool): allow full unpickling of that checkpoint.
        load_epoch (int): the epoch in the default test checkpoint name.
        test_csv (Path): the test fires, in scoring order.
        plot_dir (Path): where test overlays go.
        test_every (int): score the test fires every N epochs (0 = never).
        test_after_train (bool): test the best checkpoint after training.
        upload_checkpoints (bool): upload kept checkpoints to W&B.
        wandb_entity (str): the W&B team.
        expected_channels (int | None): -nc, checked against the data.
        keep_checkpoints (int): how many best checkpoints stay on disk.
    """

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
    loss: LossConfig = None            # None -> the paper's loss for `model` (LossConfig.paper_loss)
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
        """Fill in the default loss and check settings that depend on each other.

        Without a loss, the model gets the paper's loss. The loss and schedule
        names must be known, and -nc must match the data's channel count.

        Raises:
            ValueError: on an unknown loss or schedule, or a wrong -nc.
        """
        if self.loss is None:
            object.__setattr__(self, "loss", LossConfig(LossConfig.paper_loss(self.model)))
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
        """Count the model's input channels after preprocessing.

        This follows from the TESSERA setting. It is 43 without TESSERA and 172 with it.

        Returns:
            int: the input channels.
        """
        return model_input_channels(self.tessera)

    @property
    def tessera_root(self):
        """Give the folder of TESSERA files, or None without TESSERA.

        By default it is the data root's tessera/ folder. --tessera-dir overrides it.

        Returns:
            Path | None: the TESSERA folder.
        """
        if not self.tessera:
            return None
        return Path(self.tessera_dir).expanduser() if self.tessera_dir is not None else Path(self.data_root) / "tessera"

    # ----- model ----- #

    @property
    def model_options(self):
        """Collect the model-specific settings handed to the model builder.

        The dedicated flags come first, then any --model-option pairs, which win on
        a name clash. They become ModelSpec.options.

        Returns:
            dict: e.g. {"unetr_version": "v1", "swin_temporal_downsample": False, ...}.
        """
        return {
            "unetr_version": self.unetr_version,
            "swin_temporal_downsample": self.swin_temporal_downsample,
            **dict(self.extra_model_options),
        }

    def model_spec(self, image_size=None):
        """Build the ModelSpec the model builder receives.

        The spec carries the input channels, the classes, the days, heads, width
        and options. image_size is overridden only by the shape check.

        Args:
            image_size (tuple[int, int] | None): another (H, W), or None for the default.

        Returns:
            ModelSpec: the spec.
        """
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
        """Give the suffix that ends checkpoint names.

        It marks TESSERA runs, non-Dice losses and decayed schedules. A Dice run
        with no extras keeps the released code's names.

        Returns:
            str: e.g. "_tessera_dicece_lr-plateau".
        """
        tessera_tag = "_tessera" if self.tessera else ""
        return tessera_tag + ("" if self.loss.name == "dice" else f"_{self.loss.name}") + self.schedule.tag

    @property
    def run_name(self):
        """Give the W&B run name.

        It starts with train_ or test_ and lists the model, heads, width, batch size
        and the run tag. Old runs are found again by these names, so they never
        change.

        Returns:
            str: e.g. "train_pred_swinunetr3d_num_heads_3_hidden_size_36_batchsize_8_dicece".
        """
        return (
            f"{'test' if self.test_only else 'train'}_{self.mode}_{self.model}_"
            f"num_heads_{self.num_heads}_hidden_size_{self.feature_size}_"
            f"batchsize_{self.batch_size}{'_tessera' if self.tessera else ''}"
            f"{'' if self.loss.name == 'dice' else '_' + self.loss.name}"
            f"{self.schedule.tag}"
        )

    @property
    def checkpoint_template(self):
        """Give the checkpoint file name, with an {epoch} placeholder.

        CheckpointStore fills in the 1-based epoch. The name records the model,
        width, batch size, channels, days and run tag.

        Returns:
            str: the file name template.
        """
        return (
            f"model_{self.model}_mode_{self.mode}_num_heads_{self.num_heads}_"
            f"hidden_size_{self.feature_size}_batchsize_{self.batch_size}_"
            "checkpoint_epoch_{epoch}_"
            f"nc_{self.input_channels}_ts_{self.ts_length}{self.run_tag}.pth"
        )

    @property
    def test_checkpoint(self):
        """Give where -test looks for a checkpoint when none is given.

        This follows the released code [released]. It is saved_models/ with the template at load_epoch. Training saves into the
        W&B run folder instead (docs/KNOWN_ISSUES.md), so pass --checkpoint.

        Returns:
            Path: the default test checkpoint path.
        """
        return Path("saved_models") / self.checkpoint_template.format(epoch=self.load_epoch)

    def artifact_name(self, run_id):
        """Name the W&B model artifact that holds a run's checkpoints.

        The name combines the model, the run tag and the run id. Every uploaded
        checkpoint is a new version of it.

        Args:
            run_id (str): the W&B run id.

        Returns:
            str: the artifact name.
        """
        return f"{self.model}{self.run_tag.replace('_', '-')}-checkpoints-{run_id}"

    # ----- what gets recorded ----- #

    def checkpoint_settings(self, tessera_stats):
        """Collect the settings stored in every checkpoint.

        A later test run needs them to rebuild exactly the same inputs: TESSERA
        mode and statistics, channels and temporal merging. The loss, schedule and
        model options are stored too, as a record.

        Args:
            tessera_stats (dict | None): the training TESSERA statistics.

        Returns:
            dict: the settings.
        """
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
        """Collect the run's W&B config: hyperparameters, loss, schedule and model options.

        This is what the W&B dashboard shows and filters by. Model-specific keys
        appear only when they apply.

        Args:
            on_tpu (bool): record that the run uses a TPU.

        Returns:
            dict: the config.
        """
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
