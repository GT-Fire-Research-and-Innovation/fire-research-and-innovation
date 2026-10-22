"""Training defaults: everything the trainer uses unless a CLI flag overrides it.

Level 0 (config). Used by training/ and cli/train.py. Provenance tags:
[paper] TS-SatFire paper text, [released] the released TS-SatFire code,
[ours] neither.
"""

# ----- Reproducibility ----- #

SEED = 42

# ----- Task ----- #

# Two output channels: 0 = background, 1 = newly burned ("fire growth").
NUM_CLASSES = 2
# Sigmoid probability above which a pixel counts as predicted.
PREDICTION_THRESHOLD = 0.5

# ----- Loss ----- #

# dice: sigmoid Dice loss [released]. dicece: 1:1 Dice + cross entropy [paper].
# dicefocal: sigmoid Dice plus a focal term on the fire channel [ours].
LOSSES = ("dice", "dicece", "dicefocal")
# "a 1:1 mixture of Dice and CE loss", CE weight [1.0, 446.7836]. [paper]
# The default CE weight is counted from the training labels in window_config.json
# (--ce-weight auto) [ours]; this is the fallback when the file has no usable counts.
PAPER_CE_WEIGHT = (1.0, 446.7836)
LAMBDA_DICE = 1.0
LAMBDA_CE = 1.0
# Focal term of dicefocal: sigmoid focal loss (Lin et al. 2017) on the fire channel,
# averaged over pixels. gamma down-weights easy pixels; alpha weighs fire pixels and
# 1 - alpha background (None = off). Defaults are Lin et al.'s gamma 2, alpha 0.25.
# "1:1" means lambda_dice = lambda_focal = 1 on the two means, as in MONAI's
# DiceFocalLoss; it is not equal influence (docs/swinunetr3d/07-learning-signal.md). [ours]
LAMBDA_FOCAL = 1.0
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# ----- Optimizer and learning-rate schedule ----- #

DEFAULT_EPOCHS = 100
# none: constant rate [paper, released]. plateau/cosine: decay [ours].
LR_SCHEDULES = ("none", "plateau", "cosine")
LR_DECAY_FACTOR = 0.5
LR_PATIENCE = 5
LR_MIN = 1e-6

# ----- Checkpoints ----- #

# How many lowest-validation-loss checkpoints stay on disk.
TOP_N_CHECKPOINTS = 1

# ----- Data loading ----- #

DEFAULT_NUM_WORKERS = 0
# Unzipping the window archives once (--unzip-dir auto -> <data-root>-npy) gives
# byte-identical arrays and skips decompression on every read, but takes about 5.5x the
# disk and its speed-up on Kaggle is unmeasured, so it is opt-in. [ours]
DEFAULT_UNZIP_DIR = "off"
# TESSERA pixels with less coverage are zeroed and masked. [ours]
TESSERA_MIN_COVERAGE = 0.5

# ----- Weights & Biases ----- #

WANDB_PROJECT = "AFBAPred"
WANDB_ENTITY = "gt-fri"
