## For the original github: https://github.com/zhaoyutim/TS-SatFire
# TS-SatFire Time-Series Multi-Task Satellite Imagery Dataset for Wildfire Detection and Prediction

🔥This paper has been accepted in Nature Scientific Data! Link:[[https://www.nature.com/articles/s41597-025-06271-3]]

## Repo structure

Next-day wildfire spread prediction with SwinUNETR-3D (and any model you add),
organized as a hierarchy that builds toward the model. Start with
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the model itself is explained
from scratch in [docs/swinunetr3d/](docs/swinunetr3d/).

    .
    ├── firepred/                 # the package (python -m firepred.cli.<name>)
    │   ├── config/               # L0 every project-wide constant: channels, dataset, paths, training
    │   ├── data/                 # L1 GeoTIFFs/TESSERA -> archives -> tensors; notebooks call windows_master.py, tessera_master.py
    │   ├── models/               # L2 blocks/ (Swin pieces), zoo/ (one file per model), registry
    │   ├── training/             # L3 loss, metrics, loop, evaluation, W&B, checkpoints
    │   └── cli/                  # L4 train, dataset, check, new_model
    ├── tests/                    # pytest: unit tests, golden/ (same-output values), end-to-end runs
    ├── checks/                   # dataset checks and experiment diagnostics (start with count_windows.py)
    ├── tools/                    # trace_shapes.py: real tensor shapes through SwinUNETR-3D
    ├── scripts/ice/              # GT PACE ICE: setup, partitions, Slurm jobs for data, training and tests, README guide
    ├── docs/                     # ARCHITECTURE, KNOWN_ISSUES, swinunetr3d/ explainer
    ├── roi/                      # fire metadata CSVs (ids, dates, locations, label rule)
    └── figures/                  # dataset figures from the paper

## Abstract

We introduce a comprehensive multi-temporal remote sensing dataset covering the entire life cycle of wildfires for active fire detection, daily wildfire monitoring and next-day wildfire prediction. This multi-task dataset comprises 179 wildfire events recorded majority in the US between January 2017 and October 2021. For each wildfire, images from the beginning until the end of the wildfire are provided. It includes a total of 3552 surface reflectance images along with auxiliary data such as weather, topography, and fuel information. Labels for current active fires (AF) and burned areas (BA) are provided for each image. Manual quality assurance is performed for all AF labels and BA test labels.The dataset sets three distinct tasks: a) active fire detection, b) daily burned area mapping and c) daily wildfire progression prediction. Detection tasks, such as active fire detection and burned area mapping, require pixel-wise classification utilizing multi-spectral, multi-temporal images. Prediction tasks involve learning the underlying physical processes by integrating satellite observations with auxiliary data. The primary objective of this dataset is to stimulate further research in wildfire monitoring, particularly leveraging advanced deep learning models capable of effectively processing multi-temporal, multi-spectral images to detect fires and accurately predict fire progression. The benchmarks for detection and prediction tasks indicate that utilizing both spatial and temporal information is crucial for this dataset.

## Spectral bands used in the dataset
![Alt text](figures/flowchart.svg?raw=true "Dataset Setup")
![Alt text](figures/Bands.png?raw=true "Dataset Channels")


## Preparing the dataset
The dataset in GeoTIFF format can be downloaded from the Kaggle link below:
[[https://www.kaggle.com/datasets/z789456sx/ts-satfire](https://www.kaggle.com/datasets/z789456sx/ts-satfire)]

Prepare the environment
```
conda env create -f environment.yml
```
Window and TESSERA archives are Zstandard-compressed. On Python < 3.14 (Kaggle, Colab, PACE ICE) also run
`pip install backports.zstd` before reading or building data. The code runs on Python 3.9 and newer;
on 3.9 pip installs backports.zstd 1.3.0, its last release for 3.9.

The Kaggle/Colab notebooks run this local checkout, not a GitHub clone: `python tools/pack_code.py`
writes `dist/firepred-code.zip` (uncommitted changes included, with a `code_version.json` saying which
commit and whether it was modified); upload it and point the notebook's `CODE_SOURCE` at it.

Build or download the prediction dataset (streams from/to Hugging Face; token from `--token` or `$HF_TOKEN`),
then record its statistics in `./data/window_config.json`:
```
python -m firepred.cli.dataset process -ts 6 -it 1          # raw GeoTIFFs -> window archives
python -m firepred.cli.dataset download -ts 6 -it 1 --data-root ./data
python -m firepred.cli.dataset count-pixels --data-root ./data            # background/fire pixels per split
python -m firepred.cli.dataset compute-normalization --data-root ./data   # per-channel mean/std of train/
```
Training reads `<data-root>/window_config.json` first and will not start without it.
`download` copies it from the window repo; `upload-zip` packs it into data.zip; `--upload`
on either statistics command also writes it back to the window repo. `rebuild-zip` does all of
it: it downloads every archive, measures both statistics into `window_config.json` (local and
repo), then zips and uploads data.zip.

### Daily archives (format 3): the same windows, about 6x smaller
A window archive stores every window whole, so with `-it 1` each day is stored up to 6 times.
A daily archive stores each day once: terrain and land cover once per fire, the other channels in
blocks of 6 days, and the targets as bits. Windows are cut from the days when they are read. Every
reader (training, statistics, checks) gets the same arrays bit for bit, and the window list comes
from the number of days, so nothing changes for training. A data root holds the archives once,
with no data.zip and nothing to unzip.
```
python -m firepred.cli.dataset convert-daily -ts 6 -it 1 --data-root ./data --dest-root ./data-daily
python -m firepred.cli.dataset --window-repo SamuelWu318/ts-satfire-processed-6-1 convert-daily -ts 6 -it 1 \
    --dest-root ./data-daily --dest-repo SamuelWu318/ts-satfire-daily   # one fire on disk at a time
python -m firepred.cli.dataset --window-repo SamuelWu318/ts-satfire-daily download --data-root ./data-daily
```
`convert-daily` reads every converted window and target back and compares them bit for bit with the
source, then keeps the source's `window_config.json` statistics (they describe identical windows).
`stage` downloads a repo without data.zip one archive at a time, straight into its split folder.
- One daily archive gives windows of any length and stride whose target days it holds.
  `open_fire(path, length, interval)` reads them; an archive converted from 6-1 windows serves
  `-ts 6` and longer, at any `-it`. Training still uses the settings in `window_config.json`.
- Code from before format 3 stops on these files with "unsupported format version 3".
  Format-2 archives still read as before.

## Train and test
```
python -m firepred.cli.check model -m swinunetr3d            # model sanity check, one step
python -m firepred.cli.check labels --data-root ./data        # background vs fire pixels of train/
python checks/count_windows.py --data-root ./data --batch-size 8  # windows and batches per epoch
python -m firepred.cli.train -m swinunetr3d -mode pred -b 8 -r 1 -lr 0.001 -nh 3 -ed 36 -ts 6 -it 1 --data-root ./data
python -m firepred.cli.train ... -test --checkpoint <file.pth>
```
Training scores the test fires with its best checkpoint at the end. Inputs are standardized
with the training split's mean/std from `window_config.json` (`--normalization paper` uses the
released fixed numbers instead); checkpoints record the numbers, and `-test` reuses them. The
Dice + CE class weights default to `--ce-weight auto`: 1.0 and the training split's
background-to-fire pixel ratio from `window_config.json` (`check labels` recounts the same number
from the labels). Without usable counts the run falls back to the paper's `1.0 446.7836` and
records that in W&B (`defaults_used`); two numbers set the weights by hand. Dice + CE is MONAI's
`DiceCELoss`, as in the paper: the class weight scales both the Dice and the CE term. Add a model with
`python -m firepred.cli.new_model NAME` (see docs/ARCHITECTURE.md). This repository
covers the paper's prediction task only.

**Resuming (`--resume FILE`, new):** training rewrites FILE after every epoch with the weights,
optimizer, gradient scaler, schedule, kept checkpoints, every random generator and the W&B run id.
Running the same command again continues after the last finished epoch, in the same W&B run, and
ends with the same weights as a run that was never stopped. A file from a run with other settings
(model, epochs, learning rate, batch size, seed, -ts/-it) is refused.

## GT PACE ICE

`scripts/ice/` runs everything from a terminal on PACE ICE: `ssh <gt-username>@login-ice.pace.gatech.edu`
(GT VPN when off campus), or in a browser ICE OnDemand > Clusters > ICE Shell Access. By default the jobs
use the `ice-gpu` partition (one GPU per job) and `ice-cpu`, both under your Slurm default QOS, with a
16 h limit per job. `bash scripts/ice/partitions.sh` shows which partitions accept
your jobs and under which QOS; choose others with `PARTITION=` (one or a comma list such as
`ice-gpu,coe-gpu`, whichever has a free GPU first), `QOS=`, `CPU_PARTITION=`, `CPU_QOS=`,
`ACCOUNT=` and `GPU_TYPE=` (e.g. `h100`). `plan.sh` checks each listed partition with
`sbatch --test-only` and leaves out one that refuses the jobs (College of
Engineering H100/H200 nodes: `PARTITION=coe-gpu`). The environment is a venv from ICE's own Python 3.9 by default, with
the paper's torch 2.3.1 and MONAI 1.3.2; `setup.sh` explains the other choices. Everything large lives
in `~/scratch` (the home directory is small: `pace-quota`). Full guide (quickstart, running, testing,
partitions, jobs and quotas): [scripts/ice/README.md](scripts/ice/README.md).
```
git clone -b daily-store https://github.com/GT-Fire-Research-and-Innovation/fire-research-and-innovation.git
cd fire-research-and-innovation
bash scripts/ice/setup.sh                       # once: Python env in ~/scratch/envs/firepred, HF + W&B logins
bash scripts/ice/partitions.sh                  # where your jobs are accepted (partitions, QOS, GPUs, free nodes)
NAME=swin_daily bash scripts/ice/plan.sh        # data (first time) -> 3 chained 16 h jobs
bash scripts/ice/status.sh                      # jobs, run states, last epoch of the latest logs
```
`setup.sh` only checks library imports; it does not run pytest. Tests are optional:
`NAME=swin_daily SMOKE=1 bash scripts/ice/plan.sh` adds a GPU smoke test before training,
and `sbatch scripts/ice/test.sbatch` runs the full CPU test suite (`SLOW=1` includes slow tests).

- `data.sbatch` converts `ts-satfire-processed-6-1` into a daily root at `~/scratch/firepred/data-daily`
  (`DAILY_REPO=...` downloads an existing daily repo instead; `DEST_REPO=...` also uploads the conversion).
- `train.sbatch` trains one run with `--resume ~/scratch/firepred/runs/$NAME.state`, so each job of a
  chain continues the last one; a finished run leaves `$NAME.done` and the rest of the chain exits.
  `MODEL`, `BATCH`, `LR` and `EXTRA` (any other train flags) set the run. The daily archives give the
  windows they were converted with (6 days, 1 day apart), so `-ts 6 -it 1` (the defaults) must match
  the data root's `window_config.json`.
- Checkpoints and W&B runs go to `~/scratch/firepred/wandb/`; logs to `logs/` in the repository.

## Author

#### Yu Zhao (zhaoyutim@gmail.com), Sebastian Gerard(sgerard@kth.se), Yifang Ban (yifang@kth.se), KTH Royal Institute of Technology, Stockholm, Sweden

## Acknowledgement
#### The research is part of the project ‘Sentinel4Wildfire’ funded by Formas, the Swedish research council for sustainable development and the project ‘EO-AI4Global Change’ funded by Digital Futures.
