# PACE ICE guide (`daily-store` branch)

Everything needed to train and test models on Georgia Tech's PACE ICE cluster with the scripts in
`scripts/ice/`. Run every command from the repository root on an ICE login node unless a step says
otherwise. Click a ▸ line to open its commands.

1. [Quickstart](#1-quickstart): one-time setup and data download, up to the point of launching a model.
2. [Run a model](#2-run-a-model): choosing the dataset, the model, the parameters and the partition.
3. [Test a model](#3-test-a-model): finding a run's test F1 and IoU, or testing a chosen checkpoint.
4. [Reference](#4-reference): partitions, QOS, GPUs, queues, storage and file locations.

---

## 1. Quickstart

Do steps 1 to 6 once. After that you never reinstall: see [Staying up to date](#staying-up-to-date).

### What you need before starting

| Item | Where to get it | Used for |
| --- | --- | --- |
| PACE ICE access | Your GT account (GT VPN when off campus) | Logging in, submitting jobs |
| GitHub access to the repo | `GT-Fire-Research-and-Innovation/fire-research-and-innovation` | Cloning. If `git clone` asks for a password, paste a GitHub personal access token, not your password |
| Hugging Face token (read) | https://huggingface.co/settings/tokens | Downloading datasets. Some dataset repos are gated: open each one on huggingface.co while logged in and request access |
| Weights & Biases API key | https://wandb.ai/authorize | Logging runs to team `gt-fri`, project `AFBAPred`. Ask a team admin to invite you to `gt-fri` |

Paste tokens only into the terminal prompts below. Never put them in a file in the repo or in a chat.

### Step 1: log in to ICE

```bash
ssh <gt-username>@login-ice.pace.gatech.edu
```

In a browser, ICE OnDemand > Clusters > ICE Shell Access opens the same terminal.

### Step 2: clone this branch

```bash
git clone -b daily-store https://github.com/GT-Fire-Research-and-Innovation/fire-research-and-innovation.git
cd fire-research-and-innovation
```

<details>
<summary>Also use the <code>fire-split-zones</code> branch</summary>

```bash
cd ~
git clone -b fire-split-zones https://github.com/GT-Fire-Research-and-Innovation/fire-research-and-innovation.git fire-research-and-innovation-zones
```

Both clones share the Python environment, data and run folders in `~/scratch`. Give runs distinct
`NAME`s across the two clones.

</details>

<details>
<summary>Check which branch a clone is on, or switch it</summary>

```bash
git branch --show-current
git switch daily-store
```

</details>

### Step 3: build the Python environment and log in

```bash
bash scripts/ice/setup.sh
```

This builds a venv in `~/scratch/envs/firepred` from ICE's Python 3.9 with torch 2.3.1 and MONAI 1.3.2
(the paper's versions). It takes several minutes, then prompts twice:

1. **Hugging Face**: paste your Hugging Face token. If asked whether to add it as a git credential, answer `n`.
2. **W&B**: paste your W&B API key.

Each login is skipped when you are already logged in. The script ends with `Ready.`.

<details>
<summary>Check or redo your Hugging Face and W&B logins</summary>

```bash
source scripts/ice/env.sh                      # the Hugging Face token lives in ~/scratch/.cache/huggingface
hf auth whoami || huggingface-cli whoami       # prints your Hugging Face user name
hf auth login  || huggingface-cli login        # log in to Hugging Face again
wandb login --relogin                          # paste a new W&B API key
```

</details>

<details>
<summary>Use another Python version or library stack</summary>

```bash
PYTHON=/path/to/python3.12 bash scripts/ice/setup.sh   # venv from another interpreter
CONDA_PYTHON=3.12 bash scripts/ice/setup.sh            # conda environment from ICE's mamba or anaconda3 module
STACK=latest bash scripts/ice/setup.sh                 # newest torch and MONAI for that Python
```

An existing environment is reused, so delete it first to switch:
`rm -rf ~/scratch/envs/firepred`. The golden tests were captured with the default stack.

</details>

### Step 4: check which partitions accept your jobs

```bash
bash scripts/ice/partitions.sh coc-gpu ice-gpu pace-gpu ice-cpu
```

Read the `ACCESS` and `QOS` columns. The scripts default to GPU partition `coc-gpu`, CPU partition
`ice-cpu` and QOS `coc-ice`.

<details>
<summary><code>coc-gpu</code> and <code>ice-cpu</code> say <code>yes</code> with QOS <code>coc-ice</code></summary>

Nothing to set. Every command in this guide works as written.

</details>

<details>
<summary>They say <code>no</code>, or show another QOS</summary>

Pick a GPU row and a CPU row that say `yes`, and pass their names and QOS to every `plan.sh`
command. For example, when `ice-gpu` and `ice-cpu` say `yes` with QOS `default`:

```bash
NAME=swin_daily PARTITION=ice-gpu CPU_PARTITION=ice-cpu QOS=default CPU_QOS=default bash scripts/ice/plan.sh
```

In hand-written `sbatch` commands in this guide, replace `-p coc-gpu --qos=coc-ice` with
`-p ice-gpu`, and leave out `--qos` when the QOS is `default`. Run `bash scripts/ice/partitions.sh`
with no arguments to see every partition.

</details>

### Step 5: download a dataset

Training reads a local folder, the data root. Download one now with a CPU job. Open the option you
want. Each command can be run again after a failure or time limit: files already downloaded are kept.

<details>
<summary><b>Daily dataset, no TESSERA</b> (recommended to start)</summary>

```bash
sbatch -J fp-download -p ice-cpu --qos=coc-ice -c 8 --mem=64G -t 16:00:00 -o logs/%x-%j.out \
  --wrap 'source scripts/ice/env.sh && python -m firepred.cli.dataset \
    --window-repo SamuelWu318/ts-satfire-daily --work-dir "$FIRE_WORK/tmp" \
    download -ts 6 -it 1 --data-root "$HOME/scratch/firepred/data-daily"'
```

Or by hand on an interactive CPU node:

```bash
salloc -p ice-cpu --qos=coc-ice -c 8 --mem=64G -t 4:00:00
hostname                     # a compute node; if it shows a login node, run: srun --pty bash
source scripts/ice/env.sh

python -m firepred.cli.dataset \
  --window-repo SamuelWu318/ts-satfire-daily \
  --work-dir "$FIRE_WORK/tmp" \
  download -ts 6 -it 1 \
  --data-root "$HOME/scratch/firepred/data-daily"

exit
```

Train on it with `DATA_ROOT="$HOME/scratch/firepred/data-daily"`, which is also the default.

</details>

<details>
<summary><b>Daily dataset with TESSERA embeddings</b></summary>

```bash
sbatch -J fp-download-tessera -p ice-cpu --qos=coc-ice -c 8 --mem=64G -t 16:00:00 -o logs/%x-%j.out \
  --wrap 'source scripts/ice/env.sh && python -m firepred.cli.dataset \
    --window-repo SamuelWu318/ts-satfire-daily --tessera-repo SamuelWu318/ts-tesserafire \
    download -ts 6 -it 1 --tessera --data-root "$HOME/scratch/firepred/data-daily-tessera"'
```

Or by hand on an interactive CPU node:

```bash
salloc -p ice-cpu --qos=coc-ice -c 8 --mem=64G -t 4:00:00
hostname                     # a compute node; if it shows a login node, run: srun --pty bash
source scripts/ice/env.sh

python -m firepred.cli.dataset \
  --window-repo SamuelWu318/ts-satfire-daily \
  --tessera-repo SamuelWu318/ts-tesserafire \
  download -ts 6 -it 1 --tessera \
  --data-root "$HOME/scratch/firepred/data-daily-tessera"

exit
```

Fires without prior-year TESSERA are left out, so keep this in its own data root. Embeddings land in
`<data root>/tessera`. Train on it with `DATA_ROOT="$HOME/scratch/firepred/data-daily-tessera"` and
`--tessera` in `EXTRA`.

</details>

<details>
<summary><b>Convert the processed 6-day windows into a daily dataset</b> (slow; only if the daily repo is unavailable)</summary>

```bash
sbatch -J fp-convert -p ice-cpu --qos=coc-ice -c 8 --mem=64G -t 16:00:00 -o logs/%x-%j.out \
  --wrap 'source scripts/ice/env.sh && python -m firepred.cli.dataset \
    --window-repo SamuelWu318/ts-satfire-processed-6-1 --work-dir "$FIRE_WORK/tmp" \
    convert-daily -ts 6 -it 1 --dest-root "$HOME/scratch/firepred/data-daily-converted"'
```

It downloads one fire at a time, converts it and checks it reads back identical. Fires already
converted are skipped when you run it again.

</details>

<details>
<summary><b>Convert and upload the daily dataset to your own Hugging Face repo</b></summary>

Your token needs write access to the target repo.

```bash
sbatch -J fp-convert -p ice-cpu --qos=coc-ice -c 8 --mem=64G -t 16:00:00 -o logs/%x-%j.out \
  --wrap 'source scripts/ice/env.sh && python -m firepred.cli.dataset \
    --window-repo SamuelWu318/ts-satfire-processed-6-1 --work-dir "$FIRE_WORK/tmp" \
    convert-daily -ts 6 -it 1 --dest-root "$HOME/scratch/firepred/data-daily-converted" \
    --dest-repo <your-hf-user>/ts-satfire-daily'
```

</details>

<details>
<summary><b>Skip this step and let the first training run download the data</b></summary>

`plan.sh` queues the download itself when the data root has no `window_config.json`. Training waits
for it:

```bash
NAME=swin_daily DAILY_REPO=SamuelWu318/ts-satfire-daily \
DATA_ROOT="$HOME/scratch/firepred/data-daily" bash scripts/ice/plan.sh
```

Without `DAILY_REPO`, `plan.sh` converts the processed 6-day windows instead, which is slower.
`plan.sh` does not download TESSERA embeddings.

</details>

<details>
<summary>Check that a download finished</summary>

```bash
squeue -u $USER                                   # is the job still running?
tail -5 logs/fp-download-<jobid>.out              # last lines of its log
D="$HOME/scratch/firepred/data-daily"
ls "$D"                                           # train  val  test  window_config.json  (tessera)
for s in train val test; do echo "$s: $(ls "$D/$s" | wc -l) files"; done
du -sh "$D"
```

A finished download has `window_config.json` and files in all three splits. Run the same download
command again when the job stopped early.

</details>

### Step 6 (optional): run the test suite

```bash
sbatch scripts/ice/test.sbatch
tail -3 logs/fp-test-<jobid>.out      # once the job ends: the pytest summary line
```

You are now ready to launch a model: go to [section 2](#2-run-a-model).

### Staying up to date

Your environment, datasets, run states, W&B runs and logins all live outside the repository, in
`~/scratch` and your home config. Pulling new code never touches them. **Do not delete the clone or
`~/scratch` when the GitHub branch changes.** Pull, look at what changed, then open the matching case.

```bash
cd ~/fire-research-and-innovation     # or wherever you cloned this branch
git pull
git log --oneline ORIG_HEAD..HEAD     # the commits that came in
git diff --stat ORIG_HEAD HEAD        # the files they changed
```

<details>
<summary>Only Python code changed (<code>firepred/</code>, models, training)</summary>

Nothing to redo. The next job you submit uses the new code.

</details>

<details>
<summary><code>scripts/ice/setup.sh</code> changed (library versions or packages)</summary>

```bash
bash scripts/ice/setup.sh
```

It reuses the environment, updates packages and skips logins you already did.

</details>

<details>
<summary>Rebuild the Python environment from scratch</summary>

Only when the Python version or `STACK` changes, or the environment is broken.

```bash
rm -rf ~/scratch/envs/firepred
bash scripts/ice/setup.sh
```

</details>

<details>
<summary>The dataset format changed, or you want a newer upload of the dataset</summary>

An existing data root is never replaced. Download into a new folder and train on that:

```bash
sbatch -J fp-download -p ice-cpu --qos=coc-ice -c 8 --mem=64G -t 16:00:00 -o logs/%x-%j.out \
  --wrap 'source scripts/ice/env.sh && python -m firepred.cli.dataset \
    --window-repo SamuelWu318/ts-satfire-daily --work-dir "$FIRE_WORK/tmp" \
    download -ts 6 -it 1 --data-root "$HOME/scratch/firepred/data-daily-v2"'
```

Then use `DATA_ROOT="$HOME/scratch/firepred/data-daily-v2"` in `plan.sh`. Delete the old root with
`rm -rf` once no run needs it.

</details>

<details>
<summary><code>git pull</code> refuses because you edited files</summary>

```bash
git status                     # which files you changed
git restore scripts/ice/plan.sh   # drop an edit you do not need (one file)
git stash && git pull && git stash pop   # or keep your edits on top of the new code
```

</details>

<details>
<summary>You have jobs queued</summary>

Each job runs the code in the clone at the moment it starts. If you pull while a training chain is
queued, its later jobs run the new code. Check first, and pull after the chain ends:

```bash
squeue -u $USER
```

To keep pulling meanwhile, use a second clone for new work.

</details>

---

## 2. Run a model

One command, `plan.sh`, queues everything for a run: data preparation when the data root is empty,
an optional GPU smoke test, then a chain of training jobs that continue one another.

```bash
NAME=swin_daily_lr0001 \
MODEL=swinunetr3d \
DATA_ROOT="$HOME/scratch/firepred/data-daily" \
BATCH=8 LR=0.0001 TS=6 IT=1 CHAIN=3 \
EXTRA="--epochs 100 --lr-schedule cosine --lr-min 0.000001" \
bash scripts/ice/plan.sh

bash scripts/ice/status.sh
```

Put settings before `bash scripts/ice/plan.sh` as above. Every setting except `NAME` has a default.
Each section below gives ready commands for one choice; combine their settings in one command.

### 2.1 Name the run

`NAME` identifies the run. Its state is saved to `~/scratch/firepred/runs/$NAME.state` after every
epoch, and a finished run leaves `$NAME.done`. Put the settings you vary into the name, such as
`swin_daily_lr0001_cosine`.

<details>
<summary>Start a new experiment</summary>

Use a name that does not exist yet:

```bash
ls ~/scratch/firepred/runs/          # names already used
NAME=swin_daily_lr0001_cosine LR=0.0001 EXTRA="--lr-schedule cosine" bash scripts/ice/plan.sh
```

</details>

<details>
<summary>Continue a run that stopped (time limit, failure, cancelled)</summary>

Submit the same command with the same `NAME` and settings. It resumes after the last finished epoch:

```bash
NAME=swin_daily_lr0001_cosine LR=0.0001 EXTRA="--lr-schedule cosine" CHAIN=2 bash scripts/ice/plan.sh
```

Resuming is refused when the model, loss, learning-rate schedule, TESSERA use, epochs, learning
rate, batch size, seed, `TS` or `IT` differ from the saved state. A finished run exits at once.

</details>

### 2.2 Choose the dataset

Point `DATA_ROOT` at a folder you downloaded in [Step 5](#step-5-download-a-dataset).

<details>
<summary><b>Daily dataset</b> (<code>data-daily</code>, the default)</summary>

```bash
NAME=swin_daily DATA_ROOT="$HOME/scratch/firepred/data-daily" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><b>Daily dataset with TESSERA</b> (<code>data-daily-tessera</code>)</summary>

Add `--tessera` to `EXTRA`:

```bash
NAME=swin_tessera DATA_ROOT="$HOME/scratch/firepred/data-daily-tessera" \
EXTRA="--tessera --epochs 100" bash scripts/ice/plan.sh
```

Embeddings are read from `<DATA_ROOT>/tessera`. Add `--tessera-dir <path>` to `EXTRA` for another
folder, and `--tessera-cells-per-pixel N` for embeddings built on a finer grid.

</details>

<details>
<summary><b>Converted dataset</b> (<code>data-daily-converted</code>)</summary>

```bash
NAME=swin_converted DATA_ROOT="$HOME/scratch/firepred/data-daily-converted" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><b>A data root that does not exist yet</b> (download as part of the run)</summary>

```bash
NAME=swin_daily DAILY_REPO=SamuelWu318/ts-satfire-daily \
DATA_ROOT="$HOME/scratch/firepred/data-daily-new" bash scripts/ice/plan.sh
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `DAILY_REPO` | unset | Download this daily dataset repo |
| `SRC_REPO` | `SamuelWu318/ts-satfire-processed-6-1` | Used when `DAILY_REPO` is unset: convert these windows (slower) |
| `DEST_REPO` | unset | Also upload the conversion to this repo |

`plan.sh` skips this whenever `DATA_ROOT/window_config.json` exists. Changing a repo setting never
replaces an existing data root, so use a new `DATA_ROOT` for each dataset.

</details>

### 2.3 Choose the model

Set `MODEL`. Every other setting stays the same.

<details>
<summary><code>swinunetr3d</code>: SwinUNETR-3D (default)</summary>

```bash
NAME=swin_daily MODEL=swinunetr3d bash scripts/ice/plan.sh
```

</details>

<details>
<summary><code>swinunetr3d_tpool</code>: SwinUNETR-3D that learns a weight per input day</summary>

It learns how to weight each day's output instead of averaging them.

```bash
NAME=swin_tpool MODEL=swinunetr3d_tpool bash scripts/ice/plan.sh
```

</details>

<details>
<summary><code>swinunetr3d</code> with TESSERA as extra input channels</summary>

```bash
NAME=swin_tessera_channels MODEL=swinunetr3d DATA_ROOT="$HOME/scratch/firepred/data-daily-tessera" \
EXTRA="--tessera" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><code>swinunetr3d_tessera</code>: SwinUNETR-3D with TESSERA as multi-scale context</summary>

TESSERA conditions every skip connection instead of joining the input.

```bash
NAME=swin_tessera_ctx MODEL=swinunetr3d_tessera DATA_ROOT="$HOME/scratch/firepred/data-daily-tessera" \
EXTRA="--tessera" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><code>swinunetr3d_tessera_tpool</code>: TESSERA context plus learned day weights</summary>

```bash
NAME=swin_tessera_tpool MODEL=swinunetr3d_tessera_tpool DATA_ROOT="$HOME/scratch/firepred/data-daily-tessera" \
EXTRA="--tessera" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><code>unetr3d</code>: UNETR-3D</summary>

```bash
NAME=unetr_daily MODEL=unetr3d bash scripts/ice/plan.sh
```

</details>

<details>
<summary><code>unet3d</code>: 3-D U-Net</summary>

```bash
NAME=unet3d_daily MODEL=unet3d bash scripts/ice/plan.sh
```

</details>

<details>
<summary><code>attunet</code>: Attention U-Net (default loss <code>dice</code>)</summary>

Every other model defaults to the `dicece` loss.

```bash
NAME=attunet_daily MODEL=attunet bash scripts/ice/plan.sh
```

</details>

<details>
<summary><code>convlstm</code>: ConvLSTM U-Net</summary>

```bash
NAME=convlstm_daily MODEL=convlstm bash scripts/ice/plan.sh
```

</details>

<details>
<summary><code>res18unet</code>: ResNet-18 U-Net over the days stacked as channels</summary>

```bash
NAME=res18_daily MODEL=res18unet bash scripts/ice/plan.sh
```

</details>

### 2.4 Choose the parameters

| Setting | Default | Meaning |
| --- | --- | --- |
| `BATCH` | `8` | Batch size |
| `LR` | `0.001` | Learning rate |
| `TS` | `6` | Input window length in consecutive days |
| `IT` | `1` | Days between window starts. `TS=6 IT=2` uses days 1-6 to predict day 7, then days 3-8 to predict day 9 |
| `CHAIN` | `3` | Training jobs queued one after another, each up to 16 hours, continuing the same run |
| `SMOKE` | `0` | `1` adds a short GPU check before training |
| `EXTRA` | empty | Any other training flags, in one quoted string |

`train.sbatch` already passes `-mode pred -nh 3 -ed 36 --num-workers 6`; flags in `EXTRA` override them.

On this branch, training checks `TS` and `IT` against `DATA_ROOT/window_config.json`. The daily
dataset needs `TS=6 IT=1`, the defaults. Other window settings need the `fire-split-zones` branch.

<details>
<summary><b>Quick trial</b>: 2 epochs, one job, with a GPU smoke test</summary>

```bash
NAME=trial_swin CHAIN=1 SMOKE=1 EXTRA="--epochs 2" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><b>Paper settings</b>: learning rate 0.001, constant, 100 epochs</summary>

These are the defaults:

```bash
NAME=swin_paper BATCH=8 LR=0.001 EXTRA="--epochs 100" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><b>Cosine learning-rate decay</b></summary>

```bash
NAME=swin_lr0001_cosine LR=0.0001 \
EXTRA="--epochs 100 --lr-schedule cosine --lr-min 0.000001" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><b>Plateau learning-rate decay</b>: halve after 5 epochs without improvement</summary>

```bash
NAME=swin_plateau LR=0.001 \
EXTRA="--epochs 100 --lr-schedule plateau --lr-patience 5 --lr-decay-factor 0.5 --lr-min 0.000001" \
bash scripts/ice/plan.sh
```

</details>

<details>
<summary><b>Learning-rate sweep</b>: one run per value</summary>

```bash
for LR in 0.001 0.0005 0.0001; do
  NAME=swin_lr$LR LR=$LR EXTRA="--epochs 100 --lr-schedule cosine --lr-min 0.000001" \
  bash scripts/ice/plan.sh
done
```

</details>

<details>
<summary><b>Repeat with different seeds</b></summary>

```bash
for SEED in 1 2 3; do
  NAME=swin_seed$SEED EXTRA="--epochs 100 -seed $SEED" bash scripts/ice/plan.sh
done
```

</details>

<details>
<summary><b>Batch size</b></summary>

```bash
NAME=swin_b16 BATCH=16 bash scripts/ice/plan.sh
```

A larger batch needs more GPU memory. Pair it with `GPU_TYPE=h100` or `h200` if it runs out.

</details>

<details>
<summary><b>Loss and class weights</b></summary>

```bash
NAME=swin_dice EXTRA="--loss dice" bash scripts/ice/plan.sh                 # Dice only
NAME=swin_dicece_w10 EXTRA="--loss dicece --ce-weight 1 10" bash scripts/ice/plan.sh   # background 1, fire 10
NAME=swin_dicece_auto EXTRA="--loss dicece --ce-weight auto" bash scripts/ice/plan.sh  # weights from the data's pixel ratio
```

</details>

<details>
<summary><b>More epochs than the chain can fit</b></summary>

Each job runs up to 16 hours. Queue more jobs for long runs:

```bash
NAME=swin_long CHAIN=5 EXTRA="--epochs 200" bash scripts/ice/plan.sh
```

If the chain ends early, run the same command again: it continues from the saved state.

</details>

<details>
<summary><b>Chart test scores during training</b></summary>

```bash
NAME=swin_watch EXTRA="--epochs 100 --test-every 10" bash scripts/ice/plan.sh
```

Scores the test fires every 10 epochs for W&B charts only. The kept checkpoint is still chosen by
validation loss.

</details>

<details>
<summary><b>Other flags</b></summary>

| Flag | Meaning |
| --- | --- |
| `--no-test-after-train` | Skip the final test after training |
| `--upload-checkpoints` | Store each best checkpoint as a W&B artifact |
| `--model-option KEY=VALUE` | Model-specific setting, repeatable |
| `--swin-temporal-downsample` | SwinUNETR-3D: halve the days in patch merging, as the released code does |

Every flag:

```bash
source scripts/ice/env.sh
python -m firepred.cli.train --help
```

</details>

### 2.5 Choose where it runs

| Setting | Default | Meaning |
| --- | --- | --- |
| `PARTITION` | `coc-gpu` | GPU partition, or a comma list: each job starts on whichever has a free GPU first |
| `QOS` | `coc-ice` | QOS of the GPU jobs. `QOS=default` uses your Slurm default |
| `GPU_TYPE` | any | One GPU model, such as `h100`, `h200`, `a100` |
| `CPU_PARTITION` | `ice-cpu` | Partition of the data job |
| `CPU_QOS` | same as `QOS` | QOS of the data job |
| `ACCOUNT` | your default | Slurm account |

`plan.sh` tests each listed partition with `sbatch --test-only` and drops one that refuses the job.

<details>
<summary>Default: <code>coc-gpu</code> under <code>coc-ice</code></summary>

```bash
NAME=swin_daily bash scripts/ice/plan.sh
```

</details>

<details>
<summary>Whichever of two partitions has a free GPU first</summary>

```bash
NAME=swin_daily PARTITION=coc-gpu,ice-gpu bash scripts/ice/plan.sh
```

All partitions in one list must accept the same QOS.

</details>

<details>
<summary>Only H100 GPUs</summary>

```bash
GPU_TYPE=h100 bash scripts/ice/partitions.sh coc-gpu ice-gpu      # who has them for you
NAME=swin_h100 PARTITION=coc-gpu,ice-gpu GPU_TYPE=h100 bash scripts/ice/plan.sh
```

</details>

<details>
<summary>Your Slurm default QOS (<code>partitions.sh</code> shows <code>default</code>)</summary>

```bash
NAME=swin_ice PARTITION=ice-gpu CPU_PARTITION=ice-cpu QOS=default CPU_QOS=default bash scripts/ice/plan.sh
```

</details>

<details>
<summary>Another partition with its own QOS</summary>

```bash
NAME=swin_pace PARTITION=pace-gpu QOS=<qos from partitions.sh> CPU_QOS=coc-ice bash scripts/ice/plan.sh
NAME=swin_coe PARTITION=coe-gpu QOS=coe-ice CPU_QOS=coc-ice bash scripts/ice/plan.sh   # College of Engineering
```

</details>

### 2.6 Watch, resume and stop

<details>
<summary>See all jobs, runs and the latest epoch</summary>

```bash
bash scripts/ice/status.sh
```

</details>

<details>
<summary>Follow one run's log</summary>

```bash
ls -t logs/fp-swin_daily-*.out | head -3        # newest logs of run swin_daily
tail -f "$(ls -t logs/fp-swin_daily-*.out | head -1)"
```

Logs are `logs/fp-<NAME>-<jobid>.out` in the clone you submitted from. Press Ctrl-C to stop following.

</details>

<details>
<summary>Find out why a job failed</summary>

```bash
grep -h -m3 "Traceback\|Error\|CANCELLED\|DUE TO TIME LIMIT" logs/fp-swin_daily-*.out
sacct -u $USER -S today --format=JobID,JobName%25,State,Elapsed,ExitCode
```

`DUE TO TIME LIMIT` is normal inside a chain: the next job continues the run.

</details>

<details>
<summary>Stop a run</summary>

```bash
scancel -n fp-swin_daily          # every queued and running job of run swin_daily
```

Restart it later with the same `plan.sh` command and `NAME`.

</details>

<details>
<summary>Add more jobs to a running chain</summary>

Run the same command again. The new jobs continue from the saved state:

```bash
NAME=swin_daily CHAIN=2 bash scripts/ice/plan.sh
```

</details>

**Smoke test caveat:** `SMOKE=1` passes the model, data root and `TS`, but not `EXTRA` or
`IT`. It does not reproduce a TESSERA or custom-interval run.

---

## 3. Test a model

Every training run tests itself when its last epoch ends. It reloads the checkpoint with the lowest
validation loss and scores the test fires: the mean over fires of each fire's mean over its windows,
as in the paper. The training log ends with a line in this form:

```
Model F1: <f1>; background F1: <bg f1>; model IoU: <iou>; background IoU: <bg iou>
```

The W&B run summary holds the same scores as `test_f1` and `test_iou`. Only one checkpoint per run
stays on disk, the one with the lowest validation loss.

### Find a run's test F1 and IoU

<details>
<summary>One run's scores</summary>

```bash
NAME=swin_daily_lr0001
grep -h "Model F1" logs/fp-$NAME-*.out
```

No line means the run has not finished, or it was trained with `--no-test-after-train`. Then test it
by hand as below.

</details>

<details>
<summary>Compare every finished run</summary>

```bash
for f in ~/scratch/firepred/runs/*.done; do
  n=$(basename "$f" .done)
  echo "$n: $(grep -h 'Model F1' logs/fp-$n-*.out 2>/dev/null | tail -1)"
done
```

Runs submitted from the other clone have their logs in that clone's `logs/` folder.

</details>

<details>
<summary>Open a run in W&B</summary>

```bash
NAME=swin_daily_lr0001
grep -ho "https://wandb.ai/[^ ]*/runs/[a-z0-9]*" logs/fp-$NAME-*.out | head -1
```

Open the printed link. The Overview tab lists `test_f1` and `test_iou`. W&B run names, such as
`train_pred_swinunetr3d_num_heads_3_hidden_size_36_batchsize_8_dicece_lr-cosine`, do not include your
`NAME` or learning rate, so different runs often share one. Use the link instead of searching by name.

</details>

### Test a chosen checkpoint

Use this when a run stopped before its final test, was trained with `--no-test-after-train`, or
should be scored again after a code change. Each command reads the checkpoint path from the run's
log, then submits a GPU job. Match `-m`, `-ts`, `-it` and the TESSERA flags to the run. `-b` and `-lr`
are required but do not change the scores.

<details>
<summary><b>Plain run</b> (no TESSERA)</summary>

```bash
NAME=swin_daily_lr0001
CKPT=$(grep -h "Best checkpoint" logs/fp-$NAME-*.out | tail -1 | sed 's/.*: //')
echo "$CKPT"                       # must print a .pth path

sbatch -J fp-test-$NAME -p coc-gpu --qos=coc-ice --gres=gpu:1 -c 8 --mem=64G -t 2:00:00 \
  -o logs/%x-%j.out --wrap "source scripts/ice/env.sh && python -m firepred.cli.train -test \
  --checkpoint $CKPT -m swinunetr3d -mode pred -b 8 -lr 0.001 -nh 3 -ed 36 -ts 6 -it 1 \
  --data-root $HOME/scratch/firepred/data-daily --num-workers 6"
```

</details>

<details>
<summary><b>TESSERA run</b></summary>

TESSERA checkpoints on the default torch 2.3.1 need `--trust-checkpoint`. Use it only for
checkpoints your own runs created.

```bash
NAME=swin_tessera_ctx
CKPT=$(grep -h "Best checkpoint" logs/fp-$NAME-*.out | tail -1 | sed 's/.*: //')
echo "$CKPT"

sbatch -J fp-test-$NAME -p coc-gpu --qos=coc-ice --gres=gpu:1 -c 8 --mem=64G -t 2:00:00 \
  -o logs/%x-%j.out --wrap "source scripts/ice/env.sh && python -m firepred.cli.train -test \
  --checkpoint $CKPT --trust-checkpoint --tessera -m swinunetr3d_tessera -mode pred -b 8 -lr 0.001 \
  -nh 3 -ed 36 -ts 6 -it 1 --data-root $HOME/scratch/firepred/data-daily-tessera --num-workers 6"
```

</details>

<details>
<summary><b>Run with no "Best checkpoint" line</b> (stopped before the end)</summary>

Find the checkpoint through the run's W&B id:

```bash
NAME=swin_daily_lr0001
RUN_ID=$(grep -ho "wandb.ai/[^ ]*/runs/[a-z0-9]*" logs/fp-$NAME-*.out | head -1 | sed 's#.*/##')
CKPT=$(ls -t ~/scratch/firepred/wandb/*-$RUN_ID/files/*.pth | head -1)
echo "$CKPT"
```

Then submit the test job from the matching case above.

</details>

<details>
<summary><b>Test interactively on a GPU node</b></summary>

```bash
salloc -p coc-gpu --qos=coc-ice --gres=gpu:1 -c 8 --mem=64G -t 2:00:00
hostname                     # a compute node; if it shows a login node, run: srun --pty bash
source scripts/ice/env.sh
NAME=swin_daily_lr0001
CKPT=$(grep -h "Best checkpoint" logs/fp-$NAME-*.out | tail -1 | sed 's/.*: //')

python -m firepred.cli.train -test --checkpoint "$CKPT" \
  -m swinunetr3d -mode pred -b 8 -lr 0.001 -nh 3 -ed 36 -ts 6 -it 1 \
  --data-root "$HOME/scratch/firepred/data-daily" --num-workers 6

exit
```

</details>

<details>
<summary><b>Read the result of a test job</b></summary>

```bash
NAME=swin_daily_lr0001
grep -h "Model F1" logs/fp-test-$NAME-*.out
ls evaluation_plot/               # prediction plots (change with --plot-dir <path>)
```

A new W&B run named `test_pred_...` also holds `test_f1` and `test_iou`.

</details>

**Always pass `--checkpoint`.** Without it, `-test` takes the newest checkpoint whose file name matches
the model, batch size and tags. Runs with different learning rates share that file name, so it can
pick another run's checkpoint. A checkpoint trained with other inputs is refused with an error naming
the difference. Raise `-t` if a test job hits its time limit.

---

## 4. Reference

### Where things live

| What | Location |
| --- | --- |
| Python environment | `~/scratch/envs/firepred` (`FIREPRED_ENV`) |
| Data, run states, W&B runs | `~/scratch/firepred` (`FIRE_WORK`) |
| Default data root | `~/scratch/firepred/data-daily` (`DATA_ROOT`) |
| Run state and finished marker | `~/scratch/firepred/runs/<NAME>.state` and `<NAME>.done` |
| Checkpoints and W&B files | `~/scratch/firepred/wandb/run-<date>-<run-id>/files/` |
| Job logs | `logs/` in the clone you submitted from |
| Hugging Face cache and token | `~/scratch/.cache/huggingface` |
| W&B project | https://wandb.ai/gt-fri/AFBAPred |

All paths are set in `scripts/ice/env.sh`. Everything large lives in `~/scratch` because the home
directory is small.

### Partitions, QOS and accounts

<details>
<summary>Which partitions accept my jobs, and under which QOS?</summary>

```bash
bash scripts/ice/partitions.sh                                  # every partition
bash scripts/ice/partitions.sh coc-gpu ice-gpu pace-gpu         # only these
GPU_TYPE=h100 bash scripts/ice/partitions.sh coc-gpu ice-gpu    # can I get an H100 there?
```

The `QOS` column names the QOS that worked; pass it as `QOS=...`. `default` means pass
`QOS=default`, so `plan.sh` omits `--qos` and Slurm picks your default. One job carries one QOS, so
partitions that need different QOS cannot share one `PARTITION` list.

</details>

<details>
<summary>My accounts and QOS, and what each QOS allows</summary>

```bash
sacctmgr show assoc user=$USER format=account%15,partition%15,qos%40
sacctmgr show qos name=coc-ice format=name%12,maxwall,maxtrespu%30,maxjobspu,maxsubmitpu
scontrol show partition coc-gpu        # AllowAccounts, AllowQos, MaxTime
```

An empty partition in the first command means any partition.

</details>

<details>
<summary>Would this exact job be accepted, and when would it start?</summary>

```bash
sbatch --test-only -p coc-gpu --qos=coc-ice --gres=gpu:1 -t 1:00:00 --wrap true
```

It submits nothing. It fails with Slurm's reason, such as "Invalid qos specification", when you may
not use that partition, QOS, account or GPU model.

</details>

<details>
<summary>Submit a script by hand instead of through <code>plan.sh</code></summary>

Options on the command line override the script's `#SBATCH` lines. `QOS` and `CPU_QOS` are read by
`plan.sh` only, so pass `--qos` yourself:

```bash
NAME=swin_daily sbatch -p coc-gpu,ice-gpu --qos=coc-ice scripts/ice/train.sbatch
DAILY_REPO=SamuelWu318/ts-satfire-daily sbatch -p ice-cpu --qos=coc-ice scripts/ice/data.sbatch
```

</details>

### Nodes and GPUs

<details>
<summary>How full is each partition, and which GPUs does it have?</summary>

```bash
sinfo -s
sinfo -p coc-gpu,ice-gpu,pace-gpu -o "%.12P %.5a %.10l %.5D %.8t %G"
sinfo -o "%.12P %G" | sort -u                     # GPU models per partition
pace-check-queue coc-gpu                          # PACE's own view of a queue
```

</details>

<details>
<summary>Which nodes have free GPUs right now?</summary>

```bash
sinfo -p coc-gpu,ice-gpu,pace-gpu -t idle,mix -N -o "%N %P %t %G"
sinfo -N -p coc-gpu -O "NodeList:24,StateCompact:8,Gres:28,GresUsed:34,CPUsState:16,Memory:9"
```

`CPUsState` reads allocated/idle/other/total. A `mix` node may still have free GPUs; `GresUsed`
shows how many are taken. `gpu:h100:4` means four H100s per node, and `h100` is what `GPU_TYPE` takes.

</details>

<details>
<summary>One node in detail</summary>

```bash
scontrol show node <node-name>
squeue -w <node-name>                 # what runs on it
```

</details>

### Watching jobs

<details>
<summary>My jobs, and when pending ones will start</summary>

```bash
bash scripts/ice/status.sh
squeue -u $USER -o "%.10i %.20j %.9P %.8T %.10M %.12l %.24R"
squeue -u $USER --start
```

</details>

<details>
<summary>Why is a job pending?</summary>

```bash
scontrol show job <jobid>             # Reason, Partition, NodeList
```

| Reason | Meaning |
| --- | --- |
| `Priority`, `Resources` | Waiting its turn or for a free node. Nothing to fix |
| `Dependency` | Waiting for an earlier job of the chain |
| `DependencyNeverSatisfied` | The job it waits for failed, such as the data job or smoke test. `scancel` it |
| `QOSMaxGRESPerUser`, `AssocGrpGRES` | You already use the most GPUs your QOS allows. It starts when one ends |
| `ReqNodeNotAvail` | The nodes are reserved, often for maintenance |
| `PartitionTimeLimit` | `-t` is longer than the partition allows |

</details>

<details>
<summary>Finished jobs: state, time, memory, exit code</summary>

```bash
sacct -u $USER -S today --format=JobID,JobName%20,Partition,State,Elapsed,MaxRSS,ExitCode
seff <jobid>                          # how well a finished job used its CPUs and memory
```

</details>

<details>
<summary>How busy is a partition, and what is my priority?</summary>

```bash
squeue -p coc-gpu -t PENDING | wc -l
squeue -p coc-gpu -t RUNNING | wc -l
sprio -u $USER
sshare -u $USER
```

</details>

<details>
<summary>Cancel jobs</summary>

```bash
scancel <jobid>              # one job
scancel -n fp-<NAME>         # every job of one run's chain
scancel -u $USER -t PENDING  # all of my pending jobs
scancel -u $USER             # all of my jobs
```

</details>

### Interactive GPU session

<details>
<summary>Get a GPU node for an hour</summary>

```bash
salloc -p coc-gpu --qos=coc-ice --gres=gpu:1 -c 8 --mem=32G -t 1:00:00
hostname                     # a compute node; if it is still a login node, run: srun --pty bash
nvidia-smi
source scripts/ice/env.sh    # this repository's Python environment
exit                         # give the node back
```

</details>

### Python environment

<details>
<summary>What is installed</summary>

```bash
source scripts/ice/env.sh
python -V
pip list | grep -i -E "torch|monai|numpy|wandb|huggingface"
```

</details>

<details>
<summary>Run the tests</summary>

```bash
sbatch scripts/ice/test.sbatch                                  # CPU node, without slow tests
SLOW=1 sbatch scripts/ice/test.sbatch                           # include slow end-to-end tests
PYTEST_ARGS="tests/test_golden.py -x" sbatch scripts/ice/test.sbatch
NAME=swin_daily SMOKE=1 bash scripts/ice/plan.sh                # GPU smoke test before training
```

</details>

### Storage

<details>
<summary>My quota and what uses it</summary>

```bash
pace-quota
du -sh ~/scratch/* | sort -h
du -sh ~/scratch/firepred/* | sort -h
```

</details>
