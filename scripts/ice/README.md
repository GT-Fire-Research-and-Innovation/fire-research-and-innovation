# PACE ICE guide (`daily-store` branch)

Everything needed to train and test models on Georgia Tech's PACE ICE cluster with the scripts in
`scripts/ice/`. Run every command from the repository root on an ICE login node unless a step says
otherwise. Click a ▸ line to open its commands.

1. [Quickstart](#1-quickstart): one-time setup and data download, up to the point of launching a model.
2. [Run a model](#2-run-a-model): the pieces of a run command, every parameter, and examples.
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
bash scripts/ice/partitions.sh ice-gpu coe-gpu ice-cpu
```

Read the `ACCESS` and `QOS` columns. The scripts default to GPU partition `ice-gpu`, CPU partition
`ice-cpu` and your Slurm default QOS. Commands in this guide pass no `--qos`, so they use that default too.

<details>
<summary><code>ice-gpu</code> and <code>ice-cpu</code> say <code>yes</code> with QOS <code>default</code></summary>

Nothing to set. Every command in this guide works as written.

</details>

<details>
<summary>A row says <code>no</code>, or shows a named QOS such as <code>coc-ice</code></summary>

`ice-gpu` says `no` but `coe-gpu` says `yes`: use `coe-gpu` instead.

```bash
NAME=swin_daily PARTITION=coe-gpu bash scripts/ice/plan.sh
```

A row says `yes` only under a named QOS: pass it as `QOS`. Set `CPU_QOS=default` when the CPU row
says `default`, because `CPU_QOS` otherwise copies `QOS`.

```bash
NAME=swin_daily QOS=coc-ice CPU_QOS=default bash scripts/ice/plan.sh
```

In hand-written `sbatch` and `salloc` commands in this guide, swap `-p ice-gpu` for the partition that
works, and add `--qos=<name>` only when that partition needs a named QOS. Run
`bash scripts/ice/partitions.sh` with no arguments to see every partition.

</details>

### Step 5: download a dataset

Training reads a local folder, the data root. A download is one command built from pieces: start
from the base command, then apply the pieces you want. Files already downloaded are kept, so run the
same command again if it stops.

**Base: the daily dataset.**

```bash
source scripts/ice/env.sh

python -m firepred.cli.dataset \
  --window-repo SamuelWu318/ts-satfire-daily \
  download -ts 6 -it 1 \
  --data-root "$HOME/scratch/firepred/data-daily"
```

**Pieces.** Each piece changes the base command as the table says.

| Piece | Change to the base command | What it adds |
| --- | --- | --- |
| TESSERA | Add `--tessera-repo SamuelWu318/ts-tesserafire` before `download`, add `--tessera` after `-it 1`, and use the data root `data-daily-tessera` | TESSERA embeddings in `<data root>/tessera`. Fires without prior-year TESSERA are left out |
| Compute node | Run `salloc -p ice-cpu -c 8 --mem=64G -t 4:00:00` first, then the command, then `exit` | Keeps a long download alive after you log out |

Rules for putting pieces together:

- Options about repositories (`--window-repo`, `--tessera-repo`) go before the word `download`. Options
  about the download (`-ts`, `-it`, `--tessera`, `--data-root`) go after it.
- Keep `-ts 6 -it 1`. The daily dataset holds 6-day windows one day apart, and training on this branch
  uses the same `TS=6 IT=1`.
- Give a TESSERA download its own data root. It leaves out fires, so it is not the same dataset as
  `data-daily`.
- After `salloc`, check that `hostname` names a compute node. If it shows a login node, run
  `srun --pty bash`.

The ready-made combinations:

<details>
<summary><b>Daily</b> (base only, recommended to start)</summary>

```bash
source scripts/ice/env.sh

python -m firepred.cli.dataset \
  --window-repo SamuelWu318/ts-satfire-daily \
  download -ts 6 -it 1 \
  --data-root "$HOME/scratch/firepred/data-daily"
```

Train on it with `DATA_ROOT="$HOME/scratch/firepred/data-daily"`, which is also the default.

</details>

<details>
<summary><b>Daily + TESSERA</b></summary>

```bash
source scripts/ice/env.sh

python -m firepred.cli.dataset \
  --window-repo SamuelWu318/ts-satfire-daily \
  --tessera-repo SamuelWu318/ts-tesserafire \
  download -ts 6 -it 1 --tessera \
  --data-root "$HOME/scratch/firepred/data-daily-tessera"
```

Train on it with `DATA_ROOT="$HOME/scratch/firepred/data-daily-tessera"` and `--tessera` in `EXTRA`.

</details>

Zone files and fire split training are on the `fire-split-zones` branch only.

Other dataset tasks:

<details>
<summary><b>Convert the processed 6-day windows into a daily dataset</b> (slow; only if the daily repo is unavailable)</summary>

```bash
source scripts/ice/env.sh
mkdir -p "$HOME/scratch/firepred/tmp"

python -m firepred.cli.dataset \
  --window-repo SamuelWu318/ts-satfire-processed-6-1 \
  --work-dir "$HOME/scratch/firepred/tmp" \
  convert-daily -ts 6 -it 1 \
  --dest-root "$HOME/scratch/firepred/data-daily-converted"
```

It downloads one fire at a time into `--work-dir`, converts it and checks it reads back identical.
Fires already converted are skipped when you run it again. Add `--dest-repo <your-hf-user>/ts-satfire-daily`
after `--dest-root` to also upload the result. Your token needs write access to that repo.

</details>

<details>
<summary><b>Check that a download finished</b></summary>

```bash
D="$HOME/scratch/firepred/data-daily"            # or data-daily-tessera
ls "$D"                                           # train  val  test  window_config.json  (tessera)
for s in train val test; do echo "$s: $(ls "$D/$s" | wc -l) files"; done
ls "$D/tessera" 2>/dev/null | wc -l              # TESSERA files, on a TESSERA data root
du -sh "$D"
```

A finished download has `window_config.json` and files in all three splits. Run the same download
command again when it stopped early.

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
source scripts/ice/env.sh

python -m firepred.cli.dataset \
  --window-repo SamuelWu318/ts-satfire-daily \
  download -ts 6 -it 1 \
  --data-root "$HOME/scratch/firepred/data-daily-v2"
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
an optional GPU smoke test, then a chain of training jobs that continue one another. You build the
command from pieces: settings written before `bash scripts/ice/plan.sh`.

```bash
NAME=<run name> <piece> <piece> ... EXTRA="<training flags of every piece>" bash scripts/ice/plan.sh
bash scripts/ice/status.sh          # watch it
```

1. Pick one piece from each table in [2.1](#21-pieces) you care about. Skip a table to keep its default.
2. Write the settings in any order before `bash scripts/ice/plan.sh`.
3. Join the training flags of all pieces into one `EXTRA="..."`. A second `EXTRA=` replaces the first
   instead of adding to it.

[2.2](#22-every-parameter) lists every setting and flag. [2.3](#23-examples) has finished commands.

### 2.1 Pieces

**Name** (required)

| Piece | Write |
| --- | --- |
| New run | `NAME=<new name>`, such as `NAME=swin_daily_lr0001_cosine`. `ls ~/scratch/firepred/runs/` shows names already used |
| Continue a stopped run | The same `NAME` and the same settings as before. It resumes after the last finished epoch |

Each run saves its state to `~/scratch/firepred/runs/$NAME.state` after every epoch. A finished run
leaves `$NAME.done`, and later jobs with that name exit at once. Resuming is refused when the model,
loss, learning-rate schedule, TESSERA use, epochs, learning rate, batch size, seed, `TS` or `IT`
differ from the saved state.

**Dataset.** Download it first ([Step 5](#step-5-download-a-dataset)). A data root without
`window_config.json` makes `plan.sh` queue its own slow conversion job.

| Piece | Write | Notes |
| --- | --- | --- |
| Daily (default) | nothing, or `DATA_ROOT="$HOME/scratch/firepred/data-daily"` | |
| TESSERA | `DATA_ROOT="$HOME/scratch/firepred/data-daily-tessera"` **and** `--tessera` in `EXTRA` | Both halves are needed. Without `--tessera` the TESSERA files are ignored |
| Converted | `DATA_ROOT="$HOME/scratch/firepred/data-daily-converted"` | A dataset you converted yourself |

**Model.** One `MODEL=<name>`. [The model table](#models) lists each model's options.

| Piece | Write | Notes |
| --- | --- | --- |
| SwinUNETR-3D (default) | nothing, or `MODEL=swinunetr3d` | With the TESSERA piece, TESSERA joins the input as extra channels |
| SwinUNETR-3D, learned day weights | `MODEL=swinunetr3d_tpool` | |
| SwinUNETR-3D, TESSERA as context | `MODEL=swinunetr3d_tessera` | **Needs the TESSERA piece** |
| TESSERA context and day weights | `MODEL=swinunetr3d_tessera_tpool` | **Needs the TESSERA piece** |
| UNETR-3D | `MODEL=unetr3d` | |
| 3-D U-Net | `MODEL=unet3d` | |
| Attention U-Net | `MODEL=attunet` | Default loss `dice`; every other model defaults to `dicece` |
| ConvLSTM U-Net | `MODEL=convlstm` | |
| ResNet-18 U-Net | `MODEL=res18unet` | |

The two TESSERA-context models stop about one second after W&B starts when `--tessera` is missing,
with `swinunetr3d_tessera needs TESSERA inputs (train with --tessera); got 43 input channels`.

**Training.** Settings go before `bash`, flags go inside `EXTRA`.

| Piece | Write |
| --- | --- |
| Learning rate | `LR=0.0001` (default `0.001`) |
| Batch size | `BATCH=16` (default `8`). Larger batches need more GPU memory, so pair them with `GPU_TYPE=h100` or `h200` |
| Epochs | `--epochs 200` (default `100`) |
| Seed | `-seed 1` (default `42`) |
| Cosine decay | `--lr-schedule cosine --lr-min 0.000001` |
| Plateau decay | `--lr-schedule plateau --lr-patience 5 --lr-decay-factor 0.5 --lr-min 0.000001` |
| Dice loss | `--loss dice` |
| Dice + cross-entropy, fixed weights | `--loss dicece --ce-weight 1 10` (background 1, fire 10) |
| Dice + cross-entropy, weights from the data | `--loss dicece --ce-weight auto` (the default for `dicece`) |
| Test scores in W&B during training | `--test-every 10`. Charts only: the kept checkpoint is still chosen by validation loss |
| No final test | `--no-test-after-train` |
| Keep checkpoints in W&B | `--upload-checkpoints` |
| Model option | `--model-option KEY=VALUE`, repeatable. See [the model table](#models) |

On this branch `TS` and `IT` must match `DATA_ROOT/window_config.json`, so the daily dataset needs the
defaults `TS=6 IT=1`. Other window settings need the `fire-split-zones` branch.

**Length and checks**

| Piece | Write | Notes |
| --- | --- | --- |
| Queue more jobs | `CHAIN=5` (default `3`) | Each job runs up to 16 hours and continues the last. Raise it for long runs |
| Quick trial | `CHAIN=1` and `--epochs 2` in `EXTRA` | Finishes in one short job |
| GPU smoke test first | `SMOKE=1` | One training step on a GPU before the chain starts. It ignores `EXTRA` and `IT`, so **leave it off for TESSERA runs**: a TESSERA-context model fails the smoke test, and training then never starts |

**Where it runs.** [Step 4](#step-4-check-which-partitions-accept-your-jobs) shows which partitions you may use.

| Piece | Write | Notes |
| --- | --- | --- |
| `ice-gpu` (default) | nothing | Your default QOS |
| First free GPU of several partitions | `PARTITION=ice-gpu,coe-gpu` | All partitions in one list must accept the same QOS |
| College of Engineering H100/H200 nodes | `PARTITION=coe-gpu` | |
| One GPU model | `GPU_TYPE=h100` | Also `h200`, `a100`. `ice-gpu` has AMD `mi210` and `r9700` GPUs this CUDA build cannot use |
| A named QOS | `QOS=coc-ice CPU_QOS=default` | `CPU_QOS=default` keeps the data job on your default QOS |
| Another Slurm account | `ACCOUNT=<account>` | |

`plan.sh` tests each listed partition with `sbatch --test-only` and drops one that refuses the job.

### 2.2 Every parameter

#### Settings before `bash scripts/ice/plan.sh`

| Setting | Default | Meaning |
| --- | --- | --- |
| `NAME` | required | Run name. Also the job name `fp-$NAME` and the log `logs/fp-$NAME-<jobid>.out` |
| `MODEL` | `swinunetr3d` | Model, from [the model table](#models) |
| `DATA_ROOT` | `~/scratch/firepred/data-daily` | Data root training reads |
| `BATCH` | `8` | Batch size |
| `LR` | `0.001` | Learning rate |
| `TS` | `6` | Input days per window. Must match `window_config.json`: `6` on the daily dataset |
| `IT` | `1` | Days between window starts. Must match `window_config.json`: `1` on the daily dataset |
| `EXTRA` | empty | Every other training flag, in one quoted string. See the flag table below |
| `CHAIN` | `3` | Training jobs queued in a row, each up to 16 hours |
| `SMOKE` | `0` | `1` adds a one-step GPU check before training. Not for TESSERA runs |
| `PARTITION` | `ice-gpu` | GPU partition, or a comma list |
| `GPU_TYPE` | any GPU | One GPU model, such as `h100`, `h200` or `a100` |
| `QOS` | your default | QOS of the GPU jobs. Set it only for a named QOS |
| `CPU_PARTITION` | `ice-cpu` | Partition of the data job |
| `CPU_QOS` | same as `QOS` | QOS of the data job. `default` means your default |
| `ACCOUNT` | your default | Slurm account of every job |
| `FIRE_WORK` | `~/scratch/firepred` | Folder for data, run states and W&B files |
| `FIREPRED_ENV` | `~/scratch/envs/firepred` | Python environment |

Only when `plan.sh` queues a data job, because `DATA_ROOT` has no `window_config.json`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `DAILY_REPO` | unset | Download daily archives from this repo, such as `SamuelWu318/ts-satfire-daily`, instead of converting |
| `SRC_REPO` | `SamuelWu318/ts-satfire-processed-6-1` | Window repo the conversion reads |
| `DEST_REPO` | unset | Also upload the converted archives to this repo |

Each training job gets one GPU, 8 CPUs, 64 GB of memory and 16 hours. `train.sbatch` always passes
`-mode pred -nh 3 -ed 36 --num-workers 6` and `--resume` with the run state.
Flags in `EXTRA` come last, so `-nh`, `-ed` and `--num-workers` in `EXTRA` override those.

#### Training flags for `EXTRA`

| Flag | Default | Meaning |
| --- | --- | --- |
| **Length and repeatability** | | |
| `--epochs N` | `100` | Epochs of the whole run, across every job of the chain |
| `-seed N` | `42` | Seed of the weights, shuffling and augmentation |
| `-r N` | `1` | Run number, recorded only |
| `--num-workers N` | `6` | Processes that load data |
| `--device auto\|cpu\|cuda\|tpu` | `auto` | `auto` takes the GPU when one is visible |
| **Loss** | | |
| `--loss dice\|dicece` | per model | `dice` for `attunet`, `dicece` for the rest. `dicece` is MONAI's DiceCELoss |
| `--ce-weight auto` or `--ce-weight BG FIRE` | `auto` | `dicece` class weights. `auto` is 1 and the training split's background-to-fire pixel ratio |
| `--lambda-dice X` | `1.0` | Weight of the Dice term |
| `--lambda-ce X` | `1.0` | Weight of the cross-entropy term |
| **Learning-rate schedule** | | |
| `--lr-schedule none\|plateau\|cosine` | `none` | `none` keeps the rate constant, as in the paper |
| `--lr-min X` | `0.000001` | Lowest rate, for `plateau` and `cosine` |
| `--lr-decay-factor X` | `0.5` | `plateau`: multiply the rate by this |
| `--lr-patience N` | `5` | `plateau`: epochs without a lower validation loss before decaying |
| **Inputs** | | |
| `--normalization window-config\|paper` | `window-config` | Input statistics: the training split's, from `window_config.json`, or the released code's fixed numbers |
| `--unzip-dir off\|auto\|PATH` | `off` | Unzip the archives once for faster reads. `auto` puts them beside the data root |
| `-nc N` | derived | Check the input channels: `43`, or `172` with `--tessera` |
| **TESSERA** | | |
| `--tessera` | off | Add TESSERA. Required by `swinunetr3d_tessera` and `swinunetr3d_tessera_tpool` |
| `--tessera-dir PATH` | `<DATA_ROOT>/tessera` | Another folder of TESSERA files |
| `--tessera-cells-per-pixel K` | `1` | TESSERA cells per VIIRS pixel side. Must match how the files were built |
| `--tessera-min-coverage X` | `0.5` | TESSERA pixels below this coverage are zeroed and masked |
| **Model** | | |
| `-nh N` | `3` | Attention heads of the SwinUNETR-3D models |
| `-ed N` | `36` | Feature size of the SwinUNETR-3D models |
| `--swin-temporal-downsample` | off | SwinUNETR-3D models: also halve the days in patch merging, as the released code does |
| `--unetr-version v0\|v1` | `v1` | `unetr3d` sizes. `v1` is the paper's |
| `--model-option KEY=VALUE` | none | Model setting, repeatable. See [the model table](#models) |
| **Testing and logging** | | |
| `--no-test-after-train` | off | Skip the final test. By default the best checkpoint scores the test fires after training |
| `--test-every N` | `0` | Also score the test fires every N epochs, for W&B charts only |
| `--test-csv PATH` | `roi/us_fire_2021_out_new.csv` | Fire list of the test scoring |
| `--plot-dir PATH` | `evaluation_plot` | Folder of prediction plots from the final test |
| `--upload-checkpoints` | off | Store each best checkpoint as a W&B artifact |
| `--wandb-user NAME` | `gt-fri` | W&B team. The project is `AFBAPred` |

Set through a setting instead of `EXTRA`: `-m` (`MODEL`), `-b` (`BATCH`), `-lr` (`LR`), `-ts` (`TS`),
`-it` (`IT`), `--data-root` (`DATA_ROOT`) and `--resume` (`NAME`).
`-test`, `--checkpoint`, `--trust-checkpoint` and `--load-epoch` are for testing, in [section 3](#3-test-a-model).
`python -m firepred.cli.train --help` prints every flag, after `source scripts/ice/env.sh`.

#### Models

| `MODEL` | Model | TESSERA | Default loss | `--model-option` (default) |
| --- | --- | --- | --- | --- |
| `swinunetr3d` | SwinUNETR-3D, the paper's main model | Optional, as 129 extra input channels | `dicece` | `attn_version=v1` (or `v2`) |
| `swinunetr3d_tpool` | SwinUNETR-3D that learns a weight per input day instead of averaging the days | Optional, as extra input channels | `dicece` | `attn_version=v1` |
| `swinunetr3d_tessera` | SwinUNETR-3D whose skip connections are conditioned on TESSERA at every scale | **Required** | `dicece` | `context_channels=32`, `attn_version=v1` |
| `swinunetr3d_tessera_tpool` | `swinunetr3d_tessera` with learned day weights | **Required** | `dicece` | `context_channels=32`, `attn_version=v1` |
| `unetr3d` | UNETR-3D | Optional | `dicece` | none; use `--unetr-version` |
| `unet3d` | 3-D U-Net | Optional | `dicece` | none |
| `attunet` | Attention U-Net | Optional | `dice` | none |
| `convlstm` | ConvLSTM U-Net: a shared 2-D encoder with a ConvLSTM over the days | Optional | `dicece` | `width=16` |
| `res18unet` | ResNet-18 U-Net over the days stacked as channels | Optional | `dicece` | `reduce_width=64`, `full_width=32` |

`-nh`, `-ed` and `--swin-temporal-downsample` apply to the four SwinUNETR-3D models only.

### 2.3 Examples

Each example names its pieces, then gives the command.

<details open>
<summary><b>Paper settings</b>: daily, SwinUNETR-3D, rate 0.001 constant, 100 epochs</summary>

Pieces: name, daily (default), SwinUNETR-3D (default), epochs.

```bash
NAME=swin_paper MODEL=swinunetr3d BATCH=8 LR=0.001 EXTRA="--epochs 100" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><b>Quick trial</b> with a GPU smoke test</summary>

Pieces: name, quick trial, GPU smoke test first.

```bash
NAME=trial_swin CHAIN=1 SMOKE=1 EXTRA="--epochs 2" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><b>Cosine decay, H100 only</b></summary>

Pieces: name, learning rate, epochs, cosine decay, first free GPU of several partitions, one GPU model.

```bash
NAME=swin_lr0001_cosine_h100 LR=0.0001 PARTITION=ice-gpu,coe-gpu GPU_TYPE=h100 \
EXTRA="--epochs 100 --lr-schedule cosine --lr-min 0.000001" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><b>TESSERA as context</b>: <code>swinunetr3d_tessera</code></summary>

Pieces: name, TESSERA (both halves), TESSERA-context model, epochs.

```bash
NAME=swin_tessera_ctx MODEL=swinunetr3d_tessera DATA_ROOT="$HOME/scratch/firepred/data-daily-tessera" \
EXTRA="--tessera --epochs 100" bash scripts/ice/plan.sh
```

</details>

<details>
<summary><b>Learning-rate sweep</b>: one run per value</summary>

Pieces: name, learning rate, epochs, cosine decay, repeated in a loop with a distinct `NAME` each time.

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
<summary><b>Another model with a model option</b>: wider ConvLSTM, weighted loss</summary>

Pieces: name, ConvLSTM, model option, Dice + cross-entropy with fixed weights.

```bash
NAME=convlstm_w32_ce10 MODEL=convlstm \
EXTRA="--model-option width=32 --loss dicece --ce-weight 1 10" bash scripts/ice/plan.sh
```

</details>

### 2.4 Watch, resume and stop

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
grep -l "no GPU visible" logs/fp-swin_daily-*.out       # jobs that got no NVIDIA GPU, such as an AMD node
sacct -u $USER -S today --format=JobID,JobName%25,State,Elapsed,ExitCode
```

`DUE TO TIME LIMIT` is normal inside a chain: the next job continues the run. A log with
`no GPU visible` ran on the CPU; cancel it and resubmit with `GPU_TYPE=h100` or another NVIDIA model.
A W&B run that ends after about a second means the settings were wrong, and every job of the chain
fails the same way. The most common case is a TESSERA-context model without `--tessera` in `EXTRA`.

</details>

<details>
<summary>Continue a run that stopped (time limit, failure, cancelled)</summary>

Submit the same command with the same `NAME` and settings. It resumes after the last finished epoch:

```bash
NAME=swin_daily_lr0001_cosine LR=0.0001 EXTRA="--lr-schedule cosine" CHAIN=2 bash scripts/ice/plan.sh
```

</details>

<details>
<summary>Add more jobs to a running chain</summary>

Run the same command again. The new jobs continue from the saved state:

```bash
NAME=swin_daily CHAIN=2 bash scripts/ice/plan.sh
```

</details>

<details>
<summary>Stop a run</summary>

```bash
scancel -n fp-swin_daily          # every queued and running job of run swin_daily
```

Restart it later with the same `plan.sh` command and `NAME`.

</details>

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
should be scored again after a code change. Like a training command, a test command is a base plus
pieces that match how the run was trained.

**Base: find the checkpoint, then submit a GPU job.**

```bash
NAME=swin_daily_lr0001
CKPT=$(grep -h "Best checkpoint" logs/fp-$NAME-*.out | tail -1 | sed 's/.*: //')
echo "$CKPT"                       # must print a .pth path

sbatch -J fp-test-$NAME -p ice-gpu --gres=gpu:1 -c 8 --mem=64G -t 2:00:00 \
  -o logs/%x-%j.out --wrap "source scripts/ice/env.sh && python -m firepred.cli.train -test \
  --checkpoint $CKPT -m swinunetr3d -mode pred -b 8 -lr 0.001 -nh 3 -ed 36 -ts 6 -it 1 \
  --data-root $HOME/scratch/firepred/data-daily --num-workers 6"
```

**Pieces.** Change the base where the run's training differed. `-b` and `-lr` are required but do
not change the scores.

| The run was trained with | Change in the test command |
| --- | --- |
| Another `MODEL` | `-m <model>` |
| `-nh`, `-ed`, `--swin-temporal-downsample`, `--unetr-version` or `--model-option` in `EXTRA` | The same flags and values |
| TESSERA | Add `--tessera --trust-checkpoint`, use `--data-root $HOME/scratch/firepred/data-daily-tessera`, and add `--tessera-cells-per-pixel K` if training used it |
| Another GPU partition or QOS | `-p <partition>`, and `--qos=<name>` for a named QOS |
| A test that hits its time limit | A larger `-t` |

TESSERA checkpoints on the default torch 2.3.1 need `--trust-checkpoint`. Use it only for
checkpoints your own runs created. A checkpoint trained with other inputs is refused with an error
naming the difference.

<details>
<summary><b>Example: a TESSERA-context run</b></summary>

Pieces: another `MODEL`, TESSERA.

```bash
NAME=swin_tessera_ctx
CKPT=$(grep -h "Best checkpoint" logs/fp-$NAME-*.out | tail -1 | sed 's/.*: //')
echo "$CKPT"

sbatch -J fp-test-$NAME -p ice-gpu --gres=gpu:1 -c 8 --mem=64G -t 2:00:00 \
  -o logs/%x-%j.out --wrap "source scripts/ice/env.sh && python -m firepred.cli.train -test \
  --checkpoint $CKPT --trust-checkpoint --tessera -m swinunetr3d_tessera -mode pred -b 8 -lr 0.001 \
  -nh 3 -ed 36 -ts 6 -it 1 --data-root $HOME/scratch/firepred/data-daily-tessera --num-workers 6"
```

</details>

<details>
<summary><b>Run with no "Best checkpoint" line</b> (stopped before the end)</summary>

Find the checkpoint through the run's W&B id, then use it as `CKPT` in the base command:

```bash
NAME=swin_daily_lr0001
RUN_ID=$(grep -ho "wandb.ai/[^ ]*/runs/[a-z0-9]*" logs/fp-$NAME-*.out | head -1 | sed 's#.*/##')
CKPT=$(ls -t ~/scratch/firepred/wandb/*-$RUN_ID/files/*.pth | head -1)
echo "$CKPT"
```

</details>

<details>
<summary><b>Test interactively on a GPU node</b></summary>

```bash
salloc -p ice-gpu --gres=gpu:1 -c 8 --mem=64G -t 2:00:00
hostname                     # a compute node; if it shows a login node, run: srun --pty bash
source scripts/ice/env.sh
NAME=swin_daily_lr0001
CKPT=$(grep -h "Best checkpoint" logs/fp-$NAME-*.out | tail -1 | sed 's/.*: //')

python -m firepred.cli.train -test --checkpoint "$CKPT" \
  -m swinunetr3d -mode pred -b 8 -lr 0.001 -nh 3 -ed 36 -ts 6 -it 1 \
  --data-root "$HOME/scratch/firepred/data-daily" --num-workers 6

exit
```

The same pieces apply.

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
pick another run's checkpoint.

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
bash scripts/ice/partitions.sh ice-gpu coe-gpu ice-cpu          # only these
GPU_TYPE=h100 bash scripts/ice/partitions.sh ice-gpu coe-gpu    # can I get an H100 there?
```

The `QOS` column names the QOS that worked. `default` needs no setting: the scripts already use
your Slurm default. A named QOS, such as `coc-ice`, goes in `QOS=coc-ice`. One job carries one QOS, so
partitions that need different QOS cannot share one `PARTITION` list.

</details>

<details>
<summary>My accounts and QOS, and what each QOS allows</summary>

```bash
sacctmgr show assoc user=$USER format=account%15,partition%15,qos%40,defaultqos%15
sacctmgr show qos name=<qos> format=name%12,maxwall,maxtrespu%30,maxjobspu,maxsubmitpu   # a QOS from the line above
scontrol show partition ice-gpu        # AllowAccounts, AllowQos, MaxTime
```

An empty partition in the first command means any partition.

</details>

<details>
<summary>Would this exact job be accepted, and when would it start?</summary>

```bash
sbatch --test-only -p ice-gpu --gres=gpu:1 -t 1:00:00 --wrap true
```

It submits nothing. It fails with Slurm's reason, such as "Invalid qos specification", when you may
not use that partition, QOS, account or GPU model.

</details>

<details>
<summary>Submit a script by hand instead of through <code>plan.sh</code></summary>

Options on the command line override the script's `#SBATCH` lines. With none, the script's partition
and your default QOS apply. `QOS` and `CPU_QOS` are read by `plan.sh` only, so add `--qos=<name>`
yourself when you need a named QOS:

```bash
NAME=swin_daily sbatch -p ice-gpu,coe-gpu scripts/ice/train.sbatch
```

</details>

### Nodes and GPUs

<details>
<summary>How full is each partition, and which GPUs does it have?</summary>

```bash
sinfo -s
sinfo -p ice-gpu,coe-gpu -o "%.12P %.5a %.10l %.5D %.8t %G"
sinfo -o "%.12P %G" | sort -u                     # GPU models per partition
pace-check-queue ice-gpu                          # PACE's own view of a queue
```

</details>

<details>
<summary>Which nodes have free GPUs right now?</summary>

```bash
sinfo -p ice-gpu,coe-gpu -t idle,mix -N -o "%N %P %t %G"
sinfo -N -p ice-gpu -O "NodeList:24,StateCompact:8,Gres:28,GresUsed:34,CPUsState:16,Memory:9"
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
squeue -p ice-gpu -t PENDING | wc -l
squeue -p ice-gpu -t RUNNING | wc -l
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
salloc -p ice-gpu --gres=gpu:1 -c 8 --mem=32G -t 1:00:00
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
