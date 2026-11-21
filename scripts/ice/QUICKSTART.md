# PACE ICE quickstart

From nothing to a training run on Georgia Tech's PACE ICE cluster, in order. Every command runs on an
ICE login node from the repository root unless a step says otherwise. The full guide, with testing a
model and every option, is [README.md](README.md).

1. [Log in](#1-log-in)
2. [Get the code](#2-get-the-code)
3. [Set up Python and log in to Hugging Face and W&B](#3-set-up-python-and-log-in-to-hugging-face-and-wb)
4. [Download the two datasets](#4-download-the-two-datasets)
5. [Run a model](#5-run-a-model)
6. [Every argument explained](#6-every-argument-explained)
7. [Optional parameters](#7-optional-parameters)
8. [Watch and stop runs](#8-watch-and-stop-runs)

---

## Before you start

| You need | Where | For |
| --- | --- | --- |
| GT account with ICE access | Your course or PACE | Logging in and running jobs |
| GT VPN (GlobalProtect) | https://vpn.gatech.edu | Reaching ICE from off campus |
| GitHub access to the repo | `GT-Fire-Research-and-Innovation/fire-research-and-innovation` | Cloning. Use a personal access token (https://github.com/settings/tokens) when `git` asks for a password |
| Hugging Face token (read) | https://huggingface.co/settings/tokens | Downloading the datasets. Open each dataset page while logged in and request access if it is gated |
| Weights & Biases API key | https://wandb.ai/authorize | Logging runs to team `gt-fri`, project `AFBAPred`. Ask a team admin for an invite |

Paste tokens only into the terminal prompts below, never into a file in the repo or a chat.

## 1. Log in

Off campus, connect the GT VPN first: open GlobalProtect, portal `vpn.gatech.edu`, and sign in with
your GT account. On campus, skip this.

```bash
ssh <gt-username>@login-ice.pace.gatech.edu
```

No terminal at hand: in a browser, ICE OnDemand > Clusters > ICE Shell Access opens the same shell.

## 2. Get the code

Once:

```bash
cd ~
git clone https://github.com/GT-Fire-Research-and-Innovation/fire-research-and-innovation.git
cd fire-research-and-innovation
```

Every later session:

```bash
cd ~/fire-research-and-innovation
git pull
```

Pulling never touches your environment, data or runs, which live in `~/scratch`. Never delete the
clone or `~/scratch` to update: pull, and rerun step 3 only when `scripts/ice/setup.sh` changed
(`git diff --stat ORIG_HEAD HEAD` lists what a pull changed).

## 3. Set up Python and log in to Hugging Face and W&B

Once:

```bash
bash scripts/ice/setup.sh
```

It builds a Python environment in `~/scratch/envs/firepred` (ICE's Python 3.9, torch 2.3.1, MONAI 1.3.2,
the paper's versions), checks that every library imports, then asks for two things:

1. **Hugging Face**: paste your token. Answer `n` if asked to add it as a git credential.
2. **W&B**: paste your API key.

It ends with `All imports work. ...` and `Ready.`. Rerun it any time a library is missing; it keeps
what is installed and fixes the rest.

Every later session, before running Python by hand:

```bash
source scripts/ice/env.sh
```

## 4. Download the two datasets

Both come from the Hugging Face repo `SamuelWu318/ts-satfire-daily` (one archive per fire, one frame
per day). Download each into its own folder, since the TESSERA one leaves out fires without TESSERA.

| Folder | Holds | Train with |
| --- | --- | --- |
| `~/scratch/firepred/data-daily` | The daily dataset, every fire | `swinunetr3d` and other models without TESSERA |
| `~/scratch/firepred/data-daily-tessera` | The same archives, minus fires without prior-year TESSERA, plus TESSERA embeddings in `tessera/` | `swinunetr3d_tessera` and other TESSERA models |

**Daily, without TESSERA** (`data-daily`):

```bash
source scripts/ice/env.sh

python -m firepred.cli.dataset \
  --window-repo SamuelWu318/ts-satfire-daily \
  download -ts 6 -it 1 \
  --data-root "$HOME/scratch/firepred/data-daily"
```

**Daily, with TESSERA** (`data-daily-tessera`):

```bash
source scripts/ice/env.sh

python -m firepred.cli.dataset \
  --window-repo SamuelWu318/ts-satfire-daily \
  --tessera-repo SamuelWu318/ts-tesserafire \
  download -ts 6 -it 1 --tessera \
  --data-root "$HOME/scratch/firepred/data-daily-tessera"
```

Keep `-ts 6 -it 1`: they name how the daily archives were made. Files already on disk are kept, so if
a download stops, run the same command again.

Check that both finished:

```bash
for D in data-daily data-daily-tessera; do
  echo "$D: $(ls ~/scratch/firepred/$D/train | wc -l) train, $(ls ~/scratch/firepred/$D/val | wc -l) val, $(ls ~/scratch/firepred/$D/test | wc -l) test, $(ls ~/scratch/firepred/$D/tessera 2>/dev/null | wc -l) tessera"
done
ls ~/scratch/firepred/data-daily/window_config.json ~/scratch/firepred/data-daily-tessera/window_config.json
```

Each folder needs `window_config.json` and files in `train`, `val` and `test`; only the TESSERA folder
has `tessera` files.

<details>
<summary>The download outlasts your login session</summary>

Run it on a compute node instead: start an interactive CPU session, run the same commands, then
`exit`.

```bash
salloc -p ice-cpu -c 8 --mem=64G -t 4:00:00
```

</details>

## 5. Run a model

One command from the login node queues the whole run: `plan.sh` submits `CHAIN` training jobs of up
to 16 hours each, each continuing where the last stopped. Every `TEST` epochs (default 5) the
current weights score the test fires, and the run is tested once more at the end with its best
checkpoint.

**Without TESSERA: `swinunetr3d` on `data-daily`**

```bash
NAME=swin_daily \
MODEL=swinunetr3d \
DATA_ROOT="$HOME/scratch/firepred/data-daily" \
BATCH=8 LR=0.001 TS=6 IT=1 CHAIN=3 TEST=5 \
EXTRA="--epochs 100" \
bash scripts/ice/plan.sh
```

**With TESSERA: `swinunetr3d_tessera` on `data-daily-tessera`**

```bash
NAME=swin_tessera_daily \
MODEL=swinunetr3d_tessera \
DATA_ROOT="$HOME/scratch/firepred/data-daily-tessera" \
BATCH=8 LR=0.001 TS=6 IT=1 CHAIN=3 TEST=5 \
EXTRA="--tessera --epochs 100" \
bash scripts/ice/plan.sh
```

The TESSERA run differs in three places: `MODEL`, `DATA_ROOT` and `--tessera` in `EXTRA`. All three
go together. A TESSERA model without `--tessera` stops within a second.

`plan.sh` prints the job ids, for example `training swin_daily: 123456 123457 123458`. Each job writes
`logs/fp-<NAME>-<job id>.out`.

## 6. Every argument explained

### What you set

**Change** = set it for every run. **Sometimes** = tune it when you experiment. **Leave** = keep the
value shown.

| Argument | In the examples | Default | Change? | What it does |
| --- | --- | --- | --- | --- |
| `NAME` | `swin_daily` | none, required | **Change** | The run's name. It owns `~/scratch/firepred/runs/<NAME>.state`, so the same `NAME` resumes that run (and refuses if the settings differ). Use a new name for every new run; `ls ~/scratch/firepred/runs/` lists names in use |
| `MODEL` | `swinunetr3d` | `swinunetr3d` | **Change** | Which model. TESSERA models (`swinunetr3d_tessera`, `swinunetr3d_tessera_tpool`) need the TESSERA dataset and `--tessera`; others (`swinunetr3d`, `swinunetr3d_tpool`, ...) need neither. `ls firepred/models/zoo/` lists all |
| `DATA_ROOT` | `.../data-daily` | `~/scratch/firepred/data-daily` | **Change** | The dataset folder from step 4. Must match the model: `data-daily-tessera` for TESSERA models |
| `EXTRA` | `"--epochs 100"` | empty | **Change** | Any other training flags, passed as written. Must hold `--tessera` for TESSERA models. More flags below |
| `LR` | `0.001` | `0.001` | Sometimes | Learning rate. `0.001` is the paper's value |
| `BATCH` | `8` | `8` | Sometimes | Windows per training step. Lower it (`4`) if a GPU runs out of memory |
| `TEST` | `5` | `5` | Sometimes | Score the test fires (test F1 and IoU) with the current weights every `TEST` epochs, in W&B and the log. `0` tests only at the end. The final test of the best checkpoint always runs either way |
| `CHAIN` | `3` | `3` | Sometimes | Training jobs to queue, 16 hours each at most. Extra jobs of a finished run exit at once, so too many costs nothing; too few leaves the run unfinished (queue more with the same command) |
| `TS` | `6` | `6` | **Leave** | Input days per window. Must match the dataset's `window_config.json`: `6` for both datasets |
| `IT` | `1` | `1` | **Leave** | Days between window starts. Must match the dataset: `1` |

### What `train.sbatch` always passes

These are fixed in `scripts/ice/train.sbatch`. **Leave** them.

| Flag | Value | What it does |
| --- | --- | --- |
| `-mode` | `pred` | Next-day fire prediction, the task |
| `-nh` | `3` | Attention heads of the Swin models (the paper's) |
| `-ed` | `36` | Embedding size of the Swin models (the paper's) |
| `--num-workers` | `6` | Data-loading processes (the job has 8 CPUs) |
| `--test-every` | `TEST` (`5`) | Test F1/IoU every `TEST` epochs with the current weights. `TEST=0` turns it off |
| `--resume` | `~/scratch/firepred/runs/<NAME>.state` | Saved after every epoch, so the next job continues |

### Flags for `EXTRA`

| Flag | Default | Change? | What it does |
| --- | --- | --- | --- |
| `--tessera` | off | **Change** | Read the TESSERA embeddings. Required for TESSERA models, never for others |
| `--epochs N` | `100` | Sometimes | Epochs to train |
| `--lr-schedule none\|plateau\|cosine` | `none` | Sometimes | Learning-rate decay. `none` keeps it constant (the paper); `cosine` decays smoothly to `--lr-min` over `--epochs` |
| `--lr-min X` | `0.000001` | Sometimes | Lowest learning rate for `plateau` and `cosine` |
| `--loss dice\|dicece` | `dicece` for Swin models | Sometimes | Training loss. `dicece` is the paper's |
| `--test-every N` | `TEST` (`5`) | Leave | Set it with `TEST` instead. Here it overrides `TEST` |
| `-seed N` | `42` | Sometimes | Random seed. Change it to repeat a run with another seed |
| `--no-test-after-train` | testing on | Leave | Skip scoring the test fires at the end |
| `--tessera-cells-per-pixel k` | `1` | Leave | Must match the TESSERA files: `1` for `ts-tesserafire` |

Example with several:

```bash
NAME=swin_daily_cosine MODEL=swinunetr3d DATA_ROOT="$HOME/scratch/firepred/data-daily" \
BATCH=8 LR=0.001 TS=6 IT=1 CHAIN=3 TEST=5 \
EXTRA="--epochs 200 --lr-schedule cosine --lr-min 0.000001" \
bash scripts/ice/plan.sh
```

## 7. Optional parameters

Leave these out unless you need them. Add them before `bash scripts/ice/plan.sh` like the others.

| Parameter | Default | What it does |
| --- | --- | --- |
| `PARTITION` | `ice-gpu` | GPU partition. `coe-gpu` has only H100/H200. A comma list such as `ice-gpu,coe-gpu` starts on whichever has a free GPU first. `bash scripts/ice/partitions.sh` shows which ones accept your jobs |
| `GPU_TYPE` | empty (any GPU) | One GPU model: `h100`, `h200`, `a100`, `l40s`, ... Empty takes any GPU of the partition; a job that lands on one torch cannot use stops at once and the next job of the chain tries again. `GPU_TYPE=h100` avoids that |
| `SMOKE` | `0` | `SMOKE=1` first runs one training step on a GPU with the real data, and training waits for it to pass |
| `QOS` | your Slurm default | A named QOS for the GPU jobs. Leave it unset: your default works on `ice-gpu`, `coe-gpu` and `ice-cpu` |
| `CPU_PARTITION` | `ice-cpu` | CPU partition, used only when `DATA_ROOT` has no dataset yet |
| `CPU_QOS` | same as `QOS` | QOS of that CPU job |
| `ACCOUNT` | your default | Slurm account (`sbatch -A`) |

Example:

```bash
NAME=swin_daily_h100 MODEL=swinunetr3d DATA_ROOT="$HOME/scratch/firepred/data-daily" \
BATCH=8 LR=0.001 TS=6 IT=1 CHAIN=3 TEST=5 EXTRA="--epochs 100" \
PARTITION=ice-gpu,coe-gpu GPU_TYPE=h100 SMOKE=1 \
bash scripts/ice/plan.sh
```

## 8. Watch and stop runs

```bash
bash scripts/ice/status.sh                    # your jobs, run states and latest log lines
squeue -u $USER                               # your jobs only
tail -f logs/fp-swin_daily-*.out              # follow a run's log
scancel -n fp-swin_daily                      # stop every job of the run swin_daily
```

When a run finishes, its log ends with the test scores:

```bash
grep -h "Model F1" logs/fp-swin_daily-*.out
```

While it trains, every `TEST` epochs (default 5) adds a line with the current weights' test scores:

```bash
grep -h "test fire F1" logs/fp-swin_daily-*.out    # Epoch 5: test fire F1 ..., fire IoU ... (periodic, current weights)
```

In W&B these are the charts `Test F1 — Next-Day Fire Growth (periodic, current weights)` and
`Test IoU — Next-Day Fire Growth (periodic, current weights)`. They are for watching progress only:
the checkpoint the final test uses is still the one with the lowest validation loss.

The same numbers are on the run's W&B page (`test_f1`, `test_iou`), team `gt-fri`, project
`AFBAPred`. To test a chosen checkpoint, compare runs or read every option, see [README.md](README.md).
