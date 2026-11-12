# PACE ICE guide (`daily-store` branch)

Everything needed to train and test models on Georgia Tech's PACE ICE cluster with the scripts in
`scripts/ice/`. Run every command from the repository root on an ICE login node unless a step says
otherwise.

1. [Quickstart](#1-quickstart): one-time setup, up to the point of launching a model, and how to stay up to date.
2. [Run a model](#2-run-a-model): choosing the dataset, the model and the parameters.
3. [Test a model](#3-test-a-model): finding a run's test F1 and IoU, or testing a chosen checkpoint.
4. [Reference](#4-reference): partitions, QOS, GPUs, job monitoring, storage and file locations.

---

## 1. Quickstart

Do steps 1 to 5 once. After that you never reinstall: see [Staying up to date](#staying-up-to-date).

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

To also use the `fire-split-zones` branch, clone it into a second folder:

```bash
git clone -b fire-split-zones https://github.com/GT-Fire-Research-and-Innovation/fire-research-and-innovation.git fire-research-and-innovation-zones
```

Both clones share the same Python environment, data and run folders in `~/scratch`, so give runs
distinct `NAME`s across the two clones.

### Step 3: build the Python environment and log in to Hugging Face and W&B

```bash
bash scripts/ice/setup.sh
```

This builds a venv in `~/scratch/envs/firepred` from ICE's Python 3.9 with torch 2.3.1 and MONAI 1.3.2
(the paper's versions). It takes several minutes. Then it prompts twice:

1. **Hugging Face**: paste your Hugging Face token. If asked whether to add it as a git credential, answer `n`.
2. **W&B**: paste your W&B API key.

Each login is skipped when you are already logged in. The script ends with `Ready.`.
Other Python versions or library stacks are in [Python environment](#python-environment).

### Step 4: check which partitions accept your jobs

```bash
bash scripts/ice/partitions.sh coc-gpu ice-gpu pace-gpu ice-cpu
```

Look at the `ACCESS` and `QOS` columns. The defaults are GPU partition `coc-gpu`, CPU partition
`ice-cpu` and QOS `coc-ice`. When those rows say `yes` with QOS `coc-ice`, you need no partition
settings. Otherwise note a partition that says `yes` and its QOS for the run command in section 2.
Run it with no arguments to see every partition.

### Step 5 (optional): run the test suite

```bash
sbatch scripts/ice/test.sbatch
```

It runs pytest on a CPU node. The result is in `logs/fp-test-<jobid>.out`. Setup itself only checks
that the libraries import.

You are now ready to launch a model: go to [section 2](#2-run-a-model).

### Staying up to date

Your environment, datasets, run states, W&B runs and logins all live outside the repository, in
`~/scratch` and your home config. Pulling new code never touches them. **Do not delete the clone or
`~/scratch` when the GitHub branch changes.** Pull, then redo only the steps the change needs:

```bash
cd ~/fire-research-and-innovation    # or wherever you cloned this branch
git pull
git diff --stat ORIG_HEAD HEAD
```

The last command lists the files the pull changed. Use it with this table:

| If the pull changed | Redo |
| --- | --- |
| Only Python code (`firepred/`, models, training) | Nothing. The next job you submit uses the new code |
| `scripts/ice/setup.sh` (library versions or packages) | `bash scripts/ice/setup.sh`. It reuses the environment and only updates packages; logins are skipped |
| The Python version or `STACK` you want | `rm -rf ~/scratch/envs/firepred`, then `bash scripts/ice/setup.sh` |
| Dataset format, or you want a newer dataset upload | Use a new `DATA_ROOT` (see [2.2](#22-choose-the-dataset)). An existing data root is never replaced |

If `git pull` refuses because you edited files, run `git status` to see them. Use `git restore <file>`
to drop an edit you do not need, or `git stash`, `git pull`, `git stash pop` to keep it.

**Pulling while jobs are queued.** Jobs run the code in the clone at the moment each job starts. If
you pull while a training chain is still queued, the later jobs of that chain run the new code. To
keep a long chain on one version, wait for it to finish before pulling, or pull in a separate clone.

---

## 2. Run a model

One command, `plan.sh`, queues everything for a run: data preparation when the data root is empty,
an optional GPU smoke test, then a chain of training jobs that continue one another.

```bash
NAME=swin_daily_lr0001 \
MODEL=swinunetr3d \
DAILY_REPO=SamuelWu318/ts-satfire-daily \
DATA_ROOT="$HOME/scratch/firepred/data-daily" \
BATCH=8 LR=0.0001 TS=6 IT=1 CHAIN=3 \
EXTRA="--epochs 100 --lr-schedule cosine --lr-min 0.000001" \
bash scripts/ice/plan.sh

bash scripts/ice/status.sh
```

Put settings before `bash scripts/ice/plan.sh` as above, or `export` them first. Every setting except
`NAME` has a default. The sections below explain each choice.

### 2.1 Name the run

`NAME` identifies the run. Its state is saved to `~/scratch/firepred/runs/$NAME.state` after every
epoch, and a finished run leaves `$NAME.done`.

- **New experiment or changed settings:** use a new `NAME`. Resuming is refused when the model,
  loss, learning-rate schedule, TESSERA use, epochs, learning rate, batch size, seed, `TS` or `IT`
  differ from the saved state.
- **Continue a run:** submit the same command with the same `NAME`. A finished run exits at once.

### 2.2 Choose the dataset

Training reads local files from `DATA_ROOT`. The repo settings only control how an empty `DATA_ROOT`
gets filled, by a CPU job that `plan.sh` queues before training.

| Setting | Default | Meaning |
| --- | --- | --- |
| `DATA_ROOT` | `~/scratch/firepred/data-daily` | Local data folder training reads |
| `DAILY_REPO` | unset | Download an existing daily dataset from this Hugging Face repo. Recommended: `SamuelWu318/ts-satfire-daily` |
| `SRC_REPO` | `SamuelWu318/ts-satfire-processed-6-1` | Used only when `DAILY_REPO` is unset: converts processed 6-day windows into daily archives (slower) |
| `DEST_REPO` | unset | Optionally upload the converted daily archives to this repo |

- `DAILY_REPO` wins over `SRC_REPO`.
- `plan.sh` skips data preparation when `DATA_ROOT/window_config.json` exists. Changing a repo setting
  does not replace an existing data root, so use a different `DATA_ROOT` for each dataset.
- `data.sbatch` always downloads or converts with `-ts 6 -it 1`. Your training `TS` and `IT` do not
  change that.

**Using TESSERA embeddings.** TESSERA needs a data root that holds the embeddings, plus the
`--tessera` training flag. `plan.sh` does not download embeddings. Prepare a separate root once with
a CPU job:

```bash
sbatch -J fp-tessera-data -p ice-cpu --qos=coc-ice -c 8 --mem=64G -t 16:00:00 -o logs/%x-%j.out \
  --wrap "source scripts/ice/env.sh && python -m firepred.cli.dataset \
  --window-repo SamuelWu318/ts-satfire-daily --tessera-repo SamuelWu318/ts-tesserafire \
  download -ts 6 -it 1 --tessera --data-root $HOME/scratch/firepred/data-daily-tessera"
```

This skips fires without prior-year TESSERA. Repo options go before the `download` subcommand. Then
train with `DATA_ROOT="$HOME/scratch/firepred/data-daily-tessera"` and `EXTRA="--tessera ..."`.
Embeddings are read from `<DATA_ROOT>/tessera` by default. Add `--tessera-dir <path>` to `EXTRA` for
another folder, and `--tessera-cells-per-pixel N` for embeddings built on a finer grid.

### 2.3 Choose the model

Set `MODEL` to one of:

| `MODEL` | Notes |
| --- | --- |
| `swinunetr3d` | Default |
| `swinunetr3d_tpool` | SwinUNETR-3D that learns a weight per input day instead of averaging its day outputs |
| `swinunetr3d_tessera` | Takes TESSERA input: use a TESSERA data root and `--tessera` |
| `swinunetr3d_tessera_tpool` | Takes TESSERA input: use a TESSERA data root and `--tessera` |
| `unetr3d` | |
| `unet3d` | |
| `attunet` | Its default loss is `dice`; every other model defaults to `dicece` |
| `convlstm` | |
| `res18unet` | |

### 2.4 Choose the parameters

| Setting | Default | Meaning |
| --- | --- | --- |
| `BATCH` | `8` | Batch size |
| `LR` | `0.001` | Learning rate |
| `TS` | `6` | Input window length in consecutive days |
| `IT` | `1` | Days between window starts, not between days inside a window. `TS=6 IT=2` uses days 1-6 to predict day 7, then days 3-8 to predict day 9 |
| `CHAIN` | `3` | Training jobs queued one after another, each up to 16 hours, continuing the same run |
| `SMOKE` | `0` | `1` adds a short GPU check before training |
| `EXTRA` | empty | Any other training flags, in one quoted string |

On this branch, training checks `TS` and `IT` against the values recorded in
`DATA_ROOT/window_config.json`. The default dataset needs `TS=6 IT=1`, the defaults. Another
configuration needs a data root prepared with those settings. The `fire-split-zones` branch can cut
daily archives at any `TS` of 6 or more and any `IT`, and adds fire split zone training.

Common `EXTRA` flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--epochs N` | `100` | Total epochs across the whole chain |
| `--lr-schedule none\|plateau\|cosine` | `none` | Learning-rate decay. `cosine` decays to `--lr-min` over `--epochs` |
| `--lr-patience N`, `--lr-decay-factor F` | | `plateau` only: decay by `F` after `N` epochs without a lower validation loss |
| `--lr-min F` | | `plateau` and `cosine`: the lowest learning rate |
| `--loss dice\|dicece` | per model | Loss function |
| `--ce-weight auto` or `--ce-weight BG FIRE` | `auto` | `dicece` class weights. `auto` reads the pixel ratio from the data root |
| `-seed N` | `42` | Random seed |
| `--test-every N` | `0` | Also score the test fires every `N` epochs, for charts only |
| `--no-test-after-train` | test on | Skip the final test after training |
| `--upload-checkpoints` | off | Store each best checkpoint as a W&B artifact |
| `--tessera` | off | Add TESSERA embeddings (see [2.2](#22-choose-the-dataset)) |

Example:

```bash
EXTRA="--epochs 100 --lr-schedule plateau --lr-patience 5 --lr-decay-factor 0.5 --lr-min 0.000001"
```

`train.sbatch` already passes `-mode pred -nh 3 -ed 36 --num-workers 6`. Flags in `EXTRA` come later
and override them. For every flag:

```bash
source scripts/ice/env.sh
python -m firepred.cli.train --help
```

**How many epochs fit?** If the chain ends before the last epoch, run the same `plan.sh` command
again with the same `NAME`. It skips the data step and continues from the saved state.

### 2.5 Choose where it runs

| Setting | Default | Meaning |
| --- | --- | --- |
| `PARTITION` | `coc-gpu` | GPU partition, or a comma list such as `coc-gpu,ice-gpu`: each job starts on whichever has a free GPU first |
| `QOS` | `coc-ice` | QOS of the GPU jobs. `QOS=default` uses your Slurm default |
| `GPU_TYPE` | any | One GPU model, such as `h100`, `h200`, `a100` |
| `CPU_PARTITION` | `ice-cpu` | Partition of the data job |
| `CPU_QOS` | same as `QOS` | QOS of the data job |
| `ACCOUNT` | your default | Slurm account |

`plan.sh` tests every listed partition with `sbatch --test-only` and drops one that refuses the job,
printing Slurm's reason. Use the `QOS` that `partitions.sh` reported for the partition. All partitions
in one `PARTITION` list must accept the same QOS.

```bash
NAME=swin_h100 PARTITION=coc-gpu,ice-gpu GPU_TYPE=h100 bash scripts/ice/plan.sh
NAME=swin_ice PARTITION=ice-gpu CPU_PARTITION=ice-cpu QOS=default CPU_QOS=default bash scripts/ice/plan.sh
```

Keep your model, data and training settings in the same command. More in
[Partitions, QOS and accounts](#partitions-qos-and-accounts).

### 2.6 Watch, resume and stop

```bash
bash scripts/ice/status.sh                 # jobs, run states and the last epoch in recent logs
tail -f logs/fp-<NAME>-<jobid>.out         # follow one training job
scancel -n fp-<NAME>                       # stop every job of one run's chain
```

Training logs are `logs/fp-<NAME>-<jobid>.out` inside the clone you submitted from. To restart a
stopped run, submit the same command with the same `NAME`.

**Smoke test caveat:** `SMOKE=1` passes the model, data root and `TS`, but not `EXTRA` or `IT`.
It does not reproduce a TESSERA or custom-interval run.

---

## 3. Test a model

### Where test F1 and IoU come from

By default, every training run tests itself when its last epoch ends. It reloads the checkpoint with
the lowest validation loss and scores the test fires. The scores are the mean over fires of each
fire's mean over its windows, as in the paper. Two places record them:

- **The training log** ends with per-fire scores, then one line in this form:
  `Model F1: <f1>; background F1: <bg f1>; model IoU: <iou>; background IoU: <bg iou>`.
- **The W&B run** summary holds `test_f1` and `test_iou`, also charted as
  `Test F1 — Next-Day Fire Growth` and `Test IoU — Next-Day Fire Growth`.

Only one checkpoint per run stays on disk: the one with the lowest validation loss.

### Find a run's test F1 and IoU

List your runs. A `.done` file marks a finished run:

```bash
ls ~/scratch/firepred/runs/
```

Read one run's test scores and its W&B link from its logs:

```bash
grep -h "Model F1" logs/fp-<NAME>-*.out
grep -ho "https://wandb.ai/[^ ]*/runs/[a-z0-9]*" logs/fp-<NAME>-*.out | head -1
```

Compare every finished run of this clone at once:

```bash
for f in ~/scratch/firepred/runs/*.done; do
  n=$(basename "$f" .done)
  echo "$n: $(grep -h 'Model F1' logs/fp-$n-*.out 2>/dev/null | tail -1)"
done
```

Runs submitted from the other clone have their logs in that clone's `logs/` folder.

**Finding a run in W&B.** W&B run names list the model, heads, width, batch size and tags, such as
`train_pred_swinunetr3d_num_heads_3_hidden_size_36_batchsize_8_dicece_lr-cosine`. They do not include
your `NAME` or the learning rate, so different runs often share a name. Open the link from the log
instead of searching by name.

### Test a chosen checkpoint

Test by hand when a run stopped before its final test, was trained with `--no-test-after-train`, or
should be scored again after a code change.

**Step 1: find the checkpoint.** A finished run prints its path:

```bash
grep -h "Best checkpoint" logs/fp-<NAME>-*.out | tail -1
```

When the line is missing, take the run id from the end of the W&B link above and list its files:

```bash
ls -t ~/scratch/firepred/wandb/*-<run-id>/files/*.pth
```

**Step 2: submit a test job.** Use the same `MODEL`, `TS` and `IT` as the run, and the same
model-shaping flags from its `EXTRA`: `--tessera`, `--tessera-dir`, `--tessera-cells-per-pixel`,
`--model-option`, `--swin-temporal-downsample`. `-b` and `-lr` are required by the command line but
do not change test scores.

```bash
CKPT=<path from step 1>
sbatch -J fp-test-<NAME> -p coc-gpu --qos=coc-ice --gres=gpu:1 -c 8 --mem=64G -t 2:00:00 \
  -o logs/%x-%j.out --wrap "source scripts/ice/env.sh && python -m firepred.cli.train -test \
  --checkpoint $CKPT -m swinunetr3d -mode pred -b 8 -lr 0.001 -nh 3 -ed 36 -ts 6 -it 1 \
  --data-root $HOME/scratch/firepred/data-daily --num-workers 6"
```

Raise `-t` if the job hits its time limit.

- **Always pass `--checkpoint`.** Without it, `-test` takes the newest checkpoint whose file name
  matches the model, batch size and tags. Runs with different learning rates have the same file
  name, so that can pick another run's checkpoint.
- A checkpoint trained with other inputs is refused with an error that names the difference.
- TESSERA checkpoints on the default torch 2.3.1 also need `--trust-checkpoint`. Use it only for
  checkpoints your own runs created.

**Step 3: read the result.** `logs/fp-test-<NAME>-<jobid>.out` ends with the `Model F1 ... model IoU`
line. A new W&B run named `test_pred_...` holds `test_f1` and `test_iou`. Prediction plots go to
`evaluation_plot/`; `--plot-dir <path>` changes that.

### Watch test scores during training

`EXTRA="--test-every 10"` scores the test fires every 10 epochs and charts them in W&B as periodic
test F1 and IoU. This is for watching progress only. The kept checkpoint is still chosen by
validation loss.

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

Everything large lives in `~/scratch` because the home directory is small. All paths are set in
`scripts/ice/env.sh`; export a variable before a command to override it.

### Partitions, QOS and accounts

`partitions.sh` lists each partition's free nodes, GPU models and time limit. It also asks Slurm
whether it would accept your job there, without submitting anything:

```bash
bash scripts/ice/partitions.sh                                  # every partition
bash scripts/ice/partitions.sh coc-gpu ice-gpu pace-gpu         # only these
GPU_TYPE=h100 bash scripts/ice/partitions.sh coc-gpu ice-gpu    # can I get an H100 there?
```

The `QOS` column names the QOS that worked; pass it as `QOS=...`. When it says `default`, use
`QOS=default`: `plan.sh` then omits `--qos` and Slurm picks your default. An empty value does the
same. `CPU_QOS` follows `QOS` when unset and accepts the same values. One job carries one QOS, so
partitions that need different QOS cannot share one `PARTITION` list.

| Question | Command |
| --- | --- |
| Which partitions exist, how full are they? | `sinfo -s` |
| GPUs, limits and node states per partition | `sinfo -p coc-gpu,ice-gpu,pace-gpu -o "%.12P %.5a %.10l %.5D %.8t %G"` |
| My accounts and QOS (empty partition = any) | `sacctmgr show assoc user=$USER format=account%15,partition%15,qos%40` |
| Who may use a partition, its max time | `scontrol show partition coc-gpu` (AllowAccounts, AllowQos, MaxTime) |
| Limits of a QOS (GPUs per user, wall time) | `sacctmgr show qos name=coc-ice format=name%12,maxwall,maxtrespu%30,maxjobspu,maxsubmitpu` |
| Would this exact job be accepted, and when would it start? | `sbatch --test-only -p coc-gpu --qos=coc-ice --gres=gpu:1 -t 1:00:00 --wrap true` |
| PACE's own view of a queue | `pace-check-queue coc-gpu` |

`sbatch --test-only` is the definitive check. It fails with Slurm's reason, such as
"Invalid qos specification", when you may not use that partition, QOS, account or GPU model.

**Direct `sbatch`.** The `.sbatch` scripts can also be submitted by hand. Options on the command line
override the script's `#SBATCH` lines. `QOS` and `CPU_QOS` are read by `plan.sh` only, so pass
`--qos=<name>` yourself:

```bash
NAME=swin_daily sbatch -p coc-gpu,ice-gpu --qos=coc-ice scripts/ice/train.sbatch
sbatch -p ice-cpu --qos=coc-ice scripts/ice/data.sbatch
```

### Nodes and GPUs

| Question | Command |
| --- | --- |
| Every node of a partition: state, GPUs, GPUs in use, CPUs, memory | `sinfo -N -p coc-gpu -O "NodeList:24,StateCompact:8,Gres:28,GresUsed:34,CPUsState:16,Memory:9"` |
| Which GPU models exist where | `sinfo -o "%.12P %G" \| sort -u` |
| Idle nodes only | `sinfo -p coc-gpu,ice-gpu,pace-gpu -t idle,mix -N -o "%N %P %t %G"` |
| One node in detail | `scontrol show node <node-name>` |
| What runs on a node | `squeue -w <node-name>` |

`CPUsState` reads allocated/idle/other/total. A node in state `mix` still has free CPUs and maybe free
GPUs; `GresUsed` shows how many GPUs are taken. GPU model names in `Gres`, such as `gpu:h100:4` for
four H100s per node, are what `GPU_TYPE` and `--gres=gpu:<model>:1` take.

### Watching jobs

| Question | Command |
| --- | --- |
| Everything at once (jobs, run states, latest logs) | `bash scripts/ice/status.sh` |
| My jobs | `squeue -u $USER -o "%.10i %.20j %.9P %.8T %.10M %.12l %.24R"` |
| When will my pending jobs start? | `squeue -u $USER --start` |
| Why is a job pending, and where did it run? | `scontrol show job <jobid>` |
| How busy is a partition? | `squeue -p coc-gpu -t PENDING \| wc -l` |
| Finished jobs today: state, time, memory, exit code | `sacct -u $USER -S today --format=JobID,JobName%20,Partition,State,Elapsed,MaxRSS,ExitCode` |
| How efficiently a finished job used CPUs and memory | `seff <jobid>` |
| My scheduling priority and fair share | `sprio -u $USER` and `sshare -u $USER` |

Pending reasons in the last column of `squeue`:

| Reason | Meaning |
| --- | --- |
| `Priority`, `Resources` | Waiting its turn or for a free node. Nothing to fix |
| `Dependency` | Waiting for an earlier job of the chain |
| `DependencyNeverSatisfied` | The job it waits for failed, such as the data job or smoke test. `scancel` it |
| `QOSMaxGRESPerUser`, `AssocGrpGRES` | You already use the most GPUs your QOS allows. It starts when one ends |
| `ReqNodeNotAvail` | The nodes are reserved, often for maintenance |
| `PartitionTimeLimit` | `-t` is longer than the partition allows |

### Cancelling

```bash
scancel <jobid>              # one job
scancel -n fp-<NAME>         # every job of one run's chain
scancel -u $USER -t PENDING  # all of my pending jobs
scancel -u $USER             # all of my jobs
```

### Interactive GPU session

```bash
salloc -p coc-gpu --qos=coc-ice --gres=gpu:1 -c 8 --mem=32G -t 1:00:00
hostname                     # a compute node; if it is still a login node, run: srun --pty bash
nvidia-smi
source scripts/ice/env.sh    # this repository's Python environment
exit                         # give the node back
```

### Python environment

```bash
bash scripts/ice/setup.sh                        # default: venv from ICE's python3, paper library versions
PYTHON=/path/to/python3.12 bash scripts/ice/setup.sh
CONDA_PYTHON=3.12 bash scripts/ice/setup.sh      # conda environment from ICE's mamba or anaconda3 module
STACK=latest bash scripts/ice/setup.sh           # newest torch and MONAI for that Python
```

`setup.sh` reuses an existing environment. To switch Python or stack, delete
`~/scratch/envs/firepred` first. `source scripts/ice/env.sh`, then `python -V` and `pip list`, show
what is installed. The golden tests were captured with the default stack; `STACK=latest` gives
slightly different floating-point results.

### Tests

```bash
sbatch scripts/ice/test.sbatch                               # test suite on a CPU node, without slow tests
SLOW=1 sbatch scripts/ice/test.sbatch                        # include slow end-to-end tests
PYTEST_ARGS="tests/test_golden.py -x" sbatch scripts/ice/test.sbatch
NAME=swin_daily SMOKE=1 bash scripts/ice/plan.sh             # GPU smoke test before training
```

### Storage

| Question | Command |
| --- | --- |
| My quotas and usage | `pace-quota` |
| What takes the space | `du -sh ~/scratch/* \| sort -h` |
