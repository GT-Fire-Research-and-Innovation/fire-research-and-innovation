# Training on PACE ICE

Run these commands from the repository root on an ICE login node. For partition,
GPU, queue, and storage commands, see [QUICKSTART.md](QUICKSTART.md).

## Setup and launch

```bash
bash scripts/ice/setup.sh
bash scripts/ice/partitions.sh

NAME=swin_daily_lr0001 \
MODEL=swinunetr3d \
DAILY_REPO=SamuelWu318/ts-satfire-daily \
DATA_ROOT="$HOME/scratch/firepred/data-daily" \
BATCH=8 LR=0.0001 TS=6 IT=1 CHAIN=3 \
EXTRA="--epochs 100 --lr-schedule cosine --lr-min 0.000001" \
bash scripts/ice/plan.sh

bash scripts/ice/status.sh
```

Place settings before `bash scripts/ice/plan.sh`, as above, or export them first.
The plan prepares missing data and queues training jobs that continue the same run.

## Training options

| Setting | Example | Default / meaning |
| --- | --- | --- |
| Run identifier | `NAME=experiment_01` | Required; identifies the resume state and completion marker |
| Model | `MODEL=swinunetr3d` | `swinunetr3d` |
| Batch size | `BATCH=8` | `8` |
| Learning rate | `LR=0.001` | `0.001` |
| Input window length | `TS=6` | 6 consecutive input days |
| Window interval | `IT=1` | Start a window every day |
| Local dataset | `DATA_ROOT="$HOME/scratch/my-data"` | `~/scratch/firepred/data-daily` |
| Existing daily dataset repo | `DAILY_REPO=owner/repo` | Unset; download daily archives instead of converting windows |
| Processed source repo | `SRC_REPO=owner/repo` | `SamuelWu318/ts-satfire-processed-6-1`; converted when `DAILY_REPO` is unset |
| Upload converted data | `DEST_REPO=owner/repo` | Unset; optionally upload the conversion |
| Enable TESSERA | `EXTRA="--tessera"` | Off; embeddings must already be downloaded |
| Epochs | `EXTRA="--epochs 100"` | `100` total across the training chain |
| Learning-rate schedule | `EXTRA="--lr-schedule cosine"` | `none`; also accepts `plateau` |
| Chained jobs | `CHAIN=3` | Three jobs continuing one run, each with a 16-hour limit |
| Optional GPU smoke test | `SMOKE=1` | Off |
| GPU partitions | `PARTITION=coc-gpu,ice-gpu` | `coc-gpu`; use the first available compatible partition |
| GPU type | `GPU_TYPE=h100` | Any available GPU type |
| GPU QOS | `QOS=coc-ice` | `coc-ice` |
| CPU partitions / QOS | `CPU_PARTITION=ice-cpu CPU_QOS=coc-ice` | `ice-cpu`; CPU QOS defaults to `QOS` |
| Slurm account | `ACCOUNT=my-account` | Your default account |

Available model names:

- `swinunetr3d`
- `swinunetr3d_tpool`
- `swinunetr3d_tessera`
- `swinunetr3d_tessera_tpool`
- `unetr3d`
- `unet3d`
- `attunet`
- `convlstm`
- `res18unet`

Combine additional training flags inside one `EXTRA` string:

```bash
EXTRA="--tessera --epochs 100 --lr-schedule plateau --lr-patience 5 --lr-decay-factor 0.5 --lr-min 0.000001"
```

Other flags include `--loss dicece`, `--ce-weight auto`, `-seed 42`,
`--num-workers 6`, `--test-every 10`, and `--no-test-after-train`.
The training script supplies `-mode pred -nh 3 -ed 36 --num-workers 6`;
later flags in `EXTRA` can override these values.
For all supported training flags:

```bash
source scripts/ice/env.sh
python -m firepred.cli.train --help
```

## Choosing a dataset repo

Training reads local files from `DATA_ROOT`. The repo variables only control
data preparation:

- Set `DAILY_REPO` to download an existing daily dataset.
- Otherwise, `SRC_REPO` supplies processed window archives to convert to daily
  archives. `DEST_REPO` optionally uploads the conversion.
- `DAILY_REPO` takes precedence over `SRC_REPO`.
- `plan.sh` skips preparation when `DATA_ROOT/window_config.json` exists.
  Changing a repo variable does not replace an existing local dataset. Choose a
  separate `DATA_ROOT` when switching datasets.

The current `data.sbatch` uses `-ts 6 -it 1` for both download and conversion;
changing training's `TS` and `IT` does not change those preparation arguments.

## Enabling TESSERA

TESSERA requires both local embeddings and `--tessera` during training.
The ICE data script does not automatically download the embeddings.

On an allocated compute node, prepare a separate root once:

```bash
source scripts/ice/env.sh

python -m firepred.cli.dataset \
  --window-repo SamuelWu318/ts-satfire-daily \
  --tessera-repo SamuelWu318/ts-tesserafire \
  download -ts 6 -it 1 --tessera \
  --data-root "$HOME/scratch/firepred/data-daily-tessera"
```

This downloads data and embeddings into the selected root and excludes fires
without prior-year TESSERA. Repo options go before the `download` subcommand.
Use a separate root to retain the original dataset for experiments without TESSERA.

Then submit training from the login node:

```bash
NAME=swin_tessera_01 \
MODEL=swinunetr3d \
DATA_ROOT="$HOME/scratch/firepred/data-daily-tessera" \
BATCH=8 LR=0.001 TS=6 IT=1 \
EXTRA="--tessera --epochs 100" \
bash scripts/ice/plan.sh
```

The default embedding folder is `<DATA_ROOT>/tessera`. Add
`--tessera-dir /path/to/tessera` to `EXTRA` to choose another.
For embeddings built with a finer grid, also supply the matching
`--tessera-cells-per-pixel N`.

## Windows and intervals

`TS` is the number of consecutive input days. `IT` is the spacing between
window starts, not between days inside a window. For example, `TS=6 IT=2`
uses days 1–6 to predict day 7, then days 3–8 to predict day 9.

On `fire-split-zones`, daily archives (format 3) are cut into windows at the
run's `TS` and `IT`, provided the necessary input days and target days exist.
Archives converted from `6-1` windows support lengths of 6 or longer and other
strides. Fixed window archives must match their recorded settings.

### Fire split zones

Set `ZONES=1` to train with fire split zones. `ZONES_ROOT` defaults to
`<DATA_ROOT>/zones`; `SCALE_KM` optionally changes the zone-building scale.
The ICE data job builds zones with `TS` and a fixed interval of 1.

If you change the window settings, or prepared the root separately (including
the TESSERA example above), build matching zone files on a compute node first:

```bash
source scripts/ice/env.sh

python -m firepred.cli.dataset zones \
  --data-root "$HOME/scratch/firepred/data-daily-tessera" \
  -ts 6 -it 1 \
  --zones-root "$HOME/scratch/firepred/data-daily-tessera/zones"
```

Then add `ZONES=1` to the training command, keeping the same data root, `TS`,
`IT`, and zone location. An existing `window_config.json` makes `plan.sh`
skip the data job, including zone preparation.

## Resuming and optional checks

Use a new `NAME` when changing an experiment's settings. Reuse the same name and
settings to resume. State lives at `$FIRE_WORK/runs/$NAME.state`; a completed run
leaves `$NAME.done`, causing later jobs with that name to exit immediately.
The default `FIRE_WORK` is `~/scratch/firepred`, so choose distinct names across
both checkouts when they share this location.

Setup checks library imports and does not run pytest. Tests are optional:

```bash
NAME=swin_daily SMOKE=1 bash scripts/ice/plan.sh  # add a GPU smoke test
sbatch scripts/ice/test.sbatch                  # test suite, excluding slow tests
SLOW=1 sbatch scripts/ice/test.sbatch           # include slow tests
```

The optional smoke test currently forwards the model, data root, window length,
and branch-specific zone option, but does not forward `EXTRA` or `IT`.
It therefore does not reproduce a custom TESSERA or interval configuration.

