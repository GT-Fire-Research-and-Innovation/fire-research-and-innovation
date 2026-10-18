# Architecture

`firepred` predicts which pixels of a wildfire burn **tomorrow**, from six
days of satellite, weather and terrain data (the TS-SatFire prediction task).
The code is a hierarchy that builds toward the model: each level uses only
the levels before it, so any file can be understood by reading downward.

```
                    config/          constants: dataset.py, training.py
                       │
              ┌────────┴────────┐
              ▼                 ▼
            data/            models/       data/: GeoTIFFs and TESSERA -> Hugging Face -> tensors
   (build side + load side)  blocks → zoo   (notebooks call windows_master.py, tessera_master.py)
              │                 │
              └────────┬────────┘
                       ▼
                   training/        RunConfig → Experiment: FireData → Tracker → Trainer → Evaluator
                       │
                       ▼
                     cli/           python -m firepred.cli.{train,dataset,check,new_model}
```

| Package | Level | Imports from | Job |
|---|---|---|---|
| `config/` | 0 | – | Every project-wide constant, defined once (`dataset.py`, `training.py`) |
| `data/` | 1 | config | The data end to end. Load side (training): `archive.py`, `tessera_archive.py`, `transforms.py`, `fire_dataset.py`. Build side (dataset pipeline): `fires.py`, `files.py`, `tessera.py`, `hub.py`, `windows.py` and the two masters |
| `models/` | 2 | config | `registry.py` (ModelSpec, `@register_model`), reusable `blocks/`, one file per model in `zoo/` |
| `training/` | 3 | config, data, models | One experiment as a one-way chain of classes; `flags.py` defines every training flag |
| `cli/` | 4 | everything | Thin argparse entry points |

`tests/test_layers.py` enforces the "Imports from" column and the import order
inside `data/` and `training/` (listed in each package's `__init__.py`), keeps
every `__init__.py` import-free, checks that training never loads the
build-side modules of `data/` (Kaggle installs no geospatial packages), that
the two masters import without rasterio, pyproj, geotessera, torch or
huggingface_hub, and that only `training/tracker.py` imports W&B.

Each data format has one module that writes and reads it: `data/archive.py`
for the window archives, `data/tessera_archive.py` for the TESSERA files. The
build side produces them, the load side reads them, and nothing else knows
their layout.

## Design rules

1. **Layers.** Imports only go down the table above, and inside `data/` and
   `training/` each module imports only modules listed before it (the order
   is in each package's `__init__.py` docstring).
2. **One home per constant.** Project-wide constants live in `firepred/config/`.
   Functions take them as default arguments, CLI flags default to them, and
   notebook variables map 1:1 to CLI flags. The one exception is a model's
   own architecture constants, which live at the top of its `models/zoo/` file,
   so adding a model never touches config.
3. **Dataset building is functional.** Every function on the build side of
   `data/` takes inputs and returns a value (writers return the path, uploads
   the names, deletions what they removed), never modifies its arguments, and
   keeps no module state; classes are frozen dataclasses (plans, reports,
   results). Each master step is a pure plan (`plan_windows`, `plan_tessera`)
   followed by a short loop that does the I/O. Only the two masters print.
   `tests/test_prep.py` checks all of this.
4. **Model running is class-based and one-way.** A frozen `RunConfig` is built
   once from the command line; each class receives what it needs in its
   constructor or method call and returns typed results (`EpochScores`,
   `TestScores`) upward. Nothing calls back into an earlier step and no
   closures are passed down.
5. **One source for every setting.** Each training flag is one entry in
   `training/flags.py`, naming the RunConfig field it sets (its default is the
   field's); the parser, `config_from_args`, `argv_for` and `cli.check`'s
   flags are built from it. The dataset CLI's flags set master parameters of
   the same name and take their defaults from the master signatures.
6. **Every module, class and function has a docstring** saying what it does
   (`tests/test_docstrings.py`). Module docstrings also give the level and who
   uses it.
7. **Provenance tags in comments.** `[paper]` follows the TS-SatFire paper
   text; `[released]` follows the released TS-SatFire code; `[ours]` is in
   neither. Help strings keep their "NEW, not in the paper" wording.
8. **Outputs are pinned by tests.** `tests/golden/` holds the outputs of every
   deterministic step; a refactor must reproduce them exactly. A behavior
   change is its own labelled commit, recaptured with `--reason` (see
   `docs/KNOWN_ISSUES.md`).

## Building the dataset (build side of `data/`)

Notebooks and `python -m firepred.cli.dataset` call only the masters:

| Step | Master function | CLI |
|---|---|---|
| Every local path from a work folder and the window settings | `windows_master.workspace` | – |
| Keep caches on the work disk | `windows_master.pin_caches` | – |
| A token from Kaggle secrets, Colab userdata or the environment | `windows_master.read_secret` | `--token` / `$HF_TOKEN` |
| Raw GeoTIFFs → one window archive per fire | `windows_master.process_windows` | `process` |
| Window archives (+ TESSERA) into a data root | `windows_master.download_windows` | `download` |
| Zip a data root, upload it | `windows_master.upload_data_zip` | `upload-zip` |
| Rebuild data.zip from the window repo | `windows_master.rebuild_data_zip` | `rebuild-zip` |
| Download and unpack data.zip | `windows_master.pull_data_zip` | `pull-zip` |
| Training notebooks: data (+ TESSERA view) ready to train | `windows_master.stage_training_data` | `stage` |
| Prior-year TESSERA per fire | `tessera_master.process_tessera` | `tessera` |
| Bundle TESSERA into tessera.zip | `tessera_master.upload_tessera_zip` | `tessera-zip` |
| Unpack tessera.zip, drop fires without TESSERA | `tessera_master.pull_tessera_zip` | `pull-tessera` |

Below the masters: `fires.py` (pure: splits, label rules, fire ids),
`files.py` (local files: zip, unzip, data roots), `tessera.py` (everything
about building TESSERA: a fire's grid → `FireTessera`, which fires get one,
training views without the excluded fires), `hub.py` (all Hugging Face I/O and
`window_config.json`) and `windows.py` (one fire's GeoTIFFs → `FireWindows`).
The ROI CSVs are read once per master call into a `RoiTable` and passed down.

## Follow one training run (`training/`)

`python -m firepred.cli.train -m swinunetr3d ... --data-root D` runs
`cli/train.py: parse_args → config_from_args → Experiment(config).run()` (the flags are `training/flags.py`)
(`training/experiment.py`):

1. **Device** (`training/accelerator.Device.resolve`): `--device auto` is the
   GPU when visible, else the CPU; `--device tpu` is one TPU device through
   PyTorch/XLA. Then **seed** numpy, torch, Python's random and (TPU) XLA
   (`Experiment.seed`).
2. **FireData.locate(config)**: where train/, val/, test/ are; with
   `--unzip-dir`, the archives are unpacked once (`data/archive.unzip_data_root`).
3. **`with Tracker(config, device)`** (`training/tracker.py`)
   starts the W&B run: charts, the run name and config all come from the
   RunConfig, and leaving the block finishes the run.
4. **Model**: `Experiment.prepare_model` → `models/registry.build_model("swinunetr3d", config.model_spec())`,
   wrapped in `nn.DataParallel`.
5. **Training** (skipped with `-test`):
   - `FireData.training()` builds the augmented training and plain validation
     `FireDataset`s (`data/fire_dataset.py`; TESSERA statistics from the training
     fires only) and their loaders, each on its own random generator.
   - An `Evaluator` for the test fires when `--test-every` is set, and a
     `CheckpointStore` in `tracker.checkpoint_dir` (`training/checkpoints.py`).
   - `Trainer(config, model, device).fit(training, tracker, store, evaluator)`
     (`training/trainer.py`). Each epoch: `train_epoch` → `tracker.log_train`;
     `validate` → `tracker.log_validation`; `store.offer` saves the checkpoint
     if its validation loss is among the best (and `tracker.log_checkpoint`
     uploads it with `--upload-checkpoints`); the `LearningRateSchedule` steps; every
     `--test-every` epochs `evaluator.score` → `tracker.log_periodic_test`.
     `fit` returns the best checkpoint.
6. **Testing**: `CheckpointLoader(device, config).load` (refuses a checkpoint trained with other
   inputs, and runs no pickled code unless trusted), then
   `Evaluator(config, device, test_dataset).score(model, plot_dir)` scores
   each 2021 test fire, averages over fires, and `tracker.log_final` logs it.

Inside a step: `FireDataset[i]` is a thin adapter over two plain functions:
`data/fire_dataset.read_window` reads the window (via `data/archive.open_fire`)
and `prepare_window` turns it into
`{"data": (43, 6, 256, 256), "labels": (2, 256, 256)}` with
`data/transforms.py` (normalize → augment → preprocess, all returning new
tensors). `data/fire_dataset.batch_inputs` appends the TESSERA map on the GPU,
`models/registry.forward_model` averages the per-day logits over the days,
and `training/objectives.TrainingLoss` returns the loss and its logged parts.

**Finer TESSERA (k x k cells per VIIRS pixel)** [ours]. By default each
375 m VIIRS pixel gets one TESSERA vector, the mean of the ~1400 10 m pixels
inside it. `process_tessera(..., cells_per_pixel=k)` (CLI `tessera
--cells-per-pixel k`) keeps k x k cells instead: k = 5 gives 75 m cells and a
(128, 1280, 1280) map per fire, 25 times larger, so use a separate TESSERA repo
per k. Train with the same `--tessera-cells-per-pixel k`:

- `FireDataset` reads each window's map from its file (`lazy_tessera`) and
  returns it as `static` (129, 256 k, 256 k) float16; statistics are computed by
  streaming the training maps one at a time (`iter_tessera_maps`).
- `batch_inputs` returns `(data, static)` instead of one tensor, because the
  finer map cannot be appended pixel by pixel.
- `models/registry.build_model` puts `models/blocks/tessera_stem.TesseraStem`
  in front of the backbone: a learned k x k, stride-k `Conv2d(128, 128)` turns
  each VIIRS pixel's k x k cells into one vector, the validity mask becomes the
  fraction of valid cells, and the backbone still receives 172 channels.
- Names get `_tessera<k>x<k>`, checkpoints record `tessera_cells_per_pixel`,
  and a checkpoint is refused by a run with another k.

k = 1 builds no stem and keeps every map in memory, exactly as before.

**`-m swinunetr3d_tessera`** [ours] reads TESSERA as context instead of as
input channels (`models/zoo/swinunetr3d_tessera.py`). Its own small CNN
(`models/blocks/tessera_context.TesseraContextEncoder`) learns features while
shrinking the map by k (one learned conv step per prime factor of k), refines
them on the 256 x 256 grid and builds a pyramid down to 8 x 8. SwinUNETR-3D runs
on the 43 VIIRS channels only, and each skip connection and the bottleneck is
conditioned on the context map of its size by FiLM (features x (1 + scale) +
shift, the same for every day). The FiLM projections start at zero, so the
untrained model is exactly a 43-channel SwinUNETR-3D; registered with
`takes_tessera=True`, it gets no generic stem and works for any k, including 1.
`--model-option context_channels=N` sets the context width (default 32).

Scoring never disturbs training: evaluation runs in eval mode without
augmentation, and its loaders use a private random generator, so a run with
`--test-every 1` ends with exactly the weights of one without it
(`tests/test_e2e.py`).

The model itself is explained from scratch in `docs/swinunetr3d/`.

## Adding a model

```bash
python -m firepred.cli.new_model mymodel        # creates firepred/models/zoo/mymodel.py
# edit it: layers in __init__, forward line by line
python -m firepred.cli.check model -m mymodel   # shapes, parameters, one training step
python -m firepred.cli.train -m mymodel ...     # trains; W&B and checkpoint names follow
```

Contract: `forward(x)` takes `(B, C, T, H, W)` and returns logits
`(B, K, T, H, W)` (averaged over T by the trainer) or `(B, K, H, W)`.
Builders receive a `ModelSpec` (`models/registry.py`): `in_channels`,
`num_classes`, `ts_length`, `image_size`, `num_heads`, `feature_size`, plus
`options` for anything model-specific, set with `--model-option key=value`
(notebook: `MODEL_OPTIONS`). `tests/test_models.py` builds every registered
model and checks the contract, so a new file is tested for free.

Bricks: `models/blocks/` (window ops, Swin window attention, patch merging, a
Swin stage, the full Swin encoder) and MONAI's `monai.networks.blocks`.

SwinUNETR-3D keeps its own class rather than a generic encoder/decoder
wrapper: its attribute names are checkpoint keys and its layer creation order
is the random-initialization order, so old checkpoints and results depend on them.

## Where to change things

| To change | Edit |
|---|---|
| Window length / stride, repos, crop, splits, channels, normalization stats | `firepred/config/dataset.py` |
| Training defaults (seed, loss weights, W&B) | `firepred/config/training.py` |
| How a window is augmented / encoded | `firepred/data/transforms.py` |
| What the target is | `firepred/data/windows.py` (then rebuild the dataset) |
| NIFC perimeters in the labels | `--train-val-label-sel 0/1` (and `--ignore-roi-label-sel`) on `cli.dataset process`; recorded in the repo's `window_config.json` |
| A notebook step | its master function in `firepred/data/windows_master.py` or `tessera_master.py` |
| A model | its file in `firepred/models/zoo/` |
| The loss or a score | `firepred/training/objectives.py` |
| A setting or a derived name | `firepred/training/config.py` (`RunConfig`) and, for its flag, `training/flags.py` |
| What is logged | `firepred/training/tracker.py` |
| The epoch loop | `firepred/training/trainer.py` (`Trainer.fit`) |

## Tests

```bash
pytest              # unit, golden and end-to-end tests, ~2 min
pytest -m slow      # the end-to-end run with the real SwinUNETR-3D
```

Run from the repository root: the package is used without installation.

- `tests/golden/`: `cases.py` computes every deterministic output (window
  archives, TESSERA grids, the Hugging Face pipeline against an offline fake,
  dataset items, transforms, every model's seeded forward pass and Adam step,
  losses, metrics, schedules, names, checkpoints, one training epoch,
  validation, test scoring, CLI options); `values/*.json` holds what they must
  equal. When an API moves, only the case body changes. To accept an
  intentional change: `python tests/golden/capture.py --group NAME --out
  tests/golden/values --reason "..."`.
- `tests/test_e2e.py`: whole CLI runs with a fake W&B: what gets logged, which
  checkpoint is kept, that `-test` reproduces the final scores, and that a
  seeded run is deterministic.
- `tests/support/`: synthetic data, fake Hugging Face and W&B, a tiny model.
