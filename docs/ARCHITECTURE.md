# Architecture

`firepred` predicts which pixels of a wildfire burn **tomorrow**, from six
days of satellite, weather and terrain data (the TS-SatFire prediction task).
The code is a hierarchy that builds toward the model: each level uses only
the levels before it, so any file can be understood by reading downward.

```
                    config/          constants: channels, dataset, paths, training
                       │
          ┌────────────┼──────────────────┐
          ▼            ▼                  ▼
        data/        models/            (prep/ ──► hub/)   dataset building, offline
  archives → tensors  blocks → zoo        GeoTIFF → windows → Hugging Face
          │            │
          └─────┬──────┘
                ▼
            training/       loss, metrics, loop, evaluation, W&B, checkpoints
                │
                ▼
              cli/          python -m firepred.cli.{train,dataset,check,new_model}
```

| Package | Level | Imports from | Job |
|---|---|---|---|
| `config/` | 0 | – | Every project-wide constant, defined once |
| `data/` | 1 | config | Window archives on disk → model-ready tensors |
| `models/` | 2 | config | Reusable blocks, one file per model, the registry |
| `prep/` | 1 | config, data | Raw GeoTIFFs → window arrays (needs rasterio) |
| `hub/` | 2 | config, data, prep | Hugging Face upload/download, local staging |
| `training/` | 3 | config, data, models | One experiment: loss, loop, evaluation, W&B |
| `cli/` | 4 | everything | Thin argparse entry points |

`tests/test_layers.py` enforces the "Imports from" column, keeps every
`__init__.py` import-free, and checks that training never imports
`prep/`/`hub/` (Kaggle installs no geospatial packages).

## Design rules

1. **Layers.** Imports only go down the table above.
2. **One home per constant.** Project-wide constants live in `firepred/config/`.
   Functions take them as default arguments, CLI flags default to them, and
   notebook variables map 1:1 to CLI flags. The one exception is a model's
   own architecture constants, which live at the top of its `models/zoo/` file,
   so adding a model never touches config. Module-private details may stay
   private to their module as `_NAME`.
3. **Plain functions by default.** Classes only where state is real:
   `FireDataset`, `nn.Module`s, `DatasetStreamer`, `DatasetLoader`,
   `BestCheckpoints`. No inheritance just to reuse code.
4. **One file per concept, named as a noun.** Every module starts with a
   docstring: what it does, its level, who calls it.
5. **Provenance tags in comments.** `[paper]` follows the TS-SatFire paper
   text; `[released]` follows the released TS-SatFire code; `[ours]` is in
   neither. Help strings keep their "NEW, not in the paper" wording.
6. **Moves never change behavior.** `tests/equivalence/` proves a refactor is
   bit-identical; a behavior change is its own labelled commit (see
   `docs/KNOWN_ISSUES.md`).

## Follow one training step

(For the whole program in execution order, dataset build to final test score,
with every input-dependent branch marked, see [EXECUTION_PATH.md](EXECUTION_PATH.md);
as sequence diagrams, [SEQUENCE_DIAGRAMS.md](SEQUENCE_DIAGRAMS.md).)

`python -m firepred.cli.train -m swinunetr3d ... --data-root D` runs
`cli/train.py: parse_args` → `training/run.py: run(args)`:

1. **Seed** numpy and torch (`run`). Everything random after this point is
   fixed: the order of the steps below is part of the result.
2. **W&B run** (`training/tracking.configure_wandb`; names from `training/naming.py`).
   **Device** (`training/accelerator.resolve_device`): `--device auto` (default) is the GPU
   when visible, else the CPU; `--device tpu` is one TPU device through PyTorch/XLA, whose
   lazily recorded steps run at `accelerator.mark_step` in the epoch loop.
3. **Model**: `models/registry.build_model("swinunetr3d", ModelSpec(...))`
   imports `models/zoo/swinunetr3d.py`, whose `build(spec)` returns a
   `SwinUNETR`, wrapped in `nn.DataParallel`.
4. **Loss and optimizer**: `training/losses.build_criterion`, `training/optim.py`.
5. **Data**: `data/dataset.FireDataset(D/train, augment=True)`. Item `i` is
   one window from `data/archive.open_fire`, run through
   `data/transforms.py` (normalize → augment → preprocess) into
   `{"data": (43, 6, 256, 256), "labels": (2, 256, 256)}`.
   `data/loader.make_dataloader` batches them.
6. **Epoch** (`training/loop.train_model`):
   - `train_one_epoch`: `data/loader.batch_inputs` (appends TESSERA on the
     GPU), `models/registry.forward_model` (the model's per-day logits averaged
     over the 6 days → (B, 2, 256, 256)), `training/losses.compute_loss`,
     backward, Adam step, `training/metrics.py` scores.
   - `validate`: the same without gradients.
   - W&B log; `training/checkpoints.BestCheckpoints` keeps the lowest
     validation loss; optional LR decay and periodic test scores.
7. **Test** (`training/evaluation.evaluate_model`): reload the best
   checkpoint (`training/checkpoints.read_checkpoint`, which refuses to run
   pickled code unless the checkpoint is trusted), score each 2021 test fire,
   average over fires, log.

The model itself is explained from scratch in `docs/swinunetr3d/`.

## Adding a model

```bash
python -m firepred.cli.new_model convlstm_unet      # creates firepred/models/zoo/convlstm_unet.py
# edit it: layers in __init__, forward line by line
python -m firepred.cli.check model -m convlstm_unet  # shapes, parameters, one training step
python -m firepred.cli.train -m convlstm_unet ...    # trains; W&B and checkpoint names follow
```

Contract: `forward(x)` takes `(B, C, T, H, W)` and returns logits
`(B, K, T, H, W)` (averaged over T by the trainer) or `(B, K, H, W)`.
Builders receive a `ModelSpec` (`models/spec.py`): `in_channels`,
`num_classes`, `ts_length`, `image_size`, `num_heads`, `feature_size`, plus
`options` for anything model-specific, set with `--model-option key=value`
(notebook: `MODEL_OPTIONS`). `tests/test_models.py` builds every registered
model and checks the contract, so a new file is tested for free.

Bricks: `models/blocks/` (window ops, Swin window attention, patch merging, a
Swin stage, the full Swin encoder) and MONAI's `monai.networks.blocks`.

This is the pattern used by torchvision (`register_model`/`get_model`), timm
(`@register_model`/`create_model`, defaults beside the model) and
MMSegmentation (registry plus reusable components). The repo deliberately
skips a config framework (Hydra/MMEngine) and the `src/` layout: Kaggle and
Colab run straight from a fresh clone with `python -m`, no install.

SwinUNETR-3D keeps its own class rather than a generic encoder/decoder
wrapper: its attribute names are checkpoint keys and its layer creation order
is the random-initialization order, so old checkpoints and results depend on them.

## Where to change things

| To change | Edit |
|---|---|
| Window length / stride, repos, crop, splits | `firepred/config/dataset.py` |
| Channel meaning, normalization stats | `firepred/config/channels.py` |
| Training defaults (seed, loss weights, W&B) | `firepred/config/training.py` |
| How a window is augmented / encoded | `firepred/data/transforms.py` |
| What the target is | `firepred/prep/windows.py` (then rebuild the dataset) |
| NIFC perimeters in the labels | `--train-val-label-sel 0/1` (and `--ignore-roi-label-sel`) on `cli.dataset process`; recorded in the repo's `window_config.json` |
| A model | its file in `firepred/models/zoo/` |
| The loss | `firepred/training/losses.py` |
| What is logged | `firepred/training/tracking.py`, `training/loop.py` |

## Tests

```bash
pytest                                                    # micro tests, ~20 s
python tests/equivalence/compare.py --baseline <checkout of 6ebc222> --tier quick   # ~30 s
python tests/equivalence/compare.py --baseline <checkout of 6ebc222> --tier hub     # ~1 min, offline fake Hugging Face
python tests/equivalence/compare.py --baseline <checkout of 6ebc222> --tier full --cross-test   # ~20 min
```

Run from the repository root (or set `PYTHONPATH` to it): the package is used
without installation.

`tests/equivalence/README.md` explains the bit-identity harness.
`legacy/` holds the paper's unused code (other tasks, TensorFlow models);
see `legacy/README.md`.
