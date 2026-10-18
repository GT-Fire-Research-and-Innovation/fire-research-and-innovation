# The execution path: one trunk, many branches

The program can take many routes, depending on its inputs: GPU or TPU, which
model, which loss, with or without TESSERA, training or `-test`. This document
follows the busiest route from start to finish, the **trunk**, in the order the
code runs. Where an input would send the code down a different route, a
**branch** box says what changes and where, then the walk returns to the trunk.

Every stop names the file and line that runs at that moment. Read it top to
bottom once; after that, use the branch index at the end to jump to the stop
where a given input changes things.

The same route drawn as calls between files: [SEQUENCE_DIAGRAMS.md](SEQUENCE_DIAGRAMS.md).

Line numbers were checked against the code on 2026-10-17. If an edit shifts
them, search for the quoted code in that file.

---

## The trunk in one picture

```
 STAGE 0  build the dataset (once, offline)       cli/dataset.py  -> hub/ -> prep/ -> data/archive.py
    raw GeoTIFFs on HF  ──►  one p<id>.npz per fire  ──►  data.zip on HF
                                                │
 STAGE 1  Kaggle notebook (every experiment)    ▼  notebooks/fire-research-swinunetr.ipynb
    settings ─► install ─► clone code ─► W&B login ─► data.zip ─► unzip ─► launch trainer
                                                │
 STAGE 2  command line                          ▼  cli/train.py
                                                │
 STAGE 3  set up the run                        ▼  training/run.py
    seed ─► settings ─► W&B run ─► device ─► model ─► loss/optimizer
                                                │
 STAGE 4  data objects                          ▼  data/dataset.py, data/loader.py
    FireDataset(train/val) ─► shape check ─► DataLoaders
                                                │
 STAGE 5  100 epochs                            ▼  training/loop.py
    ┌── train one epoch: load window ─► model ─► loss ─► backward ─► Adam ──┐
    │   validate ─► log ─► keep best checkpoint ─► (every 10) test peek     │
    └───────────────────────────────── repeat ──────────────────────────────┘
                                                │
 STAGE 6  test the best checkpoint              ▼  training/checkpoints.py, training/evaluation.py
    load best ─► score each 2021 fire ─► average over fires ─► log to W&B
```

## What "the trunk" means here

The trunk is the Kaggle notebook as shipped, with one change: `USE_TESSERA =
False`, the paper's baseline. (The notebook currently ships `USE_TESSERA =
True`; set it to `False` to follow this page exactly, or follow the TESSERA
branches.)

| Setting | Trunk value | Where it is set |
|---|---|---|
| accelerator | GPU (`ACCELERATOR = 'gpu'`, so `--device cuda`) | notebook cell 1 |
| model | `swinunetr3d` | notebook cell 1 → `-m` |
| loss | `dicece`, class weights (1.0, 457.1) | notebook cell 1 → `--loss`, `--ce-weight` |
| input | 27 stored channels → 43 model channels, 6 days, 256 × 256 | `-ts 6`, `-nc 43` |
| batch, epochs, learning rate | 8, 100, 0.001 constant | `-b`, `--epochs`, `-lr` |
| heads, feature size | 3, 36 | `-nh`, `-ed` |
| data loading | 2 workers, archives read directly (`--unzip-dir off`) | `--num-workers`, `--unzip-dir` |
| extras | test-fire check every 10 epochs; best checkpoints uploaded to W&B | `--test-every 10`, `--upload-checkpoints` |

---

## Stage 0: Building the dataset (once, offline)

This stage runs once per window setting (here, 6 input days, a new window every
day), long before any training. It turns raw satellite GeoTIFFs into one
archive per fire, and it explains what the trainer later reads.

Command: `python -m firepred.cli.dataset process -ts 6 -it 1`

### 0.1 The command line picks the target repo

- [`cli/dataset.py:180`](../firepred/cli/dataset.py#L180) `main()` parses the flags.
- [`cli/dataset.py:185`](../firepred/cli/dataset.py#L185): no `--window-repo` was given, so the
  repo is named after the window shape: `window_repo_for(6, 1)` →
  `SamuelWu318/ts-satfire-processed-6-1` ([`config/dataset.py:38`](../firepred/config/dataset.py#L38)).
  Each window shape gets its own repo, because archive file names do not record it.
- [`cli/dataset.py:193`](../firepred/cli/dataset.py#L193): `process` calls `process_data`.

### 0.2 `process_data`: settings, split, label rule

[`hub/publish_windows.py:191`](../firepred/hub/publish_windows.py#L191) `process_data(...)`

1. [`:224`](../firepred/hub/publish_windows.py#L224) creates a `DatasetStreamer` (logs in, creates the repo if needed).
2. [`:233`](../firepred/hub/publish_windows.py#L233) `ensure_window_config` →
   [`hub/repos.py:131`](../firepred/hub/repos.py#L131). This writes `window_config.json` (window
   length, label rules, crop, channels, split) to a new repo. For an existing repo
   it checks that the settings match.
3. [`:242`](../firepred/hub/publish_windows.py#L242) `obtain_ids` → [`hub/splits.py:60`](../firepred/hub/splits.py#L60):
   downloads the ROI CSVs (fire metadata). 2017–2020 fires go to train, except
   the fixed `VAL_IDS`, which go to val ([`splits.py:80`](../firepred/hub/splits.py#L80)).
   2021 fires go to test ([`splits.py:87`](../firepred/hub/splits.py#L87)).
4. [`:247`](../firepred/hub/publish_windows.py#L247) picks a label rule per fire. Train/val use
   `label_sel = 0` (active fire + NIFC perimeters, as in the paper). Test fires use
   the `label_sel` column of the 2021 ROI CSV, the paper's per-fire choice.
5. [`:260`](../firepred/hub/publish_windows.py#L260) `run_split` for train, val, then test.

> **↳ Branch: settings mismatch.** If the repo's `window_config.json` disagrees with this run,
> [`repos.py:131`](../firepred/hub/repos.py#L131) raises `ValueError` before anything is built.
> `--overwrite` instead rebuilds every fire and rewrites the file.
>
> **↳ Branch: label rule.** `--train-val-label-sel 1` uses active fire only for train/val
> (as in the released code). `--ignore-roi-label-sel` applies that one rule to test fires too.

### 0.3 `run_split`: one fire at a time

[`hub/publish_windows.py:49`](../firepred/hub/publish_windows.py#L49) `DatasetStreamer.run_split`

1. [`:95`](../firepred/hub/publish_windows.py#L95) skips fires already in the repo, so an interrupted run resumes.
2. [`:109`](../firepred/hub/publish_windows.py#L109) downloads this fire's GeoTIFFs (VIIRS day and night, FirePred) to a temp folder.
3. [`:123`](../firepred/hub/publish_windows.py#L123) `build_fire_windows` (0.4) builds the windows.
4. [`:139`](../firepred/hub/publish_windows.py#L139) `write_fire_archive` (0.5) writes `p<id>.npz`.
5. Temp files are deleted, and every 10 fires the staging folder is uploaded
   ([`:160`](../firepred/hub/publish_windows.py#L160), remainder at [`:179`](../firepred/hub/publish_windows.py#L179)).

### 0.4 `build_fire_windows`: GeoTIFFs → windows

[`prep/windows.py:35`](../firepred/prep/windows.py#L35)

1. [`:51`](../firepred/prep/windows.py#L51) lists the fire's days, in date order (one `VIIRS_Day` file per day).
2. Every frame is cropped to the same 256 × 256 square (rows/cols 128–384), and
   the `(windows, 27, 6, 256, 256)` array is allocated once.
3. [`:107`](../firepred/prep/windows.py#L107) slides a window: window *k* starts on day *k* and reads
   6 input days plus the next day as its target. Then, for each day:
   - [`:116`](../firepred/prep/windows.py#L116) reads the day bands, [`:129`](../firepred/prep/windows.py#L129) the night bands (zeros if
     missing) and [`:170`](../firepred/prep/windows.py#L170) the 19 FirePred bands (weather, fuel, terrain, …).
   - [`:181`](../firepred/prep/windows.py#L181) stacks 6 + 2 + 19 = **27 channels**, cropped. NaNs become 0.
   - [`:214`](../firepred/prep/windows.py#L214) gives the heat bands (day/night I4, I5) a running maximum,
     so a pixel that was hot earlier in the window still looks hot.
   - [`:228`](../firepred/prep/windows.py#L228) adds active-fire (band 6) and NIFC-perimeter (band 7)
     pixels to the burned-so-far masks. [`:265`](../firepred/prep/windows.py#L265) `label_sel`
     decides whether NIFC counts.
   - [`:288`](../firepred/prep/windows.py#L288) on the target day: label = burned now AND NOT burned on
     the last input day, so only the **newly burned pixels** (next-day fire growth).
4. [`:306`](../firepred/prep/windows.py#L306) rewinds the running state to the next window's start, so
   overlapping windows never see days after their own start.
5. Checks for NaN/Inf, reports how many windows contain new fire, and returns
   `(N, 27, 6, 256, 256)` inputs and `(N, 256, 256)` targets.

### 0.5 `write_fire_archive`: one archive per fire

[`data/archive.py:40`](../firepred/data/archive.py#L40). Writes `format_version`, `num_windows`,
`label_sel`, then one `data_###` and one `label_###` member per window
([`:64`](../firepred/data/archive.py#L64)), each compressed on its own at zlib level 1
([`:79`](../firepred/data/archive.py#L79)). The trainer can then decompress one window without
touching the rest of the fire.

### 0.6 Bundle into `data.zip`

`python -m firepred.cli.dataset download -ts 6 -it 1` gathers the archives into
`./data/{train,val,test}` ([`hub/download.py:36`](../firepred/hub/download.py#L36)).
`python -m firepred.cli.dataset upload-zip` then zips them, without a second compression, and
uploads `data.zip` to the window repo ([`hub/zips.py:51`](../firepred/hub/zips.py#L51)).
This one zip is what the notebook downloads.

> **↳ Branch: TESSERA embeddings** (needed only for `USE_TESSERA = True`).
> `cli.dataset tessera` → [`hub/publish_tessera.py:22`](../firepred/hub/publish_tessera.py#L22) builds one
> `tessera/p<id>.npz` per fire from the year *before* the fire
> ([`prep/tessera_grid.py:211`](../firepred/prep/tessera_grid.py#L211)). It skips 2017 fires, for which no
> prior-year TESSERA exists. `cli.dataset tessera-zip` →
> [`publish_tessera.py:160`](../firepred/hub/publish_tessera.py#L160) bundles them, with the drop list
> `excluded_npz.txt`, into `tessera.zip`.

---

## Stage 1: The Kaggle notebook

`notebooks/fire-research-swinunetr.ipynb` (local only; not in git). It holds
settings only; all logic comes from the cloned `firepred` package.

| Cell | What runs |
|---|---|
| 1 | All settings (table above). Asserts they are consistent and prints them. |
| 3 | Checks for a GPU, prints PyTorch/GPU info, pip-installs MONAI 1.3.2 etc. Pins torch so pip can't replace Kaggle's CUDA build. |
| 5 | `git clone` of `REPO_BRANCH`. Checks that `python -m firepred.cli.train --help` lists every flag the notebook passes. Adds the clone to `sys.path`. |
| 7 | W&B login, plus a throw-away permission-check run that is then deleted. |
| 9 | Downloads `data.zip` from `window_repo_for(6, 1)`, then [`hub/staging.py:75`](../firepred/hub/staging.py#L75) `extract_data`. That calls [`extract_zip`](../firepred/hub/staging.py#L47) (disk-space check, path-escape check, unzips once per session) and `find_data_root` (finds the one folder holding `train/ val/ test/`). |
| 13 | Load-time check: `python -m firepred.cli.check timing` ([`cli/check.py:184`](../firepred/cli/check.py#L184)). |
| 15 | Builds the training command from the settings, saves it with the code version beside the outputs, and runs it as a subprocess. **The rest of this page happens inside that subprocess.** |

> **↳ Branch: TPU** (`ACCELERATOR = 'tpu'`). Cell 3 checks that `torch_xla` is installed
> without opening the TPU (only one process may hold it, and the trainer needs it). It also
> sets `PJRT_DEVICE=TPU`. Cell 15 passes `--device tpu`. Continues at stop 3.7.
>
> **↳ Branch: attached Kaggle data.** `ATTACHED_DATA_ROOT` skips the download and unzip
> and uses that folder. `ATTACHED_ZIP` skips only the download.
>
> **↳ Branch: TESSERA** (`USE_TESSERA = True`). Cell 9 also fetches `tessera.zip` and builds a
> symlink view ([`hub/staging.py:80`](../firepred/hub/staging.py#L80)). The view holds every kept window
> archive plus `tessera/`, and leaves out the 2017 fires listed in `excluded_npz.txt`.
> Cell 15 passes `--tessera` and `-nc 172`. With `TESSERA_INPUTS = False`, it trains the
> matched 43-channel baseline on the same reduced set of fires.
>
> **↳ Branch: shape smoke test.** Cell 11 is commented out. Uncommented, it runs
> `cli.check data --train-step` ([`cli/check.py:148`](../firepred/cli/check.py#L148)): one real batch, one
> disposable training step.

---

## Stage 2: The command line

`python -u -m firepred.cli.train -m swinunetr3d -mode pred -b 8 -r 1 -lr 0.001 -nh 3 -ed 36 -nc 43 -ts 6 -it 1 --epochs 100 --num-workers 2 --device cuda --unzip-dir off --data-root … --upload-checkpoints --loss dicece --ce-weight 1.0 457.1 … --test-every 10`

- [`cli/train.py:39`](../firepred/cli/train.py#L39) `parse_args` turns the flags into `args`. Defaults
  come from [`config/training.py`](../firepred/config/training.py).
- [`cli/train.py:232`](../firepred/cli/train.py#L232) `main` → `run(args)`. Everything else is in
  [`training/run.py`](../firepred/training/run.py), whose steps run in a fixed order. The order is
  part of the result: model construction, shuffling and augmentation all draw
  from the same random-number streams.

---

## Stage 3: Setting up the run

[`training/run.py:62`](../firepred/training/run.py#L62) `run(args)`

### 3.1 Seed
[`run.py:65`](../firepred/training/run.py#L65) seeds numpy, torch and CUDA with 42. Everything random
after this line is reproducible.

### 3.2 Settings
[`run.py:74`](../firepred/training/run.py#L74): `--loss` was given (`dicece`), so the per-model
default is not used.

> **↳ Branch: no `--loss`.** `default_loss` ([`run.py:45`](../firepred/training/run.py#L45)) picks the
> paper's choice: `dice` for `attunet`, `dicece` for every other model.

### 3.3 Channel count
[`run.py:90`](../firepred/training/run.py#L90) computes 27 stored − 1 land-cover + 17 one-hot =
**43** ([`config/channels.py:127`](../firepred/config/channels.py#L127)).
[`run.py:92`](../firepred/training/run.py#L92) checks that `-nc 43` agrees.

> **↳ Branch: TESSERA.** `--tessera` gives 43 + 128 embedding bands + 1 mask = **172**.

### 3.4 Where the data is
[`run.py:103`](../firepred/training/run.py#L103): `--data-root` → `train/`, `val/`, `test/`.

> **↳ Branch: `--unzip-dir auto` or a path** ([`run.py:116`](../firepred/training/run.py#L116)).
> [`data/archive.py:227`](../firepred/data/archive.py#L227) `unzip_data_root` unpacks every archive once
> into `p<id>/` folders of `.npy` files. Every later read then skips decompression, and the
> arrays are byte-identical. It costs about 5.5× the disk.

### 3.5 Names
[`run.py:134`](../firepred/training/run.py#L134): the loss and LR configs (recorded in W&B and in
every checkpoint), plus the run tag `_dicece` that ends checkpoint names
([`training/naming.py`](../firepred/training/naming.py)).

### 3.6 W&B run
[`run.py:147`](../firepred/training/run.py#L147) → [`training/tracking.py:56`](../firepred/training/tracking.py#L56)
`configure_wandb`: `wandb.init`, chart definitions against the epoch axis, run name
`train_pred_swinunetr3d_num_heads_3_hidden_size_36_batchsize_8_dicece`, and the
hyperparameters in the run config.

### 3.7 Device
[`run.py:165`](../firepred/training/run.py#L165) → [`training/accelerator.py:43`](../firepred/training/accelerator.py#L43)
`resolve_device("cuda")` → `torch.device("cuda")`.

> **↳ Branch: TPU** (`--device tpu`). `resolve_device` imports `torch_xla` and returns the XLA
> device. `seed_device` seeds the TPU's own random generator, and W&B records `device: tpu`.
> Three later stops behave differently: 5.1e and 5.3 add `mark_step` graph cuts, 5.5
> saves through `xm.save`, and 6.2 loads onto the CPU first.
>
> **↳ Branch: `--device auto`** (the CLI default): CUDA if visible, else CPU. Same as above on a GPU machine.

### 3.8 Model
[`run.py:177`](../firepred/training/run.py#L177) `build_model("swinunetr3d", ModelSpec(43, 2, 6, 3, 36, options))`

1. [`models/registry.py:55`](../firepred/models/registry.py#L55) `build_model` → `_import_zoo`
   ([`:43`](../firepred/models/registry.py#L43)) imports every file in `models/zoo/`. Each file's
   `@register_model("…")` decorator adds its builder to the registry.
2. The `swinunetr3d` builder ([`models/zoo/swinunetr3d.py:290`](../firepred/models/zoo/swinunetr3d.py#L290))
   builds `SwinUNETR`:
   - patch (1, 2, 2): each token = 1 day × 2 × 2 pixels
   - window (6, 4, 4): all 6 days × 4 × 4 tokens
   - depths (2, 2, 2, 2), 3 heads per stage, feature size 36
   - attention `v1`, batch norm, no dropout
   - `temporal_downsample=False`: all 6 days are kept at every stage (the paper)
3. [`run.py:197`](../firepred/training/run.py#L197) wraps it in `nn.DataParallel`, which splits each batch
   across the visible GPUs (Kaggle T4 × 2: 4 windows each). Then `.to(device)`.

> **↳ Branch: another model** (`-m unet3d`, `attunet`, `unetr3d`, `swinunetr3d_tpool`, or your own).
> The registry calls that file's `build(spec)` instead. Everything else on this page
> stays the same: every model takes `(B, C, T, H, W)` and returns logits
> `(B, 2, T, H, W)` or `(B, 2, H, W)`. New models start from `python -m firepred.cli.new_model NAME`.
>
> **↳ Branch: `--swin-temporal-downsample`.** Patch merging also halves the days (6 → 3), as the
> released code does ([`models/blocks/patch_merging.py:58`](../firepred/models/blocks/patch_merging.py#L58)).
>
> **↳ Branch: `--model-option KEY=VALUE`.** It lands in `spec.options` for the builder to read,
> and W&B records it.

### 3.9 Loss, metrics, optimizer
- [`run.py:203`](../firepred/training/run.py#L203) → [`training/losses.py:44`](../firepred/training/losses.py#L44)
  `build_criterion` → [`:69`](../firepred/training/losses.py#L69) MONAI `DiceCELoss`: sigmoid Dice +
  cross entropy, 1:1, class weights (1.0, 457.1). MONAI applies the weights to both
  terms, so the loss starts near 220.
- `build_post_transform` ([`training/metrics.py:17`](../firepred/training/metrics.py#L17)): sigmoid, then
  threshold 0.5, turning logits into 0/1 masks.
- [`run.py:214`](../firepred/training/run.py#L214) Adam at a constant 0.001
  ([`training/optim.py:10`](../firepred/training/optim.py#L10)).
- [`run.py:215`](../firepred/training/run.py#L215) a gradient scaler ([`optim.py:19`](../firepred/training/optim.py#L19)),
  enabled on CUDA. There is no autocast, so training is plain float32.

> **↳ Branch: `--loss dice`.** Sigmoid Dice only (the released code's loss); no loss parts are logged.
>
> **↳ Branch: `--loss dicefocal`.** `lambda_dice` × Dice + `lambda_focal` × focal loss on the fire
> channel, with no cross entropy and no class weight. W&B logs both parts. Notebook:
> `USE_DICE_FOCAL = True`.

### 3.10 Checkpoint name
[`run.py:247`](../firepred/training/run.py#L247) builds the template
`model_swinunetr3d_mode_pred_num_heads_3_hidden_size_36_batchsize_8_checkpoint_epoch_{epoch}_nc_43_ts_6_dicece.pth`.

---

## Stage 4: Data objects

[`run.py:253`](../firepred/training/run.py#L253) `if not args.run_test:`. The trunk trains.

> **↳ Branch: `-test`.** Stages 4 and 5 are skipped entirely. Jump to 6.2 with
> `--checkpoint` (or the default name under `saved_models/`).

### 4.1 FireDataset for train and val
[`run.py:266`](../firepred/training/run.py#L266) → [`data/dataset.py:50`](../firepred/data/dataset.py#L50) `FireDataset.__init__`:

1. [`:64`](../firepred/data/dataset.py#L64) `find_fire_sources` ([`data/archive.py:128`](../firepred/data/archive.py#L128))
   lists `p<id>.npz` (or unzipped `p<id>/`) in the split folder.
2. [`:114`](../firepred/data/dataset.py#L114) builds the **global index**: open each archive's zip
   directory (no decompression), read `num_windows` and record one `(fire, window number)` pair
   per window. `len(dataset)` = all windows of all fires, so shuffling mixes fires.
3. [`:166`](../firepred/data/dataset.py#L166) sets up the normalizer: fixed per-channel mean/std from
   [`config/channels.py`](../firepred/config/channels.py), skipping the 3 angle channels and land cover.
   Also the land-cover one-hot table.

Train gets `use_augmentations=True`, val `False`.

> **↳ Branch: TESSERA.** First, [`run.py:256`](../firepred/training/run.py#L256) computes per-band
> mean/std from the **training** fires only ([`data/tessera.py:41`](../firepred/data/tessera.py#L41)).
> Then [`dataset.py:80`](../firepred/data/dataset.py#L80) loads every fire's embedding map once, up front.

### 4.2 Fail-fast shape check
[`run.py:287`](../firepred/training/run.py#L287) → [`data/loader.py:32`](../firepred/data/loader.py#L32) loads
`val_dataset[0]` and checks that its channels (43) match the model before any training time is spent.

### 4.3 DataLoaders
[`run.py:294`](../firepred/training/run.py#L294) → [`data/loader.py:12`](../firepred/data/loader.py#L12): batches of
8, shuffled for train, fixed order for val. With 2 workers it also sets
`prefetch_factor=1` and `persistent_workers=True` ([`loader.py:25`](../firepred/data/loader.py#L25)),
so worker RAM stays bounded.

> **↳ Branch: `--num-workers 0`** (the CLI default). Windows load in the main process, with no prefetching.

### 4.4 Periodic test peek
[`run.py:312`](../firepred/training/run.py#L312): `--test-every 10` and test archives exist, so a
`score_test_fires` function is prepared now. It runs at stop 5.7.

> **↳ Branch: `--test-every 0`** (the CLI default). Nothing is prepared and 5.7 never runs.

---

## Stage 5: The epoch loop

[`run.py:322`](../firepred/training/run.py#L322) → [`training/loop.py:218`](../firepred/training/loop.py#L218)
`train_model`. [`loop.py:255`](../firepred/training/loop.py#L255) `for epoch in range(100):`

### 5.1 Train one epoch
[`loop.py:257`](../firepred/training/loop.py#L257) → [`loop.py:32`](../firepred/training/loop.py#L32)
`train_one_epoch`. `model.train()`, then for every batch:

**a. A worker builds each window of the batch.** The DataLoader asks for
`dataset[i]` → [`data/dataset.py:178`](../firepred/data/dataset.py#L178) `__getitem__`:

| Step | Where | Shape after |
|---|---|---|
| read one window: open the archive, decompress only `data_###` + `label_###`, check shapes | [`dataset.py:183`](../firepred/data/dataset.py#L183) → `load_data` [`:208`](../firepred/data/dataset.py#L208) | x `(27, 6, 256, 256)`, label `(256, 256)` |
| target as two channels: background, fire | [`dataset.py:281`](../firepred/data/dataset.py#L281) | y `(2, 256, 256)` |
| standardize channels | [`dataset.py:186`](../firepred/data/dataset.py#L186) → [`data/transforms.py:34`](../firepred/data/transforms.py#L34) | same |
| random h-flip / v-flip / 90° rotation, with wind and aspect angles turned to match | [`dataset.py:194`](../firepred/data/dataset.py#L194) → [`transforms.py:68`](../firepred/data/transforms.py#L68) | same |
| angles → sin(angle); land cover → 17 one-hot planes | [`dataset.py:198`](../firepred/data/dataset.py#L198) → [`transforms.py:120`](../firepred/data/transforms.py#L120) | x `(43, 6, 256, 256)` |

The DataLoader stacks 8 of these: `data (8, 43, 6, 256, 256)`, `labels (8, 2, 256, 256)`.

> **↳ Branch: validation and test windows** skip the augmentation row.
>
> **↳ Branch: TESSERA.** [`dataset.py:191`](../firepred/data/dataset.py#L191) `load_tessera` returns the
> fire's normalized `(129, 256, 256)` map once, as `"static"`. It is flipped and rotated along
> with x, and *not* repeated over the days here, to save host RAM.

**b. Move to the GPU.** [`loop.py:49`](../firepred/training/loop.py#L49) →
[`data/loader.py:66`](../firepred/data/loader.py#L66) `batch_inputs`.

> **↳ Branch: TESSERA.** Here, on the GPU, the static map is repeated across the 6 days and
> appended after the 43 channels: `(8, 172, 6, 256, 256)`.

**c. Forward pass.** [`loop.py:56`](../firepred/training/loop.py#L56) zeroes the gradients.
[`loop.py:57`](../firepred/training/loop.py#L57) → [`models/registry.py:63`](../firepred/models/registry.py#L63)
`forward_model` → `DataParallel` → [`models/zoo/swinunetr3d.py:268`](../firepred/models/zoo/swinunetr3d.py#L268)
`SwinUNETR.forward`. Shapes are per GPU, with B = 4 on 2 GPUs and F = 36:

- **Encoder** [`swinunetr3d.py:271`](../firepred/models/zoo/swinunetr3d.py#L271) → [`models/blocks/swin_encoder.py:156`](../firepred/models/blocks/swin_encoder.py#L156):
  1. [`swin_encoder.py:159`](../firepred/models/blocks/swin_encoder.py#L159) patch embedding (a strided convolution) turns each 1 × 2 × 2
     patch into a 36-number token: `(B, 36, 6, 128, 128)`.
  2. Four stages, each a `BasicLayer.forward` ([`models/blocks/swin_block.py:287`](../firepred/models/blocks/swin_block.py#L287)):
     - [`swin_block.py:299`](../firepred/models/blocks/swin_block.py#L299) builds the shifted-window mask once
       ([`models/blocks/window_ops.py:138`](../firepred/models/blocks/window_ops.py#L138)). The time axis (6) fits in one window,
       so it is never shifted. The shift is (0, 2, 2).
     - [`swin_block.py:300`](../firepred/models/blocks/swin_block.py#L300) two Swin blocks
       ([`SwinTransformerBlock.forward`](../firepred/models/blocks/swin_block.py#L193)). Each block does:
       LayerNorm and padding;
       (second block only) roll by half a window ([`:156`](../firepred/models/blocks/swin_block.py#L156));
       cut into windows of 6 × 4 × 4 = 96 tokens ([`:165`](../firepred/models/blocks/swin_block.py#L165) →
       [`window_ops.py:28`](../firepred/models/blocks/window_ops.py#L28));
       attention inside each window ([`:169`](../firepred/models/blocks/swin_block.py#L169) →
       [`models/blocks/window_attention.py:113`](../firepred/models/blocks/window_attention.py#L113):
       query·key scores + learned relative-position bias + mask → softmax → weighted
       sum of values); stitch the windows back ([`:171`](../firepred/models/blocks/swin_block.py#L171));
       residual add ([`:201`](../firepred/models/blocks/swin_block.py#L201)); then an MLP with its own residual.
     - [`swin_block.py:304`](../firepred/models/blocks/swin_block.py#L304) patch merging
       ([`models/blocks/patch_merging.py:85`](../firepred/models/blocks/patch_merging.py#L85), H-and-W branch) halves H and W and doubles the channels.
  3. It returns five scales, each layer-normalized: `(36, 6, 128²)`, `(72, 6, 64²)`,
     `(144, 6, 32²)`, `(288, 6, 16²)`, `(576, 6, 8²)`.
- **Skip refiners** [`swinunetr3d.py:273`](../firepred/models/zoo/swinunetr3d.py#L273): convolution blocks on the raw
  input (`encoder1`, full 256²) and on the first three Swin scales.
- **Bottleneck + decoder** [`swinunetr3d.py:278`](../firepred/models/zoo/swinunetr3d.py#L278): `encoder10` refines the
  8 × 8 scale. Then `decoder5` … `decoder1` each upsample ×2, concatenate the encoder
  features of the same size and refine them: 8² → 16² → … → 256².
- **Head** [`swinunetr3d.py:286`](../firepred/models/zoo/swinunetr3d.py#L286): a 1 × 1 × 1 convolution to 2 logits per
  pixel per day: `(B, 2, 6, 256, 256)`.
- Back in `forward_model`, the logits are **averaged over the 6 days**: `(8, 2, 256, 256)`, one next-day map.

> **↳ Branch: attention version.** `v2` (cosine attention) or `ar` (with a time mask) select
> other attention classes ([`swin_block.py:83`](../firepred/models/blocks/swin_block.py#L83)). SwinUNETR-3D is hard-wired to `v1`.

**d. Loss.** [`loop.py:58`](../firepred/training/loop.py#L58) → [`training/losses.py:95`](../firepred/training/losses.py#L95)
`compute_loss` → DiceCE branch ([`:102`](../firepred/training/losses.py#L102)): returns the total plus
its Dice and CE parts for logging.

**e. Graph cut.** [`loop.py:59`](../firepred/training/loop.py#L59) `mark_step` is a no-op on the GPU.

> **↳ Branch: TPU.** Here XLA compiles and runs the forward graph once, so the scores below
> don't recompute it. A second `mark_step`, after the Adam step, runs the backward pass.

**f. Training scores (logging only).** [`loop.py:65`](../firepred/training/loop.py#L65): sigmoid + threshold,
then per-window fire F1, background F1 and fire IoU
([`training/metrics.py:60`](../firepred/training/metrics.py#L60), [`:70`](../firepred/training/metrics.py#L70)). No gradients.

**g. Learn.** [`loop.py:72`](../firepred/training/loop.py#L72): backward, Adam step, scaler update.
A non-finite running loss stops training at once ([`loop.py:90`](../firepred/training/loop.py#L90)).

### 5.2 Log the training epoch
[`loop.py:261`](../firepred/training/loop.py#L261) `wandb.log`: loss, its Dice/CE parts, F1s and IoU, against the epoch.

### 5.3 Validate
[`loop.py:277`](../firepred/training/loop.py#L277) → [`loop.py:110`](../firepred/training/loop.py#L110) `validate`:
`model.eval()` ([`:114`](../firepred/training/loop.py#L114)), no gradients, the same forward and loss on
unaugmented val windows, the same F1/IoU, plus MONAI mean IoU / mean Dice over both classes
([`loop.py:159`](../firepred/training/loop.py#L159)). Those last two average in the background class,
so they are not the paper's numbers.

### 5.4 Log the validation epoch
[`loop.py:281`](../firepred/training/loop.py#L281) `wandb.log`, including the current learning rate.

### 5.5 Keep the best checkpoint
[`loop.py:311`](../firepred/training/loop.py#L311) → [`training/checkpoints.py:90`](../firepred/training/checkpoints.py#L90)
`BestCheckpoints.offer`. If this epoch's val loss is the lowest so far, it deletes the
previous best and calls `save_checkpoint` ([`checkpoints.py:26`](../firepred/training/checkpoints.py#L26)).
That writes weights, optimizer state and the run settings
([`:37`](../firepred/training/checkpoints.py#L37) → `accelerator.save` = `torch.save`) into the W&B run folder. Then,
because of `--upload-checkpoints`, it uploads the file as a new version of a W&B model
artifact and waits for the upload ([`:51`](../firepred/training/checkpoints.py#L51) → [`:55`](../firepred/training/checkpoints.py#L55)).

> **↳ Branch: no `--upload-checkpoints`.** The checkpoint stays local only.
>
> **↳ Branch: TPU.** `accelerator.save` → `xm.save` copies TPU tensors to the CPU first, so the
> file loads on any machine.

### 5.6 Learning-rate schedule
[`loop.py:326`](../firepred/training/loop.py#L326) `step_lr_scheduler`: with `none` there is no scheduler, so nothing happens.

> **↳ Branch: `--lr-schedule plateau | cosine`** ([`training/optim.py:24`](../firepred/training/optim.py#L24)).
> `plateau` halves the rate after 5 epochs without a better val loss. `cosine` decays smoothly to
> `--lr-min`. Run and checkpoint names gain `_lr-<schedule>`.

### 5.7 Test peek (epochs 10, 20, …, 100)
[`loop.py:329`](../firepred/training/loop.py#L329): it saves every random-number state
([`loop.py:203`](../firepred/training/loop.py#L203)), scores the 2021 test fires with the current weights
(the same function as Stage 6, without plots), restores the states and logs the
scores. Training continues exactly as if the peek had not happened, and the peek never
affects which checkpoint is kept.

### 5.8 After epoch 100
[`loop.py:345`](../firepred/training/loop.py#L345) returns the path of the lowest-val-loss checkpoint.

---

## Stage 6: Testing the best checkpoint

### 6.1 Go on to testing
[`run.py:364`](../firepred/training/run.py#L364): `--test-after-train` is on by default.

> **↳ Branch: `--no-test-after-train`.** The run ends here.

### 6.2 Load the checkpoint safely
[`run.py:380`](../firepred/training/run.py#L380) → [`training/checkpoints.py:195`](../firepred/training/checkpoints.py#L195) `load_checkpoint`
→ [`read_checkpoint`](../firepred/training/checkpoints.py#L160). It first tries `weights_only=True`, which can't run
code stored in the file. Next it allows numpy's array functions. Full unpickling comes last and only
for trusted files; the checkpoint this run just wrote counts as trusted.
[`checkpoints.py:201`](../firepred/training/checkpoints.py#L201) refuses a checkpoint whose TESSERA or
temporal-downsampling setting differs from this run's.

> **↳ Branch: `-test` with someone else's checkpoint.** It is found at [`run.py:373`](../firepred/training/run.py#L373),
> and full unpickling needs `--trust-checkpoint`.
>
> **↳ Branch: TPU.** It loads onto the CPU ([`accelerator.py:76`](../firepred/training/accelerator.py#L76)), and
> `load_state_dict` then copies the values onto the TPU.

### 6.3 Test dataset
[`run.py:389`](../firepred/training/run.py#L389): a `FireDataset` on `test/` with no augmentation and the
checkpoint's TESSERA statistics. It is shape-checked at [`:390`](../firepred/training/run.py#L390).

### 6.4 Restore weights
[`run.py:398`](../firepred/training/run.py#L398) restores the weights and the optimizer state.

### 6.5 Score the paper's way
[`run.py:405`](../firepred/training/run.py#L405) → [`training/evaluation.py:102`](../firepred/training/evaluation.py#L102) `evaluate_model`:

1. [`:123`](../firepred/training/evaluation.py#L123) groups test windows by fire.
2. [`:130`](../firepred/training/evaluation.py#L130) orders the fires as in the 2021 ROI CSV, minus the one excluded fire.
3. [`:145`](../firepred/training/evaluation.py#L145) handles one fire at a time. Its own loader feeds the forward
   ([`:166`](../firepred/training/evaluation.py#L166)); predictions are thresholded and scored per window. One overlay PNG
   per window ([`:190`](../firepred/training/evaluation.py#L190)): hits, false alarms and misses over VIIRS I4.
4. [`:202`](../firepred/training/evaluation.py#L202): a fire's score = the mean over its windows.
5. [`:218`](../firepred/training/evaluation.py#L218): the model's score = the mean over fires, so a long fire
   counts as much as a short one.

### 6.6 Final log
[`training/tracking.py:101`](../firepred/training/tracking.py#L101) `log_final_test`: logs test F1 / IoU / background F1 and
writes them to the run summary.

### 6.7 Back out
`run` returns. [`cli/train.py`](../firepred/cli/train.py) calls `wandb.finish()`, and notebook cell 15 lists the saved
checkpoints under `/kaggle/working/fire-research-output`.

---

## Branch index

Each row: the input, the first stop where the path changes, and what is different.

| Input | Branches at | Effect |
|---|---|---|
| `ACCELERATOR = 'tpu'` / `--device tpu` | 1 (cell 3), 3.7 | XLA device. Graph cuts at 5.1e and 5.3, `xm.save` at 5.5, CPU load at 6.2. One TPU core/chip only; results close to but not identical to a GPU run. |
| `--device auto` / `cpu` | 3.7 | Auto: GPU if visible, else CPU. CPU: no scaler, no GPU split. |
| `USE_TESSERA = True` / `--tessera` | 0.6, 1 (cell 9), 3.3, 4.1, 5.1a/b | 172 channels. 2017 fires dropped. Embedding stats from training fires only. Static map repeated across days on the GPU. Checkpoint names end in `_tessera`. |
| `TESSERA_INPUTS = False` (with TESSERA) | 1 (cell 15) | The same reduced set of fires, but 43 channels: the matched baseline. |
| `-m <other model>` | 3.8 | That zoo file's `build(spec)`; same I/O contract, so every other stop is unchanged. |
| `--swin-temporal-downsample` | 3.8 | Days halve in patch merging (6 → 3), as in the released code. |
| `--model-option K=V` | 3.8 | Extra builder settings in `spec.options`. |
| no `--loss` | 3.2 | `dice` for `attunet`, else `dicece`. |
| `--loss dice` / `dicefocal` | 3.9, 5.1d | Dice only / Dice + focal; different parts logged. |
| `--unzip-dir auto` | 3.4 | Archives unpacked once; same arrays, no per-read decompression. |
| `--num-workers 0` | 4.3 | Loading in the main process. |
| `--test-every 0` | 4.4, 5.7 | No test peeks. |
| `--lr-schedule plateau/cosine` | 5.6 | Learning-rate decay; names gain `_lr-…`. |
| no `--upload-checkpoints` | 5.5 | Checkpoints stay local. |
| `--no-test-after-train` | 6.1 | Run ends after training. |
| `-test` (+ `--checkpoint`, `--trust-checkpoint`) | 4 | Skips Stages 4–5 and tests the given checkpoint. |
| `--train-val-label-sel 1`, `--ignore-roi-label-sel` | 0.2 | Other target definitions, set when the dataset is built. |
| `--overwrite` (dataset) | 0.2 | Rebuilds fires and rewrites `window_config.json`. |
| `ATTACHED_DATA_ROOT` / `ATTACHED_ZIP` | 1 (cell 9) | Uses Kaggle-attached data instead of downloading. |
