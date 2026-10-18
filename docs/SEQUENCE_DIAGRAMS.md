# Sequence diagrams

Who calls whom, in what order, for the whole program. These diagrams show the
same route as [EXECUTION_PATH.md](EXECUTION_PATH.md) (the trunk: Kaggle
notebook, GPU, `swinunetr3d`, Dice+CE, no TESSERA), drawn as calls between files.

How to read them:

- Each column is a file (or an outside service). Arrows are calls in time
  order, top to bottom; dashed arrows are return values. Numbers on the
  arrows count the steps.
- `file.py:123` in a label is the line that makes the call.
- `loop` boxes repeat. `alt` boxes take one of their routes, chosen by an input.
  `opt` boxes run only when their condition holds. `break` boxes stop the run.
- Diagram 1 is the whole program on one page. Diagrams 2–8 zoom into one stage
  each, in execution order. Diagram 9 covers the side entry points (`check`,
  `new_model`, TESSERA build).

The diagrams are Mermaid; GitHub and the VS Code Markdown preview render them.

1. [The whole program](#1-the-whole-program)
2. [Stage 0: building the dataset](#2-stage-0-building-the-dataset)
3. [Stage 1: the Kaggle notebook](#3-stage-1-the-kaggle-notebook)
4. [Stages 2–3: command line and run setup](#4-stages-23-command-line-and-run-setup)
5. [Stage 4: data objects](#5-stage-4-data-objects)
6. [Stage 5: the epoch loop](#6-stage-5-the-epoch-loop)
7. [Inside the model: one SwinUNETR-3D forward pass](#7-inside-the-model-one-swinunetr-3d-forward-pass)
8. [Stage 6: testing the best checkpoint](#8-stage-6-testing-the-best-checkpoint)
9. [Side entry points](#9-side-entry-points)

---

## 1. The whole program

```mermaid
sequenceDiagram
    autonumber
    actor You
    participant DSC as cli/dataset.py
    participant HUB as hub/
    participant PREP as prep/windows.py
    participant HF as Hugging Face
    participant NB as Kaggle notebook
    participant RUN as training/run.py
    participant MOD as models/
    participant DATA as data/
    participant TL as training/loop.py
    participant EVAL as training/evaluation.py
    participant WB as W&B

    rect rgba(127, 127, 127, 0.08)
    Note over You,HF: Stage 0, once per window shape (offline)
    You->>DSC: python -m firepred.cli.dataset process -ts 6 -it 1
    DSC->>HUB: process_data
    HUB->>HF: download raw GeoTIFFs, one fire at a time
    HUB->>PREP: build_fire_windows
    PREP-->>HUB: windows and next-day targets
    HUB->>HF: upload pID.npz archives, then data.zip
    end

    rect rgba(127, 127, 127, 0.08)
    Note over You,WB: Stages 1 to 6, every experiment
    You->>NB: run all cells
    NB->>HF: download data.zip
    NB->>RUN: python -m firepred.cli.train (subprocess)
    RUN->>WB: start run
    RUN->>MOD: build_model("swinunetr3d")
    RUN->>DATA: FireDataset train / val, DataLoaders
    RUN->>TL: train_model
    loop 100 epochs
        TL->>DATA: next batch of windows
        TL->>MOD: forward, loss, backward, Adam step
        TL->>WB: log train and validation scores
        TL->>TL: keep best checkpoint
    end
    TL-->>RUN: best checkpoint path
    RUN->>EVAL: score the 2021 test fires
    EVAL-->>RUN: F1, IoU (per fire, then averaged)
    RUN->>WB: log test scores
    end
```

---

## 2. Stage 0: building the dataset

`python -m firepred.cli.dataset process -ts 6 -it 1`, then `download` and
`upload-zip`. Step-by-step text: [EXECUTION_PATH.md, Stage 0](EXECUTION_PATH.md#stage-0-building-the-dataset-once-offline).

```mermaid
sequenceDiagram
    autonumber
    actor You
    participant CLI as cli/dataset.py
    participant PW as hub/publish_windows.py
    participant REPOS as hub/repos.py
    participant SPL as hub/splits.py
    participant WIN as prep/windows.py
    participant GEO as prep/geotiff.py
    participant ARC as data/archive.py
    participant ZIP as hub/download.py + zips.py
    participant HF as Hugging Face

    You->>CLI: process -ts 6 -it 1
    CLI->>CLI: main() cli/dataset.py:180<br/>window_repo_for(6, 1) :185
    CLI->>PW: process_data() cli/dataset.py:193
    PW->>HF: DatasetStreamer: login, create repo publish_windows.py:224
    PW->>REPOS: ensure_window_config(settings) :233
    REPOS->>HF: read window_config.json
    alt repo has no config yet
        REPOS->>HF: upload settings and notes
    else config differs and no overwrite flag
        REPOS-->>You: ValueError, nothing is built
    else overwrite flag given
        REPOS->>HF: replace config with this run's settings
    end
    PW->>SPL: obtain_ids() :242
    SPL->>HF: download ROI CSVs
    SPL-->>PW: train, val (VAL_IDS), test (2021) fire IDs
    PW->>PW: label rule per fire :247<br/>train/val 0 = AF + NIFC, test from ROI CSV

    loop each split: train, val, test (run_split :260)
        PW->>HF: list files already in the repo
        loop each fire of the split
            opt fire already uploaded and no overwrite flag
                PW->>PW: skip it :95
            end
            PW->>HF: download its GeoTIFFs to temp :109
            PW->>WIN: build_fire_windows(label_sel) :123
            loop each window k (starts on day k)
                loop 7 days: 6 inputs, then the target day
                    WIN->>GEO: read_tiff day, night, FirePred :116 :129 :170
                    GEO-->>WIN: band arrays
                    WIN->>WIN: stack 27 channels, crop 256x256 :181<br/>running max of heat bands :214
                    WIN->>WIN: add today's fire to burned-so-far masks :228 :265
                end
                WIN->>WIN: target = burned now and not before :288
                WIN->>WIN: rewind state to next window start :306
            end
            WIN-->>PW: inputs (N,27,6,256,256), targets (N,256,256)
            PW->>ARC: write_fire_archive() :139
            ARC->>ARC: one compressed member per window archive.py:64 :79
            ARC-->>PW: pID.npz in staging
            opt 10 fires staged
                PW->>HF: upload_folder :160
            end
        end
        PW->>HF: upload the remaining fires :179
    end

    You->>CLI: download, then upload-zip
    CLI->>ZIP: load_and_organize_data download.py:36
    ZIP->>HF: fetch every archive into data/train, val, test
    CLI->>ZIP: upload_data_zip zips.py:51
    ZIP->>HF: data.zip (stored, no second compression)
```

---

## 3. Stage 1: the Kaggle notebook

`notebooks/fire-research-swinunetr.ipynb` (local only). Cells hold settings;
all logic is in the cloned package.

```mermaid
sequenceDiagram
    autonumber
    actor You
    participant NB as notebook cells
    participant PIP as pip
    participant GH as GitHub
    participant WB as W&B
    participant HF as Hugging Face
    participant STG as hub/staging.py
    participant CHK as cli/check.py
    participant TR as trainer subprocess

    You->>NB: Run all
    NB->>NB: cell 1: settings and consistency asserts
    alt ACCELERATOR = 'gpu' (trunk)
        NB->>NB: cell 3: require CUDA, print GPUs
    else ACCELERATOR = 'tpu'
        NB->>NB: cell 3: find torch_xla without opening the TPU<br/>set PJRT_DEVICE=TPU
    end
    NB->>PIP: cell 3: install MONAI 1.3.2 etc, torch pinned
    NB->>GH: cell 5: git clone REPO_BRANCH
    NB->>TR: cell 5: firepred.cli.train --help
    TR-->>NB: every flag the notebook passes is present
    NB->>WB: cell 7: login, permission-check run
    NB->>WB: delete the permission-check run
    alt attached Kaggle data
        NB->>STG: cell 9: find_data_root(ATTACHED_DATA_ROOT)
    else trunk: download
        NB->>HF: cell 9: hf_hub_download data.zip
        NB->>STG: extract_data staging.py:75
        STG->>STG: extract_zip :47 (disk check, path check, once per session)
        STG->>STG: find_data_root (the folder with train/ val/ test/)
    end
    STG-->>NB: DATA_ROOT
    opt USE_TESSERA = True
        NB->>HF: tessera.zip
        NB->>STG: build_tessera_view staging.py:80
        STG-->>NB: symlink view without 2017 fires, plus tessera/
    end
    NB->>CHK: cell 13: check timing check.py:184
    CHK-->>NB: seconds to load one window
    NB->>NB: cell 15: build the command, save it with the code version
    NB->>TR: cell 15: python -m firepred.cli.train ... --device cuda
    Note over TR: Diagrams 4 to 8 run inside this subprocess
    TR-->>NB: exit 0
    NB->>NB: list saved checkpoints
```

---

## 4. Stages 2–3: command line and run setup

```mermaid
sequenceDiagram
    autonumber
    participant NB as notebook cell 15
    participant CLI as cli/train.py
    participant RUN as training/run.py
    participant ARC as data/archive.py
    participant NAM as training/naming.py
    participant TRK as training/tracking.py
    participant WB as W&B
    participant ACC as training/accelerator.py
    participant REG as models/registry.py
    participant ZOO as models/zoo/swinunetr3d.py
    participant LOS as training/losses.py
    participant OPM as training/optim.py

    NB->>CLI: python -m firepred.cli.train -m swinunetr3d ...
    CLI->>CLI: parse_args() train.py:39 (defaults from config/training.py)
    CLI->>RUN: main() train.py:232, run(args)
    RUN->>RUN: seed numpy, torch, CUDA with 42 run.py:65
    opt no loss flag
        RUN->>RUN: default_loss :45 (dice for attunet, else dicece)
    end
    RUN->>RUN: 43 model channels :90, check -nc :92
    RUN->>RUN: data root train/ val/ test/ :103
    opt unzip-dir auto or a path
        RUN->>ARC: unzip_data_root archive.py:227
        ARC-->>RUN: root of unpacked pID/ folders (same arrays)
    end
    RUN->>NAM: loss_config, lr_config, run_tag :134
    NAM-->>RUN: configs, tag "_dicece"
    RUN->>TRK: configure_wandb :147
    TRK->>WB: init, define epoch charts, run name, config
    RUN->>ACC: resolve_device("cuda") :165
    alt trunk: cuda or auto on a GPU machine
        ACC-->>RUN: torch.device("cuda")
    else tpu
        ACC->>ACC: import torch_xla
        ACC-->>RUN: XLA device
        RUN->>ACC: seed_device (TPU random generator)
        RUN->>WB: config device = tpu
    end
    RUN->>REG: build_model("swinunetr3d", ModelSpec) :177
    REG->>REG: _import_zoo registry.py:43 imports every zoo file
    Note over REG,ZOO: each @register_model fills the name -> builder table
    REG->>ZOO: build(spec) swinunetr3d.py:290
    ZOO-->>REG: SwinUNETR, patch (1,2,2), window (6,4,4), v1 attention
    REG-->>RUN: model
    RUN->>WB: model and schedule settings
    RUN->>RUN: nn.DataParallel(model).to(device) :197
    RUN->>LOS: build_criterion("dicece") :203
    LOS-->>RUN: DiceCELoss, weights (1.0, 457.1) losses.py:69
    RUN->>OPM: build_optimizer :214, build_grad_scaler :215
    OPM-->>RUN: Adam lr 0.001, scaler on for CUDA
    RUN->>NAM: checkpoint_template :247
```

---

## 5. Stage 4: data objects

```mermaid
sequenceDiagram
    autonumber
    participant RUN as training/run.py
    participant TES as data/tessera.py
    participant DS as data/dataset.py
    participant ARC as data/archive.py
    participant LDR as data/loader.py

    alt trunk: training
        Note over RUN: run.py:253 not -test
    else -test
        Note over RUN: skip to diagram 8
    end
    opt tessera
        RUN->>TES: compute_tessera_stats(training fires only) run.py:256
        TES-->>RUN: per-band mean, std
    end
    RUN->>DS: FireDataset(train, augment=True) run.py:266
    DS->>ARC: find_fire_sources dataset.py:64
    ARC-->>DS: pID.npz paths
    opt tessera
        DS->>DS: load every fire's embedding once :80
    end
    loop each fire archive
        DS->>ARC: open_fire (zip directory only, no decompression) :114
        ARC-->>DS: num_windows
        DS->>DS: index += (fire, window number) per window
    end
    DS->>DS: normalizer and land-cover one-hot table :166
    DS-->>RUN: train_dataset (len = all windows)
    RUN->>DS: FireDataset(val, augment=False)
    DS-->>RUN: val_dataset
    RUN->>LDR: validate_dataset_output(val) run.py:287
    LDR->>DS: val_dataset[0]
    DS-->>LDR: data (43,6,256,256)
    LDR-->>RUN: channels match the model, or raise
    RUN->>LDR: make_dataloader train (shuffle), val (fixed order) :294
    LDR-->>RUN: DataLoaders, 2 workers, prefetch 1 loader.py:25
    opt test-every 10 (trunk)
        RUN->>RUN: prepare score_test_fires :312
    end
```

---

## 6. Stage 5: the epoch loop

`DataLoader worker` stands for the 2 loader processes that call
`FireDataset.__getitem__`. Diagram 7 opens up the `forward` arrow.

```mermaid
sequenceDiagram
    autonumber
    participant RUN as training/run.py
    participant TL as training/loop.py
    participant W as DataLoader worker
    participant DS as data/dataset.py
    participant TF as data/transforms.py
    participant LDR as data/loader.py
    participant REG as models/registry.py
    participant MDL as SwinUNETR (on GPU)
    participant LOS as training/losses.py
    participant MET as training/metrics.py
    participant ACC as training/accelerator.py
    participant CK as training/checkpoints.py
    participant WB as W&B

    RUN->>TL: train_model run.py:322
    loop epoch 1 to 100 (loop.py:255)
        TL->>TL: train_one_epoch :257, model.train()
        loop each training batch of 8
            TL->>W: next batch
            loop 8 windows
                W->>DS: __getitem__(i) dataset.py:178
                DS->>DS: load_data: decompress one window :183 :208
                DS->>TF: Normalize :186
                DS->>TF: augment: flips, 90 degree turns :194
                DS->>TF: preprocess: sin(angles), land-cover one-hot :198
                DS-->>W: data (43,6,256,256), labels (2,256,256)
            end
            W-->>TL: batch (8,43,6,256,256)
            TL->>LDR: batch_inputs: move to GPU loop.py:49
            opt tessera
                LDR->>LDR: repeat static map over 6 days, append (172 ch)
            end
            TL->>REG: forward_model :57
            REG->>MDL: model(batch), split across GPUs
            MDL-->>REG: logits (8,2,6,256,256)
            REG-->>TL: mean over 6 days (8,2,256,256)
            TL->>LOS: compute_loss :58
            LOS-->>TL: total, dice part, CE part
            TL->>ACC: mark_step :59 (no-op on GPU)
            TL->>MET: threshold, fire F1, background F1, IoU :65
            TL->>TL: backward, Adam step, scaler update :72
            TL->>ACC: mark_step (no-op on GPU)
            break running loss not finite
                TL-->>RUN: RuntimeError, stop training :90
            end
        end
        TL->>WB: log training epoch :261
        TL->>TL: validate :277, model.eval(), no gradients
        loop each validation batch (no augmentation)
            TL->>REG: forward_model, compute_loss
            TL->>MET: F1, IoU, MONAI mean IoU and Dice :159
        end
        TL->>WB: log validation epoch and learning rate :281
        TL->>CK: offer(val loss) :311
        alt lowest val loss so far
            CK->>CK: delete previous best, save_checkpoint :26
            CK->>ACC: save (torch.save, or xm.save on TPU) :37
            CK->>WB: upload model artifact and wait :51
        else not better
            CK->>CK: keep the old one
        end
        opt lr-schedule plateau or cosine
            TL->>TL: step_lr_scheduler :326
        end
        opt epoch is a multiple of 10 (test-every)
            TL->>TL: save every random state :203
            TL->>RUN: score_test_fires (diagram 8, no plots)
            RUN-->>TL: test F1, IoU
            TL->>TL: restore random states
            TL->>WB: log periodic test scores :329
        end
    end
    TL-->>RUN: best checkpoint path :345
```

---

## 7. Inside the model: one SwinUNETR-3D forward pass

Shapes per GPU (B = 4 on 2 GPUs), F = 36. Step-by-step text:
[EXECUTION_PATH.md, stop 5.1c](EXECUTION_PATH.md#51-train-one-epoch) and
[docs/swinunetr3d/](swinunetr3d/).

```mermaid
sequenceDiagram
    autonumber
    participant M as zoo/swinunetr3d.py
    participant E as blocks/swin_encoder.py
    participant L as BasicLayer (swin_block.py)
    participant K as SwinTransformerBlock (swin_block.py)
    participant O as blocks/window_ops.py
    participant A as blocks/window_attention.py
    participant P as blocks/patch_merging.py
    participant U as MONAI Unetr blocks

    M->>E: swinViT(x) swinunetr3d.py:271, x (B,43,6,256,256)
    E->>E: patch_embed: 1x2x2 patch to 36 numbers swin_encoder.py:159
    Note over E: x0 (B,36,6,128,128)
    loop stage 1 to 4 (layers1 to layers4) :162 to :168
        E->>L: forward swin_block.py:287
        L->>O: compute_mask (shift 0,2,2, days never shift) :299
        O-->>L: mask, -100 between wrapped-around tokens
        loop block 1 (plain), block 2 (shifted) :300
            L->>K: forward :193
            K->>K: LayerNorm, pad :129
            opt second block
                K->>K: roll by half a window :156
            end
            K->>O: window_partition :165, windows of 6x4x4 = 96 tokens
            K->>A: attention in every window :169
            A->>A: qkv, scores + relative position bias + mask, softmax
            A-->>K: weighted sum of values
            K->>O: window_reverse :171, roll back
            K->>K: residual add :201, then MLP + residual :205
        end
        L->>P: patch merging :304, halve H and W, double channels
        P-->>E: next stage input
        E->>E: proj_out: layer norm of this scale
    end
    E-->>M: 5 scales: (36,128²) (72,64²) (144,32²) (288,16²) (576,8²), all 6 days
    M->>U: encoder1 on raw input, encoder2 to 4 on scales 0 to 2 :273 to :276
    M->>U: encoder10 bottleneck on the 8x8 scale :278
    loop decoder5 to decoder1 :279 to :284
        M->>U: upsample x2, concat same-size encoder features, refine
    end
    Note over M: 8² to 16² to 32² to 64² to 128² to 256²
    M->>M: out: 1x1x1 conv to 2 logits per pixel per day :286
    Note over M: returns (B,2,6,256,256), registry.forward_model averages over days
```

---

## 8. Stage 6: testing the best checkpoint

```mermaid
sequenceDiagram
    autonumber
    participant RUN as training/run.py
    participant CK as training/checkpoints.py
    participant ACC as training/accelerator.py
    participant DS as data/dataset.py
    participant EV as training/evaluation.py
    participant LDR as data/loader.py
    participant REG as models/registry.py
    participant MET as training/metrics.py
    participant TRK as training/tracking.py
    participant WB as W&B

    alt trunk: just trained
        RUN->>RUN: test-after-train on, path = best checkpoint run.py:364
    else -test
        RUN->>RUN: path = --checkpoint or saved_models/ default :373
    end
    RUN->>ACC: checkpoint_map_location (CPU on TPU, else the device)
    RUN->>CK: load_checkpoint :380
    CK->>CK: read_checkpoint :160<br/>weights_only, then numpy allowlist, then trusted full load
    CK->>CK: refuse TESSERA or time-downsampling mismatch :201
    CK-->>RUN: checkpoint dict
    RUN->>DS: FireDataset(test, no augmentation) :389
    RUN->>RUN: shape check :390, load weights and optimizer state :398
    RUN->>EV: evaluate_model :405
    EV->>EV: group test windows by fire :123
    EV->>EV: fire order from the 2021 ROI CSV :130
    loop each test fire :145
        EV->>LDR: loader over this fire's windows
        loop each batch
            EV->>REG: forward_model, no gradients :166
            REG-->>EV: logits (B,2,256,256)
            EV->>MET: threshold, F1, background F1, IoU per window
            opt save plots (final test)
                EV->>EV: overlay PNG: hits, false alarms, misses :190
            end
        end
        EV->>EV: fire score = mean over its windows :202
    end
    EV->>EV: model score = mean over fires :218
    EV-->>RUN: f1, iou, background_f1, fires
    RUN->>TRK: log_final_test tracking.py:101
    TRK->>WB: log and run summary
```

---

## 9. Side entry points

The pieces that run outside the trunk. Each one starts its own process.

```mermaid
sequenceDiagram
    autonumber
    actor You
    participant CHK as cli/check.py
    participant NEW as cli/new_model.py
    participant REG as models/registry.py
    participant DS as data/dataset.py
    participant DSC as cli/dataset.py
    participant PT as hub/publish_tessera.py
    participant TG as prep/tessera_grid.py
    participant HF as Hugging Face

    rect rgba(127, 127, 127, 0.08)
    Note over You,DS: quick checks before a long run, nothing saved or logged
    You->>CHK: check model -m swinunetr3d
    CHK->>REG: make_model check.py:88, build_model
    CHK->>CHK: random input, training_step :68, check output (B,2,H,W)
    You->>CHK: check data --data-root ... [--train-step]
    CHK->>DS: FireDataset train and val :143
    CHK->>CHK: check batch shapes, optionally one real step :148
    You->>CHK: check timing
    CHK->>DS: load 10 augmented windows :184
    end

    rect rgba(127, 127, 127, 0.08)
    Note over You,REG: adding a model
    You->>NEW: new_model mymodel
    NEW->>NEW: render the zoo template :15, write zoo/mymodel.py :26
    Note over REG: next build_model call imports it, so -m mymodel works
    end

    rect rgba(127, 127, 127, 0.08)
    Note over You,HF: TESSERA embeddings (for USE_TESSERA = True)
    You->>DSC: tessera
    DSC->>PT: process_tessera publish_tessera.py:22
    PT->>HF: fire split, start years, GeoTIFF list, existing files
    loop each fire with a prior-year embedding
        PT->>HF: first and last VIIRS_Day GeoTIFF
        PT->>TG: build_fire_tessera :128, tessera_grid.py:211
        TG->>TG: crop_grid :237, grid_tessera (year before the fire) :239
        TG->>TG: save_tessera_npz :262
        opt batch staged
            PT->>HF: upload_staged :83
        end
    end
    You->>DSC: tessera-zip
    DSC->>PT: upload_tessera_zip :160
    PT->>HF: tessera.zip with excluded_npz.txt (2017 fires)
    end
```
