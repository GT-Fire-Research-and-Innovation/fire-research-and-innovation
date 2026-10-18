# Known issues (kept on purpose)

The restructure on the `explain` branch changed **where** code lives, not
**what** it computes: `tests/equivalence/` proves the new package reproduces
commit `6ebc222` bit for bit. That means the quirks below were moved
unchanged. Each one changes results or names if fixed, so each fix should be
its own labelled commit, followed by a fresh baseline run.

| # | Where | What happens | Effect if fixed |
|---|---|---|---|
| 1 | `training/run.py` (`weight_decay`) | `weight_decay = lr / 10` is logged to W&B as `weight_decay`, but Adam never receives it. Every run trained **without** weight decay. | Adding it to Adam changes every result. Removing it from the W&B config only changes the config. |
| 2 | `training/losses.py` (`dicece`) | MONAI `DiceCELoss(sigmoid=True)`: the Dice term uses **sigmoid** per channel, while the cross-entropy term uses **softmax** across the two channels of the same logits. The class weight `(1, 446.78)` weights the Dice term too, so the Dice part is around 215 and dominates the total (around 220). | Changes the training signal. |
| 3 | `firepred/models/zoo/unetr3d.py` | `-m unetr3d` has never worked: the builder passes `kernel_size_up_down`, which the local `UNETR` class does not accept (TypeError), inherited from the original TS-SatFire code. | Makes the baseline usable. |
| 4 | `training/loop.py` (`validate`) | "Validation Mean IoU / Mean Dice" are means of per-batch means and include the background class. They are not the paper's numbers; the per-sample fire F1/IoU charts are. | Changes chart values. |
| 5 | label rule | The train/val burned-area rule has three defaults: `prep/windows.build_fire_windows(label_sel=1)`, `config.TRAIN_VAL_LABEL_SEL = 0` (used by `process_data`), and the data-grab notebook's `TRAIN_VAL_LABEL_SEL = 1`. Each archive records the rule that built it (`label_sel` member), and the window repo's `window_config.json` records the train/val and test rules of the build. | Changes which windows are rebuilt. |
| 6 | `training/naming.default_test_checkpoint` | `-test` without `--checkpoint` looks in `saved_models/...epoch_199...`, but training saves into the W&B run folder. Pass `--checkpoint`. | Changes a default path. |
| 7 | `config.dataset.EXCLUDED_TEST_FIRES` | The excluded test fire `US_2021_NV3700011641620210517` is not in the shipped `roi/us_fire_2021_out_new.csv`, so the filter never removes anything. | None today. |
| 8 | W&B config vs checkpoint keys | W&B logs `tessera`/`input_channels`; checkpoints store `use_tessera`/`model_input_channels`. | Breaks dashboard filters. |
| 9 | `training/loop.py` progress bar | The tqdm `loss=` postfix divides by `(batch + 1) * size of the last batch`, so it is slightly off on a partial last batch. Logged numbers are correct. | Display only. |
| 10 | `models/blocks/window_ops.compute_mask` | Every Swin stage computes the autoregressive temporal mask on every forward pass, although only `attn_version="ar"` uses it. Costs time, not accuracy. | Faster; same results. |
| 11 | `-r`, `-it` flags | Required by the CLI but unused by the trainer (`-it` is not part of any name). | CLI change. |
| 12 | Kaggle notebook | `CE_WEIGHT` is assigned three times in the settings cell; the last one wins: `(1.0, 457.1)`, not the paper's `(1.0, 446.7836)` that the markdown describes. | Notebook setting only. |
| 13 | Checkpoint pickles | Checkpoints written under numpy 2 reference `numpy._core`; full unpickling needs numpy >= 1.26 (which ships a `numpy._core` shim). The restricted loader (`training/checkpoints.read_checkpoint`) maps both names, so it works with any numpy on torch >= 2.4. | Environment, not code. |
| 14 | `data/transforms.preprocess` | Angle channels (wind direction, aspect, forecast wind direction) become `sin(angle)` only, so 0° and 180° (north/south) map to the same value. | Adding cos changes input width and results. |

Fixed on this branch, outside the equivalence claim:
- New, opt-in loss `--loss dicefocal`: `--lambda-dice` x sigmoid Dice (as
  `--loss dice`) + `--lambda-focal` x a sigmoid focal loss on the fire channel
  (`--focal-gamma`, `--focal-alpha`), no cross entropy.
  The default loss is unchanged (`dicece`, or `dice` for attunet).
- `window_config.json` records every build setting: window length and stride,
  `train_val_label_sel` (NIFC perimeters on/off), `test_label_sel` (`"roi"` or a
  rule), crop, stored channels, archive format and the split (years, validation
  ids), plus `notes` (rule meanings, zlib level, code version). An existing file
  is never rewritten except by `process --overwrite`; a conflicting setting
  stops the build.
- The old one-member NPZ format is gone: no `convert` subcommand, no
  `pull_processed_data`, no `--processed-repo` / `--dest-repo`, and a window
  repo without `window_config.json` is no longer assumed to hold 10-day /
  3-day windows. Window archives are written at zlib level 1 as before.
- `--unzip-dir` defaults to `off`: the trainer reads the window archives
  directly unless asked to unzip them (`--unzip-dir auto`). Arrays are
  byte-identical either way; unzipping costs about 5.5x the disk.
- Training with `WANDB_MODE=disabled` (no W&B account, local debugging) no
  longer crashes at the first checkpoint: the trainer creates the W&B run folder
  if W&B did not.
- Checkpoints load without running pickled code: `read_checkpoint` tries
  `torch.load(weights_only=True)`, then (torch >= 2.4) the same with numpy's
  array-rebuild functions allowed for the TESSERA statistics. Full unpickling
  happens only for the checkpoint a training run just wrote or with
  `--trust-checkpoint` (needed for TESSERA checkpoints on torch < 2.4). Loaded
  objects are identical in every path.
- Zip extraction (`hub/zips.pull_zip_data_to`, `hub/staging.extract_zip`)
  refuses archives whose member paths escape the target folder.
- `config.dataset.window_repo_for(ts, it)` now returns
  `SamuelWu318/ts-satfire-processed-{ts}-{it}`, the repos that exist on Hugging
  Face. It used to return `...-processed-window` / `...-window-ts6-it1`.
