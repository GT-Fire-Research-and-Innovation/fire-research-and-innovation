# Equivalence harness

Proves the restructured code computes **exactly** what the pre-restructure code
computed: same arrays bit for bit, same W&B calls, same checkpoints.

```bash
git worktree add --detach ../fire-baseline 6ebc222        # the reference commit
python tests/equivalence/compare.py --baseline ../fire-baseline --tier quick   # ~30 s
python tests/equivalence/compare.py --baseline ../fire-baseline --tier hub      # ~1 min
python tests/equivalence/compare.py --baseline ../fire-baseline --tier full --cross-test   # ~20 min
```

Needs torch 2.3.1, torchvision 0.18.1, monai 1.3.2, einops, numpy, pandas,
matplotlib, tqdm, rasterio, pyproj, huggingface_hub (no GPU, no network).

* `snapshot.py` runs in one tree and records everything; `compare.py` runs it
  in both trees (separate processes, CPU, fixed threads) and diffs the results.
* Symbols are found through `CANDIDATES`: new location first, then the
  baseline location. That map doubles as the old -> new migration table.
* `synth.py` writes deterministic synthetic archives, TESSERA files and
  GeoTIFFs (NaN, nodata, missing night files, ...). `fake_wandb.py` records
  every W&B call so chart names and run names are compared too.
* **quick**: config, archive format, FireDataset items (plain, augmented,
  unzipped folders, TESSERA), GeoTIFF -> windows, TESSERA gridding, Hugging
  Face helpers that run offline, every model's init/forward (SwinUNETR-3D with
  one Adam step), losses, metrics, LR schedules, CLI options and defaults.
* **hub**: the Hugging Face dataset pipeline against an offline fake hub
  (`fake_hub.py`; fake `geotessera` too): `process_data` on synthetic GeoTIFFs
  of four fires (a 2017 fire, a validation id, a 2021 test fire), downloads
  with and without TESSERA, `process_tessera` + `upload_tessera_zip` +
  `pull_tessera_data`, zip upload/pull. Compares every
  repo file (arrays decoded), every download/upload call and stdout. TESSERA files'
  `code_version` (the tree's git commit) is skipped by design.
* **full**: end-to-end training runs through the CLI (Dice+CE; Dice + TESSERA
  + plateau; 2 workers + cosine), `-test` with the run's own checkpoint and,
  with `--cross-test`, with the other tree's checkpoint.

## Intentional differences

Pass these to `--allow` when comparing against 6ebc222:

| key | why |
|---|---|
| `cli.train.option.model_option`, `cli.train.default.model_option` | new `--model-option KEY=VALUE` flag |
| `cli.train.option.trust_checkpoint`, `cli.train.default.trust_checkpoint` | new `--trust-checkpoint` flag (restricted checkpoint loading) |
| `cli.train.option.unzip_dir`, `cli.train.default.unzip_dir` | `--unzip-dir` defaults to `off` (was `auto`); arrays are identical either way |
| `cli.dataset.options` | old-format conversion removed: no `convert` subcommand, `--processed-repo`, `--dest-repo` alias or `processed` zip repo |
| `hub.window_config` | `window_config.json` records every build setting (label rules, crop, channels, split) plus notes, not only `ts_length`/`ts_interval` |
| `cli.train.option.loss`, `cli.train.option.{lambda_focal,focal_gamma,focal_alpha}`, `cli.train.default.{lambda_focal,focal_gamma,focal_alpha}` | new `--loss dicefocal` choice and its three flags; `dice` / `dicece` runs are unchanged |
| `cli.train.option.device`, `cli.train.default.device` | new `--device` flag; its default `auto` is the old CUDA-else-CPU choice, `tpu` is opt-in |
| `config.window_repo_for` | window repo names now match the real Hugging Face repos (`ts-satfire-processed-{ts}-{it}`) |
