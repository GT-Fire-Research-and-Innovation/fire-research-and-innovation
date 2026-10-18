# Migration: where everything went

The `explain` branch replaced the old scripts and folders with the `firepred`
package. It is a clean break: old import paths and scripts no longer exist.
The command-line **flags** are unchanged; only the entry points moved.

## Commands

| Before | After |
|---|---|
| `python run_spatial_temp_model_pred.py <flags>` | `python -m firepred.cli.train <same flags>` (adds `--model-option KEY=VALUE`, `--trust-checkpoint` and `--device auto|cpu|cuda|tpu`; `--unzip-dir` now defaults to `off`) |
| `python dataset_gen_pred.py <cmd> <flags>` | `python -m firepred.cli.dataset <cmd> <same flags>`, except `convert`, `--processed-repo`, `--dest-repo` and `--zip-repo processed`, which are removed |
| notebook smoke-test string | `python -m firepred.cli.check data --data-root R --train-step ...` |
| notebook load-timing snippet | `python -m firepred.cli.check timing --data-root R` |
| (none) | `python -m firepred.cli.check model -m NAME` |
| (none) | `python -m firepred.cli.new_model NAME` |
| `python run_spatial_temp_model.py`, `run_spatial_model.py`, `run_seq_model.py`, `dataset_gen_afba.py`, `validate_tessera_augmented_dataset.py` | `PYTHONPATH=.:legacy python legacy/<same name>.py` (unused paper code) |

## Symbols

| Before | After |
|---|---|
| `satimg_dataset_processor.config.*` | `firepred.config.dataset.*` (+ new `CROP_OFFSET`, `CROP_SIZE`, `SPLITS`, ...) |
| channel counts, degree/land-cover indices, normalization stats (scattered) | `firepred.config.channels` |
| `SEED`, `PAPER_CE_WEIGHT`, `LR_SCHEDULES`, W&B project/entity (trainer) | `firepred.config.training` |
| `satimg_dataset_processor.data_generator_pred_torch.FireDataset` | `firepred.data.dataset.FireDataset` (the unused `label_sel` argument is gone) |
| `...data_generator_pred_torch.Normalize`, `FireDataset.augment/preprocess` | `firepred.data.transforms.Normalize / augment / preprocess` |
| `...data_generator_pred_torch.fire_name, find_fire_sources, open_fire, unzip_data_root` | `firepred.data.archive.*` |
| `...data_generator_pred_torch.read_tessera, compute_tessera_stats, TESSERA_CHANNELS` | `firepred.data.tessera.*`, `firepred.config.channels.TESSERA_CHANNELS` |
| `PredDatasetProcessor._save_windowed_npz` | `firepred.data.archive.write_fire_archive` |
| `PredDatasetProcessor.process_location`, `._binary_label` | `firepred.prep.windows.build_fire_windows`, `binary_mask` |
| `SatProcessingUtils.read_tiff` | `firepred.prep.geotiff.read_tiff` |
| `satimg_dataset_processor.tessera` (grid part) | `firepred.prep.tessera_grid` |
| `satimg_dataset_processor.tessera.process_tessera, upload_tessera_zip` | `firepred.hub.publish_tessera` |
| `satimg_dataset_processor.tessera.pull_tessera_data` | `firepred.hub.zips.pull_tessera_data` |
| `satimg_dataset_processor.tessera.fire_start_years, tessera_excluded_npz` | `firepred.hub.splits` |
| `satimg_dataset_processor.hf_dataset.list_fire_files, read/ensure_window_config` | `firepred.hub.repos` |
| `...hf_dataset.download_roi_csvs, fire_label_sel, obtain_ids` | `firepred.hub.splits` |
| `...hf_dataset.DatasetStreamer, process_data` | `firepred.hub.publish_windows` (`DatasetStreamer` no longer subclasses `PredDatasetProcessor`) |
| `...hf_dataset.DatasetLoader, download_processed_data` | `firepred.hub.download` |
| `...hf_dataset.zip_data_from, zip_data_to, upload_data_zip, pull_zip_data_to` | `firepred.hub.zips` |
| `...hf_dataset.convert_old_npz, upload_batch, convert_and_upload`, `pull_processed_data`, `PROCESSED_REPO`, `LEGACY_WINDOW_SETTINGS` | removed: the old one-member NPZ format is no longer supported |
| `...hf_dataset.DEST_REPO` | removed; use `firepred.config.dataset.WINDOW_REPO` |
| Kaggle notebook `find_data_root, extract_zip, build_tessera_view` | `firepred.hub.staging` (+ `extract_data`) |
| `spatial_models.swinunetr.swinunetr.SwinUNETR` | `firepred.models.zoo.swinunetr3d.SwinUNETR` |
| `spatial_models.swinunetr.swin.SwinTransformerBlock, BasicLayer` | `firepred.models.blocks.swin_block` |
| `spatial_models.swinunetr.swin.SwinTransformer` | `firepred.models.blocks.swin_encoder` |
| `spatial_models.swinunetr.WindowAttentionV1/V2, AutoregressiveAttention, PatchMerging, utils` | `firepred.models.blocks.window_attention / window_attention_v2 / autoregressive_attention / patch_merging / window_ops` |
| `spatial_models.unet.UNet`, `.attentionunet.AttentionUnet`, `.unetr.unetr.UNETR` | `firepred.models.zoo.unet3d / attunet / unetr3d` |
| `run_spatial_temp_model_pred.build_model(name, channels, ...)` | `firepred.models.registry.build_model(name, ModelSpec(...))` |
| `run_spatial_temp_model_pred.forward_model(model, name, x, device)` | `firepred.models.registry.forward_model(model, x)` |
| `...build_criterion, compute_loss` | `firepred.training.losses` |
| `...fire_only_f1, background_f1, fire_only_iou` (+ MeanIoU/DiceMetric/post-transform setup) | `firepred.training.metrics` |
| `...build_lr_scheduler` (+ Adam, GradScaler) | `firepred.training.optim` |
| `...build_wandb_metric_labels, configure_wandb, log_final_test` | `firepred.training.tracking` |
| run/checkpoint/artifact name expressions in `main()` | `firepred.training.naming` |
| `...save_checkpoint, upload_checkpoint`, checkpoint loading checks | `firepred.training.checkpoints` |
| `...train_model` | `firepred.training.loop` |
| `...evaluate_model, load_test_ids, normalize_for_plot` | `firepred.training.evaluation` |
| `...main()` | `firepred.training.run.run(args)`; `...parse_args` -> `firepred.cli.train.parse_args` |
| `...make_dataloader, validate_dataset_output, batch_inputs` | `firepred.data.loader` |

`tests/equivalence/snapshot.py` (`CANDIDATES`) is the same table in code form.

## Notebooks

- Training (Kaggle): clone the branch, then
  `python -m firepred.cli.train ...` instead of `run_spatial_temp_model_pred.py`;
  `from firepred.hub.staging import extract_data, extract_zip, build_tessera_view`
  instead of the inline helpers; `firepred.config.dataset.window_repo_for` for
  the data repo; `python -m firepred.cli.check` for the smoke test and timing;
  `MODEL_NAME` / `MODEL_OPTIONS` select any registered model.
- Data grab (Colab): `from firepred.hub.publish_windows import process_data`,
  `from firepred.hub.publish_tessera import process_tessera, upload_tessera_zip`,
  `from firepred.hub.zips import ...`, `from firepred.hub.download import ...`.

Old checkpoints load unchanged: their state-dict keys and stored settings are
identical (verified by `tests/equivalence`, `-test` across both trees).
Loading is now restricted (`torch.load(weights_only=True)`, plus numpy arrays on
torch >= 2.4). A `-test` run of a TESSERA checkpoint on torch < 2.4 needs
`--trust-checkpoint`; the checkpoint a training run just wrote is always trusted.
