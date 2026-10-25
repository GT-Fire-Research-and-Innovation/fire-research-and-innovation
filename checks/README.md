# Checks and experiments

Small scripts for inspecting datasets, validating assumptions, and investigating
experiment behavior. These run separately from model training.

Count all windows in a prepared data root, with fire counts and batches per epoch:

```bash
python checks/count_windows.py --data-root ./data --batch-size 8
```

Restrict the report to training and show how windows are distributed across fires:

```bash
python checks/count_windows.py --data-root ./data --batch-size 8 --splits train --per-fire
```

In a Kaggle notebook, run from the extracted code directory:

```python
!python checks/count_windows.py --data-root "{staged.train_root}" --batch-size 8
```

Use the same data root and batch size as the training run, including its filtered
TESSERA view when applicable. The script uses the same source discovery and window
indexing as `FireDataset`, supports both `.npz` archives and extracted fire folders,
and avoids counting both representations of the same fire. It reads metadata and
checks window member names without loading the image or label arrays.

Training visits every window once per epoch, shuffles their order, and includes a
partial final batch. Consequently, training batches per epoch are
`ceil(training_windows / batch_size)`. Validation also visits every window, without
weight updates. Test totals describe all local windows; the evaluator can further
filter test fires using the test CSV and configured exclusions.

The script requires the data-reading dependencies (`numpy` and, on Python below
3.14, `backports.zstd`). The Kaggle/Colab code bundle includes `checks/`.
