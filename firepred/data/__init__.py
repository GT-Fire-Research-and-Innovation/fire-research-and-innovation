"""Level 1: window archives on disk -> tensors for the model.

    archive.py     the per-fire archive format (write, find, open, unzip)
    transforms.py  normalize, augment, preprocess one window
    dataset.py     TESSERA maps and statistics, FireDataset (every window of a
                   split as one Dataset), DataLoaders and batch assembly
"""
