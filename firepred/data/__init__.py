"""Level 1: window archives on disk -> tensors for the model.

    archive.py     the per-fire archive format (write, find, open, unzip)
    tessera.py     per-fire TESSERA embeddings and their statistics
    transforms.py  normalize, augment, preprocess one window
    dataset.py     FireDataset: every window of a split as one Dataset
    loader.py      DataLoader settings and assembling a batch for the model
"""
