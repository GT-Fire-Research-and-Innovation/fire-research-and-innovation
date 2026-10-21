"""Level 1: the data, from raw GeoTIFFs to model-ready tensors.

Two sides share this package. Training imports only the first group (no
geospatial packages, no Hugging Face); the notebooks and `cli.dataset` call the
two masters. Modules in import order (each imports only earlier ones):

    archive.py          every file format: Zstandard .npz, window archives (write, index,
                        open, unzip), TESSERA files (write, read, statistics, model input)
    stats.py            pixel counts and input normalization of a split, and the data
                        root's window_config.json that carries them to training
    fire_dataset.py     transforms (normalize, augment, preprocess), FireDataset,
                        DataLoaders and batch assembly (training)
    files.py            local files: ROI CSVs -> splits and label rules, zips, data roots
    tessera.py          building TESSERA: per-fire grids, which fires, training views
    hub.py              Hugging Face I/O and the window repo's window_config.json
    windows.py          one fire's GeoTIFFs -> input windows and next-day targets
    tessera_master.py   MASTER: every TESSERA step
    windows_master.py   MASTER: every window-data step

The dataset-building side is functional: every function returns a value,
arguments are never modified, there is no module state, only the masters
print. rasterio, pyproj, geotessera and huggingface_hub are imported inside the
functions that use them.
"""
