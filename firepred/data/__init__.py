"""Level 1: the data, from raw GeoTIFFs to model-ready tensors.

Two sides share this package. Training imports only the first group (no
geospatial packages, no Hugging Face); the notebooks and `cli.dataset` call the
two masters. Modules in import order (each imports only earlier ones):

    fires.py            pure: ROI CSVs -> splits, label rules, fire ids
    archive.py          the window archive format (write, index, open, unzip)
    class_balance.py    exact background and fire pixel counts of a split (--ce-weight auto)
    tessera_archive.py  the TESSERA file format (write, read, statistics, model input)
    transforms.py       normalize, augment, preprocess one window
    fire_dataset.py     FireDataset, DataLoaders and batch assembly (training)
    files.py            local files: zip, unzip, data roots
    tessera.py          building TESSERA: per-fire grids, which fires, training views
    hub.py              Hugging Face I/O and window_config.json
    windows.py          one fire's GeoTIFFs -> input windows and next-day targets
    tessera_master.py   MASTER: every TESSERA step
    windows_master.py   MASTER: every window-data step

The dataset-building side is functional: every function returns a value,
arguments are never modified, there is no module state, only the masters
print. rasterio, pyproj, geotessera and huggingface_hub are imported inside the
functions that use them.
"""
