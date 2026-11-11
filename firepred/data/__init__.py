"""Level 1: the data, from raw GeoTIFFs to model-ready tensors.

Two sides share this package. Training imports only the first group (no
geospatial packages, no Hugging Face); the notebooks and `cli.dataset` call the
three masters. Modules in import order (each imports only earlier ones):

    fire_records.py     records shared by fire tracking and zone files: settings, fire
                        events, zone files (data only, numpy only)
    archive.py          every file format: Zstandard .npz, window archives (write, index,
                        open, unzip), TESSERA files (write, read, statistics, model input),
                        zone files (write, read)
    stats.py            pixel counts and input normalization of a split, and the data
                        root's window_config.json that carries them to training
    fire_dataset.py     transforms (normalize, augment, preprocess), FireDataset,
                        DataLoaders and batch assembly (training)
    files.py            local files: ROI CSVs -> splits and label rules, zips, data roots
    tessera.py          building TESSERA: per-fire grids, which fires, training views
    hub.py              Hugging Face I/O and the window repo's window_config.json
    windows.py          one fire's GeoTIFFs -> input windows and next-day targets
    fire_tracks.py      one archive's targets -> separate fires day by day (ignitions,
                        merges, splits), every threshold from one separation distance
    fire_zones.py       fire tracks -> per-window zones and target owners, one training
                        sample per fire
    tessera_master.py   MASTER: every TESSERA step
    windows_master.py   MASTER: every window-data step
    zones_master.py     MASTER: a zone file beside every window archive

The dataset-building side is functional: every function returns a value,
arguments are never modified, there is no module state, only the masters
print. rasterio, pyproj, geotessera and huggingface_hub are imported inside the
functions that use them.
"""
