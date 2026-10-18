"""Level 1: the dataset pipeline (raw GeoTIFFs and TESSERA -> window data on Hugging Face).

Notebooks and cli/dataset.py call only the two masters; the other modules are
their building blocks, listed in import order (each imports only earlier ones):

    roi.py             pure: ROI CSVs -> splits, label rules, TESSERA years
    windows.py         one fire's GeoTIFFs -> input windows and next-day targets
    tessera_grid.py    prior-year TESSERA embeddings averaged onto a fire's grid
    staging.py         local files: zip, unzip, data roots, TESSERA views
    repos.py           Hugging Face I/O and window_config.json
    tessera_master.py  MASTER: every TESSERA step
    dataset_master.py  MASTER: every window-data step

Functional style: every function returns a value, arguments are never
modified, there is no module state, only the masters print. rasterio, pyproj,
geotessera and huggingface_hub are imported inside the functions that use
them, so the masters import on a training machine without them.
"""
