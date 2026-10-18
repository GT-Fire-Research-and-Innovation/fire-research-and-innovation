"""Level 2: Hugging Face upload/download and local data staging (needs huggingface_hub).

    repos.py            what a repo holds; its window settings
    splits.py           ROI metadata, train/val/test split, label rule, start years
    publish_windows.py  raw GeoTIFFs -> window archives in a window repo
    publish_tessera.py  per-fire TESSERA files and tessera.zip
    download.py         window repo (+ TESSERA) -> local data root
    zips.py             zip/upload/pull/unpack whole data roots
    staging.py          make a downloaded zip usable for training (stdlib only)
"""
