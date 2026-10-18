"""Level 2: the models.

    registry.py  ModelSpec (what every builder receives), @register_model,
                 build_model(name, spec), forward_model
    blocks/      reusable layers (Swin windows, attention, patch merging, ...)
    zoo/         one file per model; dropping a file here makes `-m <name>` work

Add a model with `python -m firepred.cli.new_model NAME` (see docs/ARCHITECTURE.md).
"""
