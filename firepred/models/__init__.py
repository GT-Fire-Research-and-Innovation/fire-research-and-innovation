"""Level 2: the models.

    spec.py      ModelSpec, the one object every model builder receives
    registry.py  @register_model, build_model(name, spec), forward_model
    blocks/      reusable layers (Swin windows, attention, patch merging, ...)
    zoo/         one file per model; dropping a file here makes `-m <name>` work

Add a model with `python -m firepred.cli.new_model NAME` (see docs/ARCHITECTURE.md).
"""
