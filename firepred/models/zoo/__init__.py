"""One file per model. Every module here that does not start with "_" is
imported by firepred.models.registry, so its @register_model builder becomes
available as `-m <name>`.

    swinunetr3d.py                SwinUNETR-3D, the main model [paper]
    swinunetr3d_tpool.py          + learned weights for the per-day maps [ours]
    swinunetr3d_tessera.py        + TESSERA as multi-scale FiLM context [ours]
    swinunetr3d_tessera_tpool.py  both of the above [ours]
    unet3d.py, attunet.py, unetr3d.py   the paper's baselines
    _template.py                  the starting point for `python -m firepred.cli.new_model NAME`
"""
