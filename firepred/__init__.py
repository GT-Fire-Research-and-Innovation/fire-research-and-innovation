"""firepred: next-day wildfire spread prediction on the TS-SatFire dataset.

The package is a hierarchy. Each level imports only from the levels above it
in this list, so any file can be understood by reading downward:

    config/     L0  every project-wide constant (data.py, training.py)
    data/       L1  the data end to end: GeoTIFFs and TESSERA -> window data on
                    Hugging Face -> tensors for the model; notebooks call
                    data/windows_master.py and data/tessera_master.py
    models/     L2  building blocks, the model zoo and its registry
    training/   L3  losses, metrics, the training loop, evaluation, W&B
    cli/        L4  command-line entry points (python -m firepred.cli.<name>)

See docs/ARCHITECTURE.md for the rules and a walkthrough of one training step.
"""
