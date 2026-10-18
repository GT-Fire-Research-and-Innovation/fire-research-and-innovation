"""firepred: next-day wildfire spread prediction on the TS-SatFire dataset.

The package is a hierarchy. Each level imports only from the levels above it
in this list, so any file can be understood by reading downward:

    config/     L0  every project-wide constant (channels, paths, dataset, training)
    data/       L1  window archives on disk -> tensors for the model
    prep/       L1  raw GeoTIFFs -> window arrays (offline dataset building)
    models/     L2  building blocks, the model zoo and its registry
    hub/        L2  Hugging Face upload/download and local data staging
    training/   L3  losses, metrics, the training loop, evaluation, W&B
    cli/        L4  command-line entry points (python -m firepred.cli.<name>)

See docs/ARCHITECTURE.md for the rules and a walkthrough of one training step.
"""
