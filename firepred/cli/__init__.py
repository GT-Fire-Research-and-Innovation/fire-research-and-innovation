"""Level 4: command-line entry points, each a thin argparse layer over the package.

    python -m firepred.cli.train      train or test a model (was run_spatial_temp_model_pred.py)
    python -m firepred.cli.dataset    build/download the dataset (was dataset_gen_pred.py)
    python -m firepred.cli.check      quick checks: a model, the data, one training step, load time
    python -m firepred.cli.new_model  start a new model file in firepred/models/zoo/
"""
