"""Level 3: training and evaluation.

    losses.py       Dice / Dice+CE / Dice+focal training loss
    metrics.py      F1 / IoU on thresholded predictions
    accelerator.py  --device: CPU, CUDA GPU or TPU (PyTorch/XLA)
    optim.py        optimizer, gradient scaler, learning-rate schedule
    naming.py       run names, checkpoint and artifact names
    tracking.py     Weights & Biases charts and run setup
    checkpoints.py  save / upload / load checkpoints, keep the best N
    loop.py         train_model: the epoch loop
    evaluation.py   evaluate_model: per-fire test scores and plots
    run.py          run(args): one full experiment, in order
"""
