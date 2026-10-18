"""Level 3: training and evaluation, as a one-way chain of classes.

Modules in import order (each imports only earlier ones):

    config.py       RunConfig (frozen): every setting and every derived name
    accelerator.py  --device: CPU, CUDA GPU or TPU (PyTorch/XLA)
    objectives.py   TrainingLoss (Dice / Dice+CE / Dice+focal) and F1/IoU scores
    checkpoints.py  CheckpointStore (best N on disk), safe loading
    tracker.py      Tracker: the only module that talks to Weights & Biases
    evaluator.py    Evaluator: per-fire test scores and plots
    trainer.py      Trainer: steps, epochs and the epoch loop
    experiment.py   FireData (datasets, loaders) and Experiment.run()

    RunConfig -> Experiment.run(): FireData -> Tracker -> model
              -> Trainer.fit(data, tracker, store, evaluator) -> Evaluator.score -> Tracker.log_final
"""
