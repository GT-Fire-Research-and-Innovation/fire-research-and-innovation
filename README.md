## For the original github: https://github.com/zhaoyutim/TS-SatFire
# TS-SatFire Time-Series Multi-Task Satellite Imagery Dataset for Wildfire Detection and Prediction

🔥This paper has been accepted in Nature Scientific Data! Link:[[https://www.nature.com/articles/s41597-025-06271-3]]

## Repo structure

Next-day wildfire spread prediction with SwinUNETR-3D (and any model you add),
organized as a hierarchy that builds toward the model. Start with
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the model itself is explained
from scratch in [docs/swinunetr3d/](docs/swinunetr3d/).

    .
    ├── firepred/                 # the package (python -m firepred.cli.<name>)
    │   ├── config/               # L0 every project-wide constant: channels, dataset, paths, training
    │   ├── data/                 # L1 GeoTIFFs/TESSERA -> archives -> tensors; notebooks call windows_master.py, tessera_master.py
    │   ├── models/               # L2 blocks/ (Swin pieces), zoo/ (one file per model), registry
    │   ├── training/             # L3 loss, metrics, loop, evaluation, W&B, checkpoints
    │   └── cli/                  # L4 train, dataset, check, new_model
    ├── tests/                    # pytest: unit tests, golden/ (same-output values), end-to-end runs
    ├── tools/                    # trace_shapes.py: real tensor shapes through SwinUNETR-3D
    ├── docs/                     # ARCHITECTURE, KNOWN_ISSUES, swinunetr3d/ explainer
    ├── roi/                      # fire metadata CSVs (ids, dates, locations, label rule)
    └── figures/                  # dataset figures from the paper

## Abstract

We introduce a comprehensive multi-temporal remote sensing dataset covering the entire life cycle of wildfires for active fire detection, daily wildfire monitoring and next-day wildfire prediction. This multi-task dataset comprises 179 wildfire events recorded majority in the US between January 2017 and October 2021. For each wildfire, images from the beginning until the end of the wildfire are provided. It includes a total of 3552 surface reflectance images along with auxiliary data such as weather, topography, and fuel information. Labels for current active fires (AF) and burned areas (BA) are provided for each image. Manual quality assurance is performed for all AF labels and BA test labels.The dataset sets three distinct tasks: a) active fire detection, b) daily burned area mapping and c) daily wildfire progression prediction. Detection tasks, such as active fire detection and burned area mapping, require pixel-wise classification utilizing multi-spectral, multi-temporal images. Prediction tasks involve learning the underlying physical processes by integrating satellite observations with auxiliary data. The primary objective of this dataset is to stimulate further research in wildfire monitoring, particularly leveraging advanced deep learning models capable of effectively processing multi-temporal, multi-spectral images to detect fires and accurately predict fire progression. The benchmarks for detection and prediction tasks indicate that utilizing both spatial and temporal information is crucial for this dataset.

## Spectral bands used in the dataset
![Alt text](figures/flowchart.svg?raw=true "Dataset Setup")
![Alt text](figures/Bands.png?raw=true "Dataset Channels")


## Preparing the dataset
The dataset in GeoTIFF format can be downloaded from the Kaggle link below:
[[https://www.kaggle.com/datasets/z789456sx/ts-satfire](https://www.kaggle.com/datasets/z789456sx/ts-satfire)]

Prepare the environment
```
conda env create -f environment.yml
```

Build or download the prediction dataset (streams from/to Hugging Face; token from `--token` or `$HF_TOKEN`):
```
python -m firepred.cli.dataset process -ts 6 -it 1          # raw GeoTIFFs -> window archives
python -m firepred.cli.dataset download -ts 6 -it 1 --data-root ./data
```

## Train and test
```
python -m firepred.cli.check model -m swinunetr3d            # model sanity check, one step
python -m firepred.cli.check labels --data-root ./data        # background vs fire pixels of train/
python -m firepred.cli.train -m swinunetr3d -mode pred -b 8 -r 1 -lr 0.001 -nh 3 -ed 36 -ts 6 -it 1 --data-root ./data
python -m firepred.cli.train ... -test --checkpoint <file.pth>
```
Training scores the test fires with its best checkpoint at the end. The Dice + CE class
weights default to the paper's `--ce-weight 1.0 446.7836`; `--ce-weight auto` instead uses
1.0 and the background-to-fire pixel ratio of the training split, counted exactly from
the labels when training starts (`check labels` prints the same number). Add a model with
`python -m firepred.cli.new_model NAME` (see docs/ARCHITECTURE.md). This repository
covers the paper's prediction task only.

## Author

#### Yu Zhao (zhaoyutim@gmail.com), Sebastian Gerard(sgerard@kth.se), Yifang Ban (yifang@kth.se), KTH Royal Institute of Technology, Stockholm, Sweden

## Acknowledgement
#### The research is part of the project ‘Sentinel4Wildfire’ funded by Formas, the Swedish research council for sustainable development and the project ‘EO-AI4Global Change’ funded by Digital Futures.
