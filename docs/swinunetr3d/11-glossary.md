# 11. Glossary

| Term | Meaning here |
|---|---|
| **AF / BA** | Active fire (pixels the satellite flags as burning that day) / burned area (accumulated burn, here AF plus NIFC perimeters). |
| **AP** | Average precision: area under the precision-recall curve, threshold-free. WildfireSpreadTS's main metric. |
| **Attention** | Each token computes a weighted average of other tokens' *values*, weighted by how well their *keys* match its *query* (chapter 3). |
| **Augmentation** | Random flips/rotations of training windows, with angle channels rotated consistently (`data/transforms.py`). |
| **BatchNorm** | Normalizes each channel with statistics of the current batch (running averages at test time). Used in all of SwinUNETR-3D's conv blocks. |
| **Channel** | One measured quantity per pixel per day (e.g. wind speed). 27 stored, 43 model inputs (172 with TESSERA). |
| **Checkpoint** | Saved weights + optimizer state + run settings (`training/checkpoints.py`). Keys like `module.swinViT.layers1.0...`. |
| **Crop** | The fixed 256 x 256-pixel (96 km) square cut from every GeoTIFF at offset 128. |
| **DataParallel** | Splits each batch across GPUs (2 x T4 on Kaggle); adds the `module.` prefix to checkpoint keys. |
| **Dice / F1** | 2TP / (2TP + FP + FN): overlap of predicted and true fire. Soft Dice is the differentiable version used as a loss. |
| **DiceCE** | MONAI loss: Dice + weighted cross-entropy, 1:1 in the paper (chapter 7). |
| **Feature size (F, `-ed`)** | Length of each token's vector after patch embedding: 36. Doubles every stage. |
| **Forward hook** | A function PyTorch calls after a module runs; `tools/trace_shapes.py` uses them to record shapes. |
| **Head (attention, `-nh`)** | One independent query/key/value set; 3 heads of 12 numbers each at stage 1. |
| **IoU** | TP / (TP + FP + FN), the paper's Equation 2. |
| **Logit** | The raw model output before sigmoid/softmax; fire is predicted where the fire logit > 0. |
| **ModelSpec** | The one object every model builder receives: channels, classes, days, image size, heads, feature size, options (`models/spec.py`). |
| **Patch embedding** | The first layer: a (1, 2, 2)-stride convolution turning 1-day x 2 x 2-pixel patches into tokens. |
| **Patch merging** | Between Swin stages: concatenate 2 x 2 tokens, LayerNorm, Linear to 2C. Halves H, W; doubles channels. |
| **Registry** | `models/registry.py`: maps `-m NAME` to the builder in `models/zoo/NAME.py`. |
| **Relative position bias** | A learned number per head for every (Δday, Δrow, Δcol) offset inside a window, added to attention scores. Attention's only sense of position. |
| **Residual connection** | `x + f(x)`: a block adds a correction instead of replacing its input. |
| **Shifted window** | Every second Swin block moves the windows by half a window (2 tokens) so information crosses window borders; implemented with `torch.roll` + a -100 mask. |
| **Skip connection** | Passing an encoder feature map directly to the decoder level of the same size (U-Net). |
| **Swin** | "Shifted windows" Transformer (Liu et al., 2021). |
| **SwinUNETR** | Swin encoder + UNETR-style convolutional decoder (Hatamizadeh et al., 2022). |
| **TESSERA** | Annual 128-number foundation-model embedding of each 10 m pixel; here averaged to 375 m, from the year before the fire. |
| **Token** | One unit attention operates on: a 1-day x 2 x 2-pixel patch at stage 1, coarser later. |
| **U-Net** | Encoder-decoder with skip connections (Ronneberger et al., 2015). |
| **U-TAE** | U-Net with temporal attention encoder (Sainte Fare Garnot & Landrieu, 2021); best model on WildfireSpreadTS. |
| **VIIRS** | The satellite instrument (375 m I-bands) providing the fire, reflectance and thermal channels. |
| **Window (data)** | One training example: 6 input days + the next day's target. Built every day (`-it 1`). |
| **Window (attention)** | A (6, 4, 4) block of tokens inside which attention runs. |
| **Zoo** | `firepred/models/zoo/`: one file per model. |
