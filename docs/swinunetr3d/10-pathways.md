# 10. Pathways: what to try next, and the evidence for each

Every claim here is sourced in [research_notes.md](research_notes.md) (numbered
references [n] point to its list). Two cautions first.

- **Benchmarks don't transfer as numbers.** WildfireSpreadTS (WSTS) predicts
  tomorrow's active-fire map; TS-SatFire predicts newly burned area only; NDWS
  uses 1 km MODIS. Treat other papers' scores as evidence of *direction*, not
  targets.
- **Ties are the norm.** On TS-SatFire prediction the top three models sit within
  one seed standard deviation (chapter 8). Any experiment needs at least 3 seeds
  before it is called a win.

Pathways are ordered by **evidence ÷ cost**. Each ends with how it becomes one
file in `firepred/models/zoo/` (see the "Adding a model" section of
[`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)).

---

## Tier 1: fix the training signal before changing the architecture

### 1a. Loss and model selection (fixes C1-C3)
**Idea.** Grade the model the way you will score it, with a loss that keeps
rare positives loud.
**Evidence.** In WSTS+ (Lahrichi et al., WACV 2026 [13]), the biggest gains came
from training choices, not architecture: dropping **focal loss** cost 24% AP,
and also selecting checkpoints by F1 instead of AP cost 29% in total. With those
fixed, a ResNet-18 U-Net and U-TAE beat attention backbones (SwinUnet, SegFormer).
Losses built on the evaluation metric beat (weighted) CE when the metric is
Dice/IoU (Eelbode et al. [55]).
**Try.** One output channel with sigmoid; focal loss [46] or Tversky /
focal-Tversky [47, 48] (Tversky lets you trade false positives against false
negatives directly); Lovász for IoU [50]; boundary loss [51] for the perimeter
where new burn happens. Select checkpoints on validation fire-F1 or AP, and tune
the 0.5 threshold on validation.
**Risk.** Calibration shifts, so re-tune the threshold. Boundary loss needs
distance maps.
**Where.** `firepred/training/objectives.py` (a new `--loss` name) and
`training/checkpoints.CheckpointStore` (select on a metric). No model change, so
it applies to every zoo model at once. **Available now:** `--loss dicefocal`
(`--lambda-dice` x sigmoid Dice + `--lambda-focal` x focal loss on the fire
channel, no cross entropy; chapter 7).

### 1b. Report what the score hides (fixes D1)
**Idea.** Measure the share of empty test windows, and report pooled F1/IoU and
AP next to the per-fire mean. Reinke et al. [57] list "empty reference and empty
prediction" as a standard pitfall; WildfireSpreadBench [14] shows model rankings
flip between AP and thresholded F1. One evaluation pass, no training.

### 1c. Give the model whole angles (fixes B4)
**Idea.** Encode each angle as `(sin, cos)` instead of `sin` alone, so north
and south wind (and north- and south-facing slopes) stop looking identical.
Wind as (u, v) components is the meteorological equivalent.
**Evidence.** Code inspection, not a paper: `sin(0°) = sin(180°) = 0`. The
TS-SatFire feature test found weather channels barely used [1]; a wind
direction that is half-ambiguous is one plausible reason.
**Cost.** Three extra input channels (43 -> 46); augmentation must rotate the
pairs. Changes the input width, so it is a new baseline, not a patch.
**Where.** `firepred/data/transforms.py: preprocess` and
`firepred/config/dataset.py` (channel count); every zoo model reads
`spec.in_channels`, so no model changes.

---

## Tier 2: architectures with direct evidence on next-day fire spread

### 2a. A learned "tomorrow" head (fixes A1). The cheapest architecture change.
**Idea.** Replace the fixed average over 6 day-slices with learned weights, so
training decides how much each day's view counts.
**Why it's safe.** Start the 6 weights at zero: softmax(0) = 1/6 each, which is
*exactly* today's mean. The experiment begins at the current model and can only
move if moving helps.
**Where.** This one exists already:
[`firepred/models/zoo/swinunetr3d_tpool.py`](../../firepred/models/zoo/swinunetr3d_tpool.py)
(`-m swinunetr3d_tpool`). It is a SwinUNETR-3D plus 6 parameters. A richer
variant: per-pixel attention pooling over T, as U-TAE does.

### 2b. U-TAE: collapse time early with attention (fixes A1, A4)
**Idea.** Encode each day with a shared 2-D CNN; at the coarsest scale, a
lightweight temporal-attention module (L-TAE) decides *per location* which days
matter, and its attention masks pool every skip connection over time. Output is
directly `(B, K, H, W)`.
**Evidence.** Best model on WildfireSpreadTS (AP 0.372 [12]) and in WSTS+
(**0.478 AP** with a ResNet-18 encoder [13]), at 1.1M-14.6M parameters (vs our
33M). Designed for satellite image time series (Sainte Fare Garnot & Landrieu,
ICCV 2021 [23]).
**Risk.** WildfireSpreadBench [14] found U-TAE predicts 16-23x the observed
fire area at threshold 0.5, so tune the threshold (1a).
**Where.** `zoo/utae.py`: per-day `UnetrBasicBlock`s (2-D or 3-D with
time-kernel 1), an L-TAE module, and a decoder from `monai.networks.blocks`.
`(B, K, H, W)` output is allowed by the registry contract.

### 2c. Physics-aware attention bias (wind and slope)
**Idea.** Swin already adds a learned bias per relative offset (chapter 4). Make
part of it physical: add `α · cos(angle(offset) − wind_direction)` and a slope
term, so tokens downwind and uphill of a fire attend to it *before* learning
anything.
**Evidence.** Esparza et al. [16] added wind/slope attention bias, retrieval and
gating to SwinUNETR on NDWS: F1 0.4216 (0.4292 as an ensemble), the strongest
SwinUNETR result in the notes. Caveat: the learned wind bias matched input wind
but only weakly matched observed fire displacement. TS-SatFire's own feature
test found weather channels barely used [1], and a bias forces their use.
**Risk.** 375 m / 24 h is coarse for front physics; preprint evidence.
**Where.** `models/blocks/window_attention_physics.py` (subclass
`WindowAttentionV1`, add the bias in `forward`), selected through a new
`attn_version` in `swin_block.py`, and `zoo/swinunetr3d_physics.py`. The wind
channels (11, 12) reach attention through `ModelSpec.options` or a side input.

### 2d. Factorized space-time attention (fixes A2)
**Idea.** Instead of one `(6, 4, 4)` window, alternate a *temporal* attention
(each location, all 6 days) with a *spatial* attention over a much larger window
(e.g. 16 x 16 tokens) or the whole map at coarse stages. TimeSformer's
"divided" attention was its best variant [21]; ViViT offers several
factorizations [22].
**Risk.** Data-hungry with about 125 training fires; under fair tuning,
attention backbones trailed ResNet-18 on WSTS+ [13].
**Where.** `blocks/divided_attention.py` reusing `window_ops` (a window of
`(6, 1, 1)` for time, `(1, k, k)` for space), and `zoo/swin_divided.py`.

---

## Tier 3: better inputs

### 3a. Foundation-model features, the right way (fixes B1, B3)
**Idea.** Use pretrained representations of the ground instead of raw
reflectances and 17 one-hot classes. TESSERA [37] is already wired in (annual
128-d embeddings per 10 m pixel, averaged to 375 m). Prithvi-EO-2.0 [34] is a
temporal multispectral foundation model (burn-scar IoU 83.2 at 600M
parameters). WSTS+ found pretraining nearly always helped [13].
**Change.** Stop repeating static maps across days: encode TESSERA/terrain once
with a small 2-D branch and inject them by FiLM (per-channel scale and shift) or
cross-attention at each scale.
**Risk.** Annual 10 m embeddings vs daily 375 m dynamics; 172 input channels
risks WSTS's "more channels can hurt" finding [12]. Run the matched no-TESSERA
baseline on the same fires (the notebook's `TESSERA_INPUTS = False`).
**Where.** `zoo/swinunetr3d_static.py`: a `static` branch and the
Swin encoder from `blocks/swin_encoder.py`. The data side already returns
`static` separately (`data/fire_dataset.py`).

### 3b. More data
WildfireSpreadTS covers 607 fires (WSTS+ 1,005) at the same 375 m resolution [12, 13],
against TS-SatFire's ~125 training fires. Pretraining on WSTS next-day active
fire, then fine-tuning on TS-SatFire's newly-burned target, is plausible but
untested.

---

## Tier 4: longer bets

| Pathway | One-line idea | Evidence | Main risk |
|---|---|---|---|
| **ConvLSTM / PredRNN / SimVP** [24-27] | Models built to forecast the *next frame*, the bias the paper says segmentation nets lack [1] | Mixed: ConvLSTM trails U-Net in AP [12] but has the best F1/IoU in [14] | Instability; little gain at T ≤ 6 |
| **Earthformer** [28] | Cuboid attention plus global vectors for Earth-system forecasting | Strong on weather nowcasting (SEVIR) | Built for dense regression, not sparse masks |
| **Mamba family** [29-33] | Global receptive field at linear cost over 98,304 tokens | **No next-day fire paper found** | Scan order; small data |
| **Neural operators / PINNs** [40-43] | Learn fire spread as a PDE solution, or constrain with Rothermel spread | Simulation-based results only | No satellite next-day study found; coarse data |
| **Generative / probabilistic** [44, 45] | Sample many plausible tomorrows, not one map (fixes D2) | Diffusion and flow-matching surrogates on simulations; flow matching on WSTS under-predicts area [14] | Sampling cost; single-sample F1 misleads |

---

## Engineering that makes every experiment cheaper (fixes D3)

Not architecture, but it multiplies how many experiments fit in a Kaggle week:

1. `torch.autocast` (bf16/fp16) around the forward pass. The `GradScaler` is
   already there. This changes numerics slightly, so it gets its own baseline.
2. DDP instead of DataParallel; GroupNorm or SyncBN instead of per-GPU BatchNorm
   on 4 samples.

## A suggested order

1. **1b** (measure), then **1a** (loss + selection + threshold) and **1c**
   (sin + cos angles) on SwinUNETR-3D *and* U-Net-3D, 3 seeds each. Expect the
   largest gain here.
2. **2a** `swinunetr3d_tpool`: one flag, starts identical to the baseline.
3. **2b** U-TAE: the strongest external evidence.
4. **3a** static branch for TESSERA, against the matched baseline.
5. **2c** physics bias, if 1-4 leave the weather channels unused.

Next: [11. Glossary](11-glossary.md).
