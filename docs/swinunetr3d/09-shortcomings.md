# 9. Shortcomings

Each item below points to the code that causes it, explains why it hurts, and
says which direction might fix it (details and evidence in
[chapter 10](10-pathways.md)). Items marked **(results)** change trained models
if fixed, so each fix needs its own experiment; see
[`docs/KNOWN_ISSUES.md`](../KNOWN_ISSUES.md).

## A. The model's shape

### A1. The "forecast" is an average of six hindcasts
`firepred/models/registry.py: forward_model` takes the decoder's output
`(B, 2, 6, 256, 256)`, one map per *input* day, and averages over the 6 days.
Nothing in the network is a "tomorrow" head. Every day-slice is pushed to
predict the same next-day target, and they are blended with fixed, equal
weights. The day-6 slice, which sits right next to the target in time,
counts no more than the day-1 slice. The mean also caps confidence: one confident
slice can't override five unsure ones.
*Direction:* a learned temporal head (attention pooling as in U-TAE, a 1 x T
convolution, or predicting from the last slice), or collapsing time inside the
network (4b, 4c).

### A2. Wide in time, narrow in space
The window is `(6, 4, 4)`: every token sees all 6 days but only a 4 x 4 patch
neighbourhood: 3 km at stage 1, 24 km at stage 4 (see [shapes.md](shapes.md)).
Wind-driven runs, smoke and weather systems act over tens of kilometres, yet the
first-stage attention, the only stage at near-pixel resolution, sees 3 km.
Shifting windows spreads information only one window per block.
*Direction:* factorized attention that spends the budget on space (4a), larger
or cross-shaped windows, or global tokens (4d, 4e).

### A3. A transformer in name, a CNN by weight
89% of the 33.2M parameters are convolutions (`encoder1-4`, `encoder10`,
`decoder1-5`). The bottleneck block `encoder10` alone holds 54%, two
576-channel 3 x 3 x 3 convolutions on an 8 x 8 grid. The parameter budget sits
where the map is coarsest, while the compute sits where it is finest:
`encoder1` and `decoder1` at 256 x 256 do 46% of the work. Attention is about
a tenth of both. This
is part of why SwinUNETR-3D ties U-Net-3D (chapter 8).
*Direction:* rebalance (a thinner bottleneck, more Swin depth at stages 1 and 2),
or accept that the backbone isn't the bottleneck and work on the loss and
inputs (4i).

### A4. History length is baked in
`window_size=(spec.ts_length, 4, 4)` makes the relative-position table
`(2T-1) x 7 x 7` entries per head. A model trained on 6 days can't load into a
5-day or 8-day configuration, and can't take a fire with fewer days of history.
*Direction:* attention over time with bias by *relative day* (already the case)
and a window that doesn't depend on T, or a recurrent/temporal-attention design
(4b, 4c).

### A5. Only relative position, never absolute context
Attention knows "two days earlier, one patch west", but not the date, the day of
the fire, latitude, or whether the window sits at the crop edge. Fire seasons
and climates differ.
*Direction:* add small scalar inputs (day-of-year, days since ignition) as
channels or tokens.

## B. The inputs

### B1. Static data copied six times
Terrain, land cover and (with `--tessera`) the 129 TESSERA channels are the same
every day, but they enter as part of a `(C, 6, H, W)` volume. With TESSERA the
first block (`encoder1`, at full 256 x 256 resolution, already 19% of the
compute without TESSERA) convolves 172 channels on each of the 6 days, and 129
of them are the same on every day. That is compute and memory spent
re-reading a constant. The paper found most auxiliary channels barely change F1,
and WildfireSpreadTS found more channels can *hurt* optimization.
*Direction:* encode static layers once (a separate 2-D branch) and inject them
by FiLM or cross-attention, not as repeated channels (4f).

### B2. A fixed 96 km crop
`config.dataset.CROP_OFFSET/CROP_SIZE` cut every GeoTIFF to rows/cols
128..384. Fires larger than 96 km or off-centre are clipped, and growth at the
crop edge has no context beyond it.
*Direction:* fire-centred crops, or sliding tiles at inference.

### B3. Few fires, noisy labels
About 125 training fires (2017-2020). Labels come from VIIRS active-fire
detections (missed under cloud and smoke) and NIFC perimeters (coarse; the paper
hand-picked a rule per test fire, `label_sel`). The train/val rule disagrees
between the code default, the config and the data-grab notebook
(KNOWN_ISSUES #5). Augmentation is limited to 8 rigid orientations.
*Direction:* pretraining (4f), a bigger dataset (WildfireSpreadTS has 607 fires),
and consistent label rules.

### B4. Half of every angle is thrown away **(results)**
`data/transforms.preprocess` replaces wind direction, aspect and forecast wind
direction with `sin(angle)` only [released]. The sine cannot tell north from
south: 0° and 180° both give 0, and 30° and 150° both give 0.5. A north wind
and a south wind, the difference between fire running toward or away from a
town, look identical to the model. Aspect loses north- versus south-facing
slopes, which matters for how dry the fuel is.
*Direction:* feed `(sin, cos)` pairs (3 extra channels), or wind as (u, v)
components. Cheap, but it changes the input width and every result.

## C. The training signal

### C1. Two probabilities in one loss **(results)**
Dice uses per-channel sigmoids, CE a softmax across the two channels, and
prediction thresholds the fire sigmoid (chapter 7). The decision boundary CE
trains (`z_fire > z_bg`) is not the one used at test time (`z_fire > 0`).
*Direction:* one output channel with sigmoid + BCE/Dice/focal, or softmax
everywhere with an argmax decision.

### C2. The class weight swamps Dice **(results)**
MONAI applies `weight = (1, 446.78)` to the Dice term too, so the loss starts
around 215 Dice + 6 CE: mostly fire-Dice. The "1:1 Dice + CE" in the paper means
something different under MONAI < 1.3, which the paper doesn't pin.
*Direction:* set the Dice and CE weights separately; try focal / Tversky / Lovász
(4i). WSTS+ found the loss gave the biggest single gain.

### C3. Selecting by loss, scoring by F1 **(results)**
`BestCheckpoints` keeps the lowest *validation loss*; the paper grades *F1/IoU*
at a 0.5 threshold. WSTS+ found that selecting models by the target metric
mattered a great deal (selecting by AP vs F1: tens of percent).
*Direction:* select on validation fire-F1 or AP; tune the threshold on validation.

### C4. Weight decay that never happens **(results)**
`weight_decay = lr / 10` is logged to W&B as if used, but Adam never receives it
(KNOWN_ISSUES #1). With few fires and 33M parameters, regularization matters.

## D. Measuring and running

### D1. Empty windows score a perfect 1
When a test window has no new fire and the model predicts none, F1 = IoU = 1
(`training/metrics.py`; sklearn's `zero_division=1` in the released code). One
false-positive pixel on such a day drops it to 0. Per-fire averages therefore
depend on how many quiet days a fire has, and reward conservative models. In a
sample of 7 real training fires, 15% of windows have no new fire (chapter 1);
the test-set share hasn't been measured yet.
*Direction:* report pooled F1/IoU and AP alongside, and state the convention.

### D2. One answer, no uncertainty
Fire spread at 375 m and 24 h has a large random component (spotting, wind
gusts, suppression). The model outputs a single map, with no notion of "60% chance
this ridge burns".
*Direction:* calibrated probabilities, ensembles, or generative models (4h).

### D3. Slower than it needs to be
- `GradScaler` without `autocast`: training runs in fp32 (memory note:
  ~6.3 s per batch of 8 on Kaggle's 2 x T4).
- `compute_mask` builds an unused autoregressive time mask in every stage on
  every forward pass (KNOWN_ISSUES #10).
- `nn.DataParallel` splits a batch of 8 into 4 + 4. BatchNorm then estimates
  statistics from 4 samples per GPU, which is noisy, and DataParallel is slower
  than DDP.

*Direction:* `torch.autocast` + bf16/fp16, skip the unused mask, GroupNorm or
SyncBN with DDP. These are faster runs, not different math, except the norm
change.

Next: [10. Pathways](10-pathways.md). What the research says to try, and how
each idea becomes one file in `firepred/models/zoo/`.
