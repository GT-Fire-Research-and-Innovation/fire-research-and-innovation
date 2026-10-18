# 5. The encoder staircase: from 3 km windows to a 96 km view

Chapter 4 gave us one Swin stage: attention inside `(6, 4, 4)` windows,
alternating regular and shifted. The encoder (`SwinTransformer` in
`firepred/models/blocks/swin_encoder.py`) stacks four of them, and between
stages it makes the token grid coarser, the same zoom-out trick as the U-Net in
chapter 2.

## Step 0: patches become tokens

`patch_embed` is a convolution with kernel = stride = `(1, 2, 2)`: each
1-day x 2 x 2-pixel patch becomes a 36-number vector (feature size `F = 36`).

```
(B, 43, 6, 256, 256)  ──patch_embed──►  (B, 36, 6, 128, 128)
```

Time is untouched: a token never mixes two days at this step. Space shrinks 2x.

## Steps 1-4: talk in windows, then merge

Each stage `layersK` (a `BasicLayer`) runs 2 Swin blocks (regular windows, then
shifted), then **patch merging** (`firepred/models/blocks/patch_merging.py`):

```
take each 2 x 2 group of neighbouring tokens  →  concatenate their vectors (4C numbers)
→ LayerNorm → Linear to 2C numbers
```

The grid halves in H and W, and the vector doubles in length. Each new token
summarises four old ones, so the same 4 x 4-token window now spans twice the
ground.

| Stage | Attention runs on | Windows | One window sees | Leaves with |
|---|---|---|---|---|
| 1 `layers1` | (36, 6, 128, 128) | 1,024 | 6 days x 3 km x 3 km | (72, 6, 64, 64) |
| 2 `layers2` | (72, 6, 64, 64) | 256 | 6 days x 6 km x 6 km | (144, 6, 32, 32) |
| 3 `layers3` | (144, 6, 32, 32) | 64 | 6 days x 12 km x 12 km | (288, 6, 16, 16) |
| 4 `layers4` | (288, 6, 16, 16) | 16 | 6 days x 24 km x 24 km | (576, 6, 8, 8) |

(Channels first: `(C, T, H, W)`. Generated numbers: [shapes.md](shapes.md).)

Read the staircase as a team of analysts reporting upward. Stage 1 analysts
watch 3 km squares at fine detail: "the fire edge is here, it moved 1 km east
since yesterday". Stage 4 analysts each cover 24 km and see the whole fire:
"this fire is running north-east along a valley". Each stage's report is
richer (more numbers per token) and coarser (fewer tokens).

## Time stays at 6 days throughout

Patch merging *could* also merge pairs of days (6 -> 3). The released TS-SatFire
code did that (`merge_temporal=True`), but the paper says "we do not use
downsampling in the temporal dimension in any of our models", and this repo
follows the paper by default. `--swin-temporal-downsample` restores the released
behaviour for old checkpoints (`firepred/training/checkpoints.load_checkpoint`
refuses a mismatch). With 6 days and windows spanning all of them, there's
nothing to gain by halving time, and every stage keeps per-day detail.

## Five reports for the decoder

The encoder returns **five** feature maps, one per scale, each
LayerNorm-normalized over channels (`proj_out` with `normalize=True`):

```
x0  (B,  36, 6, 128, 128)   right after patch_embed
x1  (B,  72, 6,  64,  64)   after stage 1
x2  (B, 144, 6,  32,  32)   after stage 2
x3  (B, 288, 6,  16,  16)   after stage 3
x4  (B, 576, 6,   8,   8)   after stage 4: 64 tokens per day, each ~12 km wide
```

These are the rungs the decoder climbs back down (chapter 6).

## The cost, in numbers

The whole Swin encoder holds **3.56M parameters, 10.7%** of the model. Most of
them are in stage 4, where the 576-number vectors make every linear layer big.
The attention score tables are tiny by comparison: 96 x 96 per window per head.
Swin's design kept attention affordable; the cost of this model lives elsewhere
(chapter 6).

Next: [6. The decoder](06-the-decoder.md).
