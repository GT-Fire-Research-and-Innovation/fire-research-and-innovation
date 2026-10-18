# 6. The decoder: climbing back to pixels

The encoder produced five reports, from fine (x0, 128 x 128) to coarse (x4,
8 x 8). The decoder turns them into a 256 x 256 answer. This part of
SwinUNETR is **pure convolution** (MONAI's UNETR blocks), and it is where most
of the parameters live. All code below is `SwinUNETR.forward` in
`firepred/models/zoo/swinunetr3d.py`, which has the shape on every line.

## Two kinds of building block

- **`UnetrBasicBlock`** (named `encoderK`): a residual block of two 3 x 3 x 3
  convolutions with BatchNorm and LeakyReLU, plus `x + ...`. It refines a
  feature map without changing its size. Think of it as "re-read this report
  carefully".
- **`UnetrUpBlock`** (named `decoderK`): (1) a *transposed* convolution with
  stride (1, 2, 2) that doubles H and W, (2) concatenation with the skip
  feature map of the new size, (3) a residual block. "Zoom in, put the
  finer-scale report beside it, reconcile the two."

## The forward pass, rung by rung

F = 36. Shapes are `(B, C, T, H, W)` with B = 1.

```python
hidden = self.swinViT(x_in)              # x0..x4, chapter 5

enc0 = self.encoder1(x_in)               # (1,  36, 6, 256, 256)  convs on the RAW 43-channel input
enc1 = self.encoder2(hidden[0])          # (1,  36, 6, 128, 128)
enc2 = self.encoder3(hidden[1])          # (1,  72, 6,  64,  64)
enc3 = self.encoder4(hidden[2])          # (1, 144, 6,  32,  32)

dec4 = self.encoder10(hidden[4])         # (1, 576, 6,   8,   8)  bottleneck
dec3 = self.decoder5(dec4, hidden[3])    # (1, 288, 6,  16,  16)  up + skip x3
dec2 = self.decoder4(dec3, enc3)         # (1, 144, 6,  32,  32)
dec1 = self.decoder3(dec2, enc2)         # (1,  72, 6,  64,  64)
dec0 = self.decoder2(dec1, enc1)         # (1,  36, 6, 128, 128)
out  = self.decoder1(dec0, enc0)         # (1,  36, 6, 256, 256)
logits = self.out(out)                   # (1,   2, 6, 256, 256)  1x1x1 conv: 36 -> 2
```

Three details are worth slowing down for:

1. **`encoder1` reads the raw input at full resolution.** The Swin encoder's
   finest report is 128 x 128 (2 x 2-pixel tokens). For a pixel-exact answer
   the decoder needs something at 256 x 256, so a convolutional block looks
   directly at the 43 input channels. It is the only block that sees
   individual pixels, and, with `decoder1`, one of the two most expensive to
   *run*: both touch all 6 x 65,536 positions (see "Where the work happens").
2. **The bottleneck is a heavyweight.** `encoder10` applies two 3 x 3 x 3
   convolutions with 576 input and output channels: 2 x 576 x 576 x 27 ≈
   **17.9M parameters, 54% of the whole model**, on an 8 x 8 grid. Most of
   SwinUNETR-3D's capacity sits at its coarsest scale.
3. **x3 skips the refining block.** Unlike x0-x2, the stage-3 report goes
   straight into `decoder5` as its skip, without its own `encoderK`. (That's the
   original Swin UNETR wiring.)

## The last step: six opinions, one answer

The network's output still has a time axis: `(1, 2, 6, 256, 256)`, two logits
per pixel **per input day**. Because attention mixed all 6 days inside every
window, each day-slice is a full-context prediction "seen from" that day. The
trainer reduces them with a plain average (`firepred/models/registry.py`):

```python
outputs = model(batch)          # (B, 2, 6, 256, 256)
outputs = outputs.mean(dim=2)   # (B, 2, 256, 256): the next-day map
```

Like averaging six forecasters who all read the same history but sit at
different days of it. It's simple and stable, but the weights are fixed and
equal (the day-6 view, closest to tomorrow, counts no more than day 1) and
nothing in the network is ever asked to *be* a tomorrow-predictor. Chapter 9
(A1) and chapter 10 (2a) return to this. A one-line, learnable alternative ships
as `-m swinunetr3d_tpool`.

## Where the parameters are

| Part | Parameters | Share |
|---|---|---|
| Swin encoder (all attention) | 3.56M | 10.7% |
| `encoder1-4` + `encoder10` (convolution) | 19.47M | 58.7% |
| `decoder1-5` (convolution) | 10.14M | 30.6% |
| `out` | 74 | 0.0% |
| **Total** | **33.17M** | |

## Where the work happens

Parameters and compute live in opposite places. Counting multiply-accumulates
for one window (`tools/trace_shapes.py`, [shapes.md](shapes.md)):

| Part | Compute (GMAC) | Share |
|---|---|---|
| Swin encoder (all attention stages) | 16.4 | 10.2% |
| `encoder1-4` + `encoder10` | 58.3 | 36.3% |
| `decoder1-5` | 86.1 | 53.5% |
| **Total** | **160.9** | |

- **`decoder1` (26.6%) and `encoder1` (19.1%) do 46% of the work.** They
  run 3 x 3 x 3 convolutions on all 6 x 256 x 256 positions; `decoder1`
  sees 72 channels (36 upsampled + 36 from the skip).
- **The bottleneck `encoder10` holds 54% of the parameters but only 4.3% of the
  compute**, because it runs on just 6 x 8 x 8 positions.
- **All the attention costs about a tenth of the work**: Swin made attention
  cheap, so the transformer is neither where the weights nor where the time go.

So SwinUNETR-3D is better described as **a convolutional U-Net whose encoder
features come from a Swin transformer**: by parameters (89% convolution) and by
compute (90% convolution). Chapter 8 shows why that matters for interpreting
the benchmark.

Next: [7. The learning signal](07-learning-signal.md).
