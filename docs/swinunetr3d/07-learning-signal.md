# 7. The learning signal: how the model is told it's wrong

The architecture says what the model *can* compute. The loss decides what it
*will* compute. With one fire pixel per ~447, the loss design matters as much
as the network.

## From logits to a map

The model outputs two numbers per pixel, the *logits* `z_bg` and `z_fire`.
They are not probabilities; any real number is allowed. To decide, the code
squashes each channel with a sigmoid, `p = 1 / (1 + e^-z)`, which maps to
(0, 1), and thresholds at 0.5 (`firepred/training/objectives.py:
Scorer.predictions`). The prediction is "fire" where `sigmoid(z_fire) > 0.5`,
which is the same as `z_fire > 0`.

## Dice loss: optimize the score you're graded on

F1 counts overlaps of yes/no masks, and you can't take a gradient of a count.
**Soft Dice** replaces the masks with probabilities:

```
Dice = 2 Σ p·y / (Σ p + Σ y)        loss = 1 - Dice
```

where y is the 0/1 target and p the predicted probability, summed over pixels.
It is F1 with "how sure" in place of "yes/no". Its strength for rare classes is
that the 65,000 background pixels can't drown out the fire, because the score is
a *ratio* of overlap to size. With no fire at all, the loss is driven by not
predicting any.

The released TS-SatFire code trains with Dice alone (`--loss dice`).

## Cross-entropy: per-pixel honesty

Weighted **cross-entropy** grades every pixel separately: `-log(probability
given to the right answer)`, times a class weight. The paper uses weights
`(1.0, 446.7836)` (background, fire), so missing one fire pixel costs as much as
misclassifying 447 background pixels. CE gives every pixel a useful gradient,
even early in training when the overlap Dice measures is still zero.

The paper switched prediction training from Dice to **Dice + CE, 1:1**
(`--loss dicece`, the default for SwinUNETR-3D) because models trained on Dice
alone "stayed stuck". This is MONAI's `DiceCELoss`
(`firepred/training/objectives.py`).

## Two quirks worth knowing (kept exactly as trained)

1. **Two definitions of probability.** In `DiceCELoss(sigmoid=True)`, the Dice
   part applies a **sigmoid to each channel independently**, so p(bg) and
   p(fire) need not sum to 1. The CE part applies a **softmax across the two
   channels**, so they must. Both push the fire logit up on fire pixels, so
   training works, but the loss measures two slightly different models at once.
   Prediction then uses the sigmoid of the fire channel alone.
2. **The class weight also scales Dice.** MONAI passes `weight` to both terms.
   The Dice part is the average of `1 × (1 − Dice_bg)` and
   `446.78 × (1 − Dice_fire)`, about **215** at the start, next to a CE of
   about **6** (`python -m firepred.cli.check model` prints both). In practice the loss is "mostly fire
   Dice, with a little CE". That's also why reported loss values sit near 220:
   compare runs by F1/IoU, not by loss.

Both are listed in [`docs/KNOWN_ISSUES.md`](../KNOWN_ISSUES.md). Changing either
changes results, so they stay until a deliberate experiment.

## Option: Dice + focal (`--loss dicefocal`)

Focal loss (Lin et al., ICCV 2017) is cross-entropy scaled by `(1 − p_t)^γ`,
where `p_t` is the probability given to the right answer. A background pixel the
model already gets right with p_t = 0.99 is multiplied by 0.0001 (γ = 2); a
missed fire pixel with p_t = 0.2 keeps 64% of its loss. The training signal
concentrates on the hard pixels at the fire front instead of the easy ocean of
background. In the WSTS+ benchmark, dropping focal loss cost 24% AP (chapter 10).

`--loss dicefocal` replaces cross-entropy with focal loss:
`--lambda-dice` × the plain sigmoid Dice of `--loss dice` (both channels, no
class weight) + `--lambda-focal` × a sigmoid focal loss on the fire channel
only (`training/objectives.py: DiceFocalLoss`; `--focal-gamma`, default 2;
`--focal-alpha`, default off, 0.25 is the paper's value). Without the 446.78
class weight both parts live on a similar scale (about 0.7 Dice + 0.1 focal at
the start), so the 1:1 default is a real mix. It is an option, not the default;
its Dice part is logged on the same chart as a `--loss dice` run.

## Optimization

| Setting | Value | Where |
|---|---|---|
| Optimizer | Adam, lr 0.001, constant | `training/trainer.py` |
| Batch | 8 windows | `-b 8` |
| Epochs | 100 | `--epochs` |
| Checkpoint kept | lowest **validation loss** | `training/checkpoints.CheckpointStore` |
| Weight decay | none in practice; `lr/10` is logged but never passed to Adam | KNOWN_ISSUES #1 |
| Gradient scaler | active on CUDA; without autocast it only rescales the loss | `training/trainer.py` |

The best checkpoint is chosen by validation *loss*, not validation F1. Because
the loss is mostly fire-Dice, the two track each other, but not perfectly.

## Making 125 training fires look like more

Data augmentation (`firepred/data/transforms.py: augment`) randomly flips the
window horizontally and vertically and rotates it by 0/90/180/270 degrees. That
gives 8 orientations of every example. The careful part is **physics
consistency**: rotating the map 90° turns an east wind into a north wind, so the
three angle channels (wind direction, aspect, forecast wind direction) are
rotated or reflected along with the pixels:

```python
x = TF.rotate(x, angle)                         # move the pixels
x[DEGREE_CHANNELS] = (x[DEGREE_CHANNELS] - angle) % 360   # and turn the arrows
```

Without the second line, augmentation would teach the model that wind
direction is meaningless. The target and the TESSERA map are transformed the
same way.

## What gets logged

Per epoch, W&B charts (`training/tracker.py`):

- training and validation loss, plus the Dice and CE parts
- fire F1, background F1 and fire IoU, each averaged per window
- validation "Mean IoU / Mean Dice", which average in the background class
  and are therefore not the paper's numbers
- the learning rate
- optionally, test F1/IoU every N epochs (`--test-every`), for watching only

At the end, the best checkpoint scores the 2021 test fires with the paper's
per-fire averaging.

Next: [8. Why this model](08-why-this-model.md).
