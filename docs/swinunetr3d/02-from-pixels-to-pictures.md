# 2. From pixels to pictures

## Attempt 1: judge each pixel on its own

The simplest model reads one pixel's column, its 43 channels x 6 days = 258
numbers, and outputs a probability of burning tomorrow. Logistic regression or
a small MLP would do.

It cannot work, and seeing why is the key to everything that follows. Take a
pixel of dry grass, 500 m east of a fire front, with a strong west wind. It
burns tomorrow. Now take an identical patch of dry grass under the identical
west wind, 50 km from any fire. It doesn't. **The two columns of numbers are
the same.** What differs is the *neighbourhood*: is there fire nearby, upwind,
uphill? A per-pixel model never sees that.

The paper has a nice contrast. Its GRU, LSTM and T4Fire models (not in
this repository, which covers the prediction task only) are per-pixel, and
they work for *active-fire detection*: "is this pixel burning right now?" is mostly answered by the
pixel's own heat. *Prediction* is a question about neighbours.

## Attempt 2: look at the neighbourhood (convolution)

A convolution slides a small stencil, say 3 x 3 pixels, across the map. At
each position it multiplies the 9 neighbours' values by 9 learned weights and
adds them up. One stencil might learn "hot pixel to my west", another "steep
slope below me". Two properties make it a good fit:

- **Locality.** Fire spreads to adjacent ground, so nearby pixels matter most.
- **Weight sharing.** The same stencil is used everywhere, because the physics
  of fire is the same in every corner of the map. A pattern learned in one
  place transfers to all others, which matters with only about 150 fire
  examples per window.

In 3-D (`kernel_size=3` on (T, H, W)), the stencil is 3 x 3 x 3: it also mixes
each day with the input days just before and after it, so it can notice
*motion*: "this pixel's neighbour got hot since yesterday".

## The receptive-field problem

One 3 x 3 layer sees 1 pixel in each direction, 375 m. Stack two and you
see 2 pixels; stack *n* and you see *n*. Big fires can advance several
kilometres in a day, and the wind and terrain that drive them are larger still. Seeing 10 km
(27 pixels) in every direction would take about 27 stacked layers at full
resolution, every one of them paying for 65,536 positions x 6 days.

The trick is to **zoom out**. After a few layers, halve the map: 256 -> 128.
Now each cell summarises 2 x 2 pixels, a 3 x 3 stencil spans 2.25 km, and
the computation is a quarter as large. Halve again and again: at 16 x 16, one
cell is 6 km wide and a 3 x 3 stencil covers 18 km.

## U-Net: zoom out to understand, zoom back in to answer

Zooming out loses detail, but the answer must be a sharp, pixel-exact map of the
fire edge. U-Net (Ronneberger et al., 2015) solves both with one shape:

```
256 ──conv──► 256 ─────────────────────────skip────────────────────────► 256 ──► answer
               │                                                         ▲
               ▼ halve                                                   │ double
              128 ──conv──► 128 ────────────skip────────────► 128 ──conv─┘
                             │                                ▲
                             ▼ halve                          │ double
                            64 ──conv──► ... bottleneck ... ──┘
```

- The **encoder** (down the left) builds understanding at coarser and coarser
  scales: "there's a fire front here, moving east, into steep forest".
- The **decoder** (up the right) grows the map back to 256 x 256.
- **Skip connections** hand each decoder level the encoder's map of the same
  size. The coarse levels know *what* will happen; the skips supply *exactly
  where*, such as the precise edge of yesterday's burn.

This is the paper's **U-Net-3D** baseline (`firepred/models/zoo/unet3d.py`):
channels 64 -> 128 -> 256 -> 512 -> 1024, with strides `(1, 2, 2)`, so it
halves space but never time. Six days is short, and the paper keeps every day
("we do not use downsampling in the temporal dimension in any of our models").

## What convolution still can't do well

A convolution's weights are **fixed once trained**. The "hot pixel to my
west" stencil applies the same weights whether the wind blows east or west. The
network can learn to combine stencils to approximate "it depends on the wind",
but only indirectly, and only within its receptive field.

What we would like is **content-dependent mixing**: each location decides,
*based on what it sees*, which other locations to listen to. A pixel under an
east wind should listen hardest to its western neighbours, and a pixel on a
ridge to the slope below. That operation is **attention**.

Next: [3. Attention from scratch](03-attention.md).
