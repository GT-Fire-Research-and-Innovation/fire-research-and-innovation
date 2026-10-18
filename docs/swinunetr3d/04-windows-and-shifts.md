# 4. Windows and shifts (the "Swin" part)

Swin Transformer (Liu et al., 2021, "**S**hifted **win**dows") makes attention
affordable with two ideas: talk only inside small windows, and move the
windows every other block.

## Idea 1: local windows

Cut the token grid into non-overlapping **windows** and run attention inside
each window separately. SwinUNETR-3D's window is `(T, 4, 4)` = `(6, 4, 4)`
tokens (`firepred/models/zoo/swinunetr3d.py`: `window_size=(spec.ts_length,
*WINDOW_HW)`):

```
one window = all 6 days x 4 x 4 patches = 96 tokens
           = 6 days x 8 x 8 pixels      = 6 days x 3 km x 3 km of ground (stage 1)
```

At stage 1 the (6, 128, 128) token grid splits into 1 x 32 x 32 = **1,024
windows**. Each computes 96 x 96 = 9,216 scores, about 9.4 million in total
instead of 9.7 billion: **1,024 times cheaper**, and the score tables fit easily
in memory. (`firepred/models/blocks/window_ops.py: window_partition` does the
cutting and `window_reverse` stitches the results back.)

Notice the shape of the window: **the whole time axis but only a 4 x 4 patch of
space**. That is the central design choice for this task:

- **Time: complete.** Every token sees all 6 days at its location and its
  neighbours. "This patch was cold on day 1, warm on day 4, hot on day 6" is
  visible in one attention step.
- **Space: local.** At stage 1 a token sees only 3 km x 3 km. Longer-range
  context has to be built up (ideas 2 and 3 below, and the stages in chapter 5).

Because the window already spans all 6 days, the code never shifts in time:
`get_window_size` sets the shift to 0 along any axis the window fully covers,
so the effective shift is `(0, 2, 2)`.

## Idea 2: shift the windows every other block

With fixed windows, a fire at the right edge of one window could never inform a
token just across the border, because they sit in different windows. So each
Swin stage alternates two kinds of block (`BasicLayer` in
`firepred/models/blocks/swin_block.py`):

```
block 0: regular windows                    block 1: windows shifted by half a window
┌───┬───┬───┐                               ┌─┬───┬───┬─┐
│ A │ B │ C │                               ├─┼───┼───┼─┤
├───┼───┼───┤        shift by (2, 2)        │ │ A'│ B'│ │   each new window straddles
│ D │ E │ F │   ─────────────────────►      ├─┼───┼───┼─┤   four old ones, so information
└───┴───┴───┘                               └─┴───┴───┴─┘   crosses the old borders
```

The implementation is a neat trick. Instead of building odd-sized windows at the
edges, `torch.roll` cyclically shifts the whole token grid by 2 tokens in H and W
(`SwinTransformerBlock.forward_part1`), runs the *regular* windowing, and rolls
back afterwards. Rolling wraps tokens from the bottom edge to the top, which
would make physically distant ground "neighbours". So `compute_mask` labels
which region every token came from, and any pair from different regions gets
**-100** added to its score. After softmax, e^-100 ≈ 0: those pairs are
ignored.

After one regular block and one shifted block, information has moved across
window borders. Repeat, and it keeps spreading.

## Idea 3: tell attention *where* things are

Attention by itself is **position-blind**: the weighted average of values comes
out the same however the tokens are shuffled. For fire this is fatal: "fire 2
patches west of me" and "fire 2 patches east of me" mean opposite things under
an east wind.

Swin adds a learned **relative position bias** to every score before the
softmax (`relative_position_bias_table` in `window_attention.py`). For each
head there is one learned number for every possible offset between two tokens
in a window:

```
offsets in time:  -5 .. +5   (11 values)
offsets in rows:  -3 .. +3   (7 values)      11 x 7 x 7 = 539 learned numbers per head
offsets in cols:  -3 .. +3   (7 values)
```

So head 2 can learn, before looking at any content, "pay extra attention to the
token one day earlier, one patch west". Content (q·k) and geometry (the bias)
add up in the same score. This bias is also the model's **only** sense of
time order. There is no separate "day number" input.

## What one Swin block costs and sees

| | Stage 1 (finest) |
|---|---|
| Tokens per window | 96 (6 days x 4 x 4) |
| Windows per sample | 1,024 |
| Ground per window | 3 km x 3 km, all 6 days |
| Learned position biases | 539 per head x 3 heads |
| Shift in the odd block | (0, 2, 2) tokens = 1.5 km |

`docs/swinunetr3d/shapes.md` (generated from the real model by
`tools/trace_shapes.py`) lists the same numbers for every stage.

**An option:** `window_attention_v2.py` (Swin V2's scaled cosine attention)
replaces V1 with `--model-option attn_version=v2`; the default is `"v1"`, as in
the released code (the paper does not say which it used). The released code
also had an "autoregressive" variant that stopped days from looking ahead in
time; it was never used and is gone. Within a window all 6 days are in the past
relative to the target, so looking "ahead" inside the window leaks nothing.

Next: [5. The encoder staircase](05-the-encoder-staircase.md). How four Swin
stages turn 3 km windows into a 96 km view.
