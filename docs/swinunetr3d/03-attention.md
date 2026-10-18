# 3. Attention from scratch

## Tokens: chopping the input into words

A transformer reads a list of **tokens**, like the words of a sentence. Here
the first layer of SwinUNETR-3D (`swinViT.patch_embed`, a convolution with
kernel = stride = (1, 2, 2)) cuts each window into patches of **1 day x 2 x 2
pixels** and describes each patch with 36 numbers (the *feature size*, `-ed
36`). One window becomes

```
6 days x 128 x 128 patches = 98,304 tokens, each a vector of 36 numbers.
```

Think of each token as a tiny analyst responsible for 750 m x 750 m of ground
on one day.

## One analyst's turn: ask, compare, collect

Attention lets every analyst update its notes by consulting the others. Each
token makes three vectors from its own 36 numbers, using three learned
linear maps (`self.qkv` in `firepred/models/blocks/window_attention.py`):

| Vector | Plain meaning | Example |
|---|---|---|
| **query** q | "what I am looking for" | a grass token under an east wind: *"is there fire to my west?"* |
| **key** k | "what I can offer, as a label" | a token at the fire front: *"hot, burning, west side"* |
| **value** v | "what I'll tell you if you listen" | the fire token's actual content |

Then, for one token:

1. **Compare**: score every other token by how well its key matches my query:
   `score_j = q · k_j / sqrt(d)`. The dot product is large when the two vectors
   point the same way. Dividing by sqrt(d) (d = 12 per head here) keeps the
   scores in a sensible range.
2. **Normalize**: `weights = softmax(scores)`. The weights are positive and sum
   to 1, a budget of attention spread over the other tokens.
3. **Collect**: `output = Σ_j weights_j · v_j`, a weighted average of what the
   others said, mostly from the tokens that matched.

In the code (`WindowAttentionV1.forward`) this is literally:

```python
q, k, v = qkv[0], qkv[1], qkv[2]
q = q * self.scale                       # 1 / sqrt(d)
attn = q @ k.transpose(-2, -1)           # every query against every key
attn = attn + relative_position_bias     # chapter 4
attn = self.softmax(attn)                # weights sum to 1
x = (attn @ v)                           # weighted average of values
x = self.proj(x)                         # mix the heads back together
```

The difference from a convolution is that the mixing weights are **computed
from the content**. Under an east wind the grass token's query matches tokens to
its west; flip the wind and the same learned maps produce a query that matches
the east. The network doesn't need a separate stencil for each wind direction.

## Several questions at once: heads

One query asks one question. **Multi-head** attention splits the 36 numbers
into 3 groups of 12 (`-nh 3`), and each head asks its own question with its own
q/k/v maps, for example "where is fire?", "what's uphill?", "what did this place
look like yesterday?". Their answers are concatenated and mixed by `proj`.

## A full transformer block: talk, then think

Attention is *communication* between tokens. A transformer block pairs it with
*computation* on each token (`SwinTransformerBlock` in
`firepred/models/blocks/swin_block.py`):

```
x = x + Attention(LayerNorm(x))     # talk: gather information from other tokens
x = x + MLP(LayerNorm(x))           # think: process it, each token on its own (36 -> 144 -> 36)
```

The `x + ...` **residual** connections mean each block only *adds a
correction* to the notes. That keeps deep stacks trainable. **LayerNorm**
rescales each token's 36 numbers to a standard range before they are used.

## The catch: everyone talking to everyone

Full attention compares every token with every other one. With n tokens that
is n² scores:

```
n  = 98,304 tokens (stage 1)
n² = 9,663,676,416 scores per head, per window
   = 38.7 GB of float32 for one head's score table
```

Our Kaggle GPUs have 16 GB. Even if they had more, almost all of those
comparisons are wasteful: a pixel's fate tomorrow is decided by ground within
a few kilometres, not by a pixel 90 km away. The fix is to let tokens talk
only within **local windows**, and to arrange the windows so information
still travels. That is the "Swin" in SwinUNETR.

Next: [4. Windows and shifts](04-windows-and-shifts.md).
