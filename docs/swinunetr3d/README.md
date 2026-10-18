# SwinUNETR-3D, from the ground up

A from-scratch explanation of the model this repo trains for next-day wildfire
spread: what it computes, why it was chosen, where it falls short, and what
the research says to try next. It assumes only basic familiarity with neural
networks. Every number comes from the code (`tools/trace_shapes.py`
regenerates [shapes.md](shapes.md)) or from a cited source
([research_notes.md](research_notes.md)).

## The whole model in one breath

Cut 6 days of a 96 km x 96 km fire region (43 measurements per 375 m pixel)
into small patches. Let each patch exchange information with the patches near it,
across **all six days at once**, inside 3 km windows that shift so information
can travel. Repeat at four zoom levels until each patch summarises 12 km. Then
climb back down with convolutions, pasting in the fine detail saved at each
zoom level, until every pixel has a "will it newly burn tomorrow?" score for each of
the 6 days. Average those 6 scores, and threshold.

## Reading order

| # | Chapter | You'll understand |
|---|---|---|
| 1 | [The question](01-the-question.md) | Inputs, the "newly burned" target, the 1-in-447 imbalance, F1/IoU |
| 2 | [From pixels to pictures](02-from-pixels-to-pictures.md) | Why per-pixel models fail; convolution; U-Net |
| 3 | [Attention from scratch](03-attention.md) | Query/key/value; heads; why full attention is impossible here |
| 4 | [Windows and shifts](04-windows-and-shifts.md) | Swin's (6, 4, 4) windows, shifting, masks, relative position bias |
| 5 | [The encoder staircase](05-the-encoder-staircase.md) | Patch embedding, patch merging, four stages, no temporal downsampling |
| 6 | [The decoder](06-the-decoder.md) | UNETR blocks, skips, where parameters vs compute live, the mean over days |
| 7 | [The learning signal](07-learning-signal.md) | Dice + weighted CE and its quirks, augmentation, what's logged |
| 8 | [Why this model, honestly](08-why-this-model.md) | The benchmark (a tie with U-Net-3D), and why that's unsurprising |
| 9 | [Shortcomings](09-shortcomings.md) | Specific weaknesses, each tied to code (incl. sine-only angles) |
| 10 | [Pathways](10-pathways.md) | Research-backed next steps, ranked, each as a `zoo/` file |
| 11 | [Glossary](11-glossary.md) | Terms |

Reference: [shapes.md](shapes.md) (every module's input/output shape,
parameter count and compute, generated), [research_notes.md](research_notes.md)
(verified facts and 58 references).

## Code map

| Concept | File |
|---|---|
| The model | `firepred/models/zoo/swinunetr3d.py` |
| Swin encoder / one stage / attention / merging / windows | `firepred/models/blocks/swin_encoder.py`, `swin_block.py`, `window_attention.py`, `patch_merging.py`, `window_ops.py` |
| Mean over days | `firepred/models/registry.py: forward_model` |
| Loss / metrics | `firepred/training/objectives.py` |
| Inputs | `firepred/config/dataset.py`, `firepred/data/transforms.py` |

## Interactive version

[explainer.html](explainer.html) is a single-file web page with the same story:
an imbalance demo, a window/shift visualizer that reproduces
`window_ops.compute_mask` exactly (checked token by token), the stage staircase,
parameter vs compute bars, the benchmark tie and the pathway map. Open it in a
browser; it needs no server.
