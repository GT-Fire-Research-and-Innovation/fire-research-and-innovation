# 8. Why this model, honestly

Sources: [research_notes.md](research_notes.md), which cites the TS-SatFire
paper (Zhao, Gerard, Ban, *Scientific Data* 12:1817, 2025) and the
literature, with references. Numbers below are copied from there.

## The argument for SwinUNETR-3D

Going by chapters 1 to 6, SwinUNETR-3D's design matches the task's structure
point for point:

| What the task needs | What SwinUNETR-3D provides |
|---|---|
| Neighbourhood matters, distant ground matters little | Attention inside local 4 x 4-patch windows; context grows stage by stage |
| *Which* neighbours matter depends on wind, slope, fuel | Content-dependent attention weights plus learned relative-position biases |
| How the fire evolved over 6 days | Each window spans **all** days, and time is never downsampled |
| Pixel-exact edges of new burn | A U-Net decoder with skips from 5 scales, including convolutions on the raw input |
| A fair fight with other models | About 30M parameters, like the paper's other baselines |

It also has a pedigree. Swin UNETR (Hatamizadeh et al., 2022) was built for 3-D
medical volumes, where small, rare structures (tumours) must be segmented
precisely, which is close in spirit to "find the 0.2% of pixels that will
burn". In the TS-SatFire paper, SwinUNETR-3D is the best model for daily
**burned-area mapping** (F1 0.855, IoU 0.768), and a footnote of the paper's seed
study names it the best prediction model.

## What the benchmark actually says

The paper's prediction table (6 input days, 2021 test fires):

| Model | Parameters | F1 | IoU |
|---|---|---|---|
| U-Net-3D | 31.7M | **0.375** | **0.338** |
| SwinUNETR-3D | 33.2M | 0.374 | 0.331 |
| UNETR-3D | 34.8M | 0.371 | 0.336 |
| Attention-U-Net-3D | 94.5M | 0.354 | 0.312 |

Three readings, from most to least important:

1. **The top three are tied.** Across three seeds, the paper reports
   F1 0.371 ± 0.003. U-Net-3D's lead over SwinUNETR-3D (+0.001 F1) is a third of
   one seed's standard deviation. The table says that *these architectures don't
   matter much for this task as posed*, not that one of them wins. (The
   footnote naming SwinUNETR-3D as best and Table 3 ranking U-Net-3D first
   disagree with each other; both are within noise.)
2. **Everyone is at about 0.37 F1.** Roughly one in three predicted fire pixels
   is right. The paper's own diagnosis: segmentation networks built to *extract
   spatial features* are "poorly suited to inferring future states". Its
   feature-importance test also found that zeroing most weather and terrain
   channels barely changes F1, while removing the satellite band I2 costs 11%.
   The models are mostly reading the fire's current shape from the satellite
   bands, not the physics.
3. **More history helps only a little.** SwinUNETR-3D with T = 6, 4, 2 days
   scores F1 0.374, 0.354, 0.366. Six days beats two by 0.008 F1, about
   3 seed standard deviations: real but small, and not monotonic.

## Why a "transformer" ties with a CNN: look at the parameters

`tools/trace_shapes.py` counts SwinUNETR-3D's 33.2M parameters by part
([shapes.md](shapes.md)):

| Part | Share of parameters |
|---|---|
| Swin transformer encoder (all attention, MLPs, merging) | **10.7%** |
| Convolutional encoder blocks, mostly the bottleneck `encoder10` | **58.7%** (the bottleneck alone: 54%) |
| Convolutional decoder | 30.6% |

Compute tells the same story: the Swin stages do about 10% of the 161 GMAC per
window, and the two full-resolution convolution blocks do 46% (chapter 6).
By weight count and by work, SwinUNETR-3D is **a U-Net with a transformer
feature extractor bolted on**. Nine-tenths of its capacity is the same kind of 3 x 3 x 3
convolution U-Net-3D uses, so similar scores are no surprise. The attention
contributes content-dependent mixing at four scales, but the paths that produce
the final map (encoder1 on the raw input, the decoder, and the skips) are
convolutional.

## The wider literature agrees

On WildfireSpreadTS (Gerard et al., NeurIPS 2023 D&B), a larger next-day
benchmark at the same 375 m resolution:

- The best published model is **U-TAE** (a CNN with temporal attention),
  AP 0.372 in the original benchmark and **0.478** in the WSTS+ re-tuning
  (Lahrichi et al., WACV 2026).
- In WSTS+, **SwinUnet trailed** (about 0.45 AP), as did SegFormer: under
  controlled tuning, attention backbones did not beat a ResNet-18 U-Net.
- The biggest single gain in WSTS+ was the loss, not the architecture: removing
  focal loss cost 24% AP, and also selecting by F1 instead of AP cost 29%.

(Scores across benchmarks are not comparable. WSTS predicts tomorrow's
active-fire map, TS-SatFire only newly burned pixels, and the metrics differ.
The *direction* of the evidence is what transfers.)

## So why keep SwinUNETR-3D as the main model?

Good reasons, as long as they're stated honestly:

- **It is a strong, published, reproducible baseline** on this exact dataset,
  tied for best on prediction and best on burned-area mapping, with a released
  configuration to match.
- **Its inductive bias is the right *kind*** (full temporal context, local
  spatial context, content-dependent mixing), even if the current *amount* of
  each isn't what limits performance.
- **It is a good platform for the next experiments**: TESSERA inputs, other
  losses, larger windows and physics-aware attention biases all slot into its
  attention or its inputs (chapter 10).

Not a good reason: "transformers beat CNNs here". The evidence doesn't
support it yet. The ceiling looks set by *what the models are asked to learn
and how they're trained*, not by the backbone. Chapter 9 lists the specific
weak points; chapter 10, what to try.

Next: [9. Shortcomings](09-shortcomings.md).
