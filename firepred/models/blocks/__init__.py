"""Reusable model layers. Each module documents its input and output shapes.

    window_ops.py           cut a volume into attention windows and back
    window_attention.py     Swin V1 window attention (relative position bias)
    window_attention_v2.py  Swin V2 window attention (scaled cosine, continuous bias)
    swin_block.py           one Swin stage of shifted-window blocks
    patch_merging.py        halve the grid and double the channels between stages
    swin_encoder.py         patch embedding + four Swin stages -> five feature scales
    tessera_stem.py         learned k x k stem: a fine TESSERA map -> the VIIRS grid [ours]
    tessera_context.py      TESSERA context CNN and FiLM conditioning [ours]
"""
