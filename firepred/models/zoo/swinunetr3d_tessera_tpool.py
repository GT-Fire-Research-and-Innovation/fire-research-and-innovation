"""swinunetr3d_tessera_tpool: TESSERA-context SwinUNETR-3D with learned day weights. [ours]

Level 2 (models/zoo). Train it with `-m swinunetr3d_tessera_tpool --tessera` and
any --tessera-cells-per-pixel k; every SwinUNETR-3D flag and the
context_channels model option apply.

It combines the two variants, each unchanged:

    swinunetr3d_tessera   TESSERA encoded by its own CNN and FiLM-conditioning
                          every scale (blocks/tessera_context.py)
    swinunetr3d_tpool     the per-day maps combined with softmax(day_logits)
                          instead of an equal average

Both start as the plain model (zero FiLM, equal day weights), so the untrained
model gives exactly a 43-channel SwinUNETR-3D's day-averaged prediction. Its
checkpoint keys are swinunetr3d_tessera's with a "backbone." prefix, plus
"day_logits".
"""

from firepred.models.registry import register_model
from firepred.models.zoo.swinunetr3d_tessera import build as build_swinunetr3d_tessera
from firepred.models.zoo.swinunetr3d_tpool import TemporalPooledSwinUNETR


@register_model("swinunetr3d_tessera_tpool", takes_tessera=True)
def build(spec):
    """Build TESSERA-context SwinUNETR-3D with one learned weight per input day.

    The backbone is exactly swinunetr3d_tessera's, so it needs TESSERA inputs.
    The day weights wrap it as in swinunetr3d_tpool, and the TESSERA map is
    passed through to the backbone.

    Args:
        spec (ModelSpec): what the model is built for.

    Returns:
        TemporalPooledSwinUNETR: the new model.

    Raises:
        ValueError: if the spec has no TESSERA channels.
    """
    return TemporalPooledSwinUNETR(build_swinunetr3d_tessera(spec), spec.ts_length)
