"""swinunetr3d_tessera: SwinUNETR-3D that reads TESSERA as multi-scale context. [ours]

Level 2 (models/zoo). Train it with `-m swinunetr3d_tessera --tessera` and any
--tessera-cells-per-pixel k; every SwinUNETR-3D flag applies.

Plain SwinUNETR-3D gets TESSERA as 129 extra input channels, copied onto every
day. This variant keeps TESSERA out of the Swin input and instead:

    TESSERA (B, 129, 256 k, 256 k)
      -> TesseraContextEncoder: learned conv steps shrink it by k to 256 x 256,
         then a pyramid 128, 64, 32, 16, 8           (blocks/tessera_context.py)
    VIIRS days (B, 43, T, 256, 256)
      -> SwinUNETR-3D encoder; every skip connection and the bottleneck is
         conditioned on the context map of its size: x (1 + scale) + shift

The FiLM projections start at zero, so the untrained model is exactly a
43-channel SwinUNETR-3D and learns how much TESSERA context to use. Its
checkpoint keys are SwinUNETR-3D's (43 input channels) plus "context." and
"film." parameters, so SwinUNETR-3D weights trained without TESSERA load with
strict=False as a starting point.
"""

from torch import nn

from firepred.config.dataset import TESSERA_CHANNELS
from firepred.models.blocks.tessera_context import ContextFiLM, TesseraContextEncoder
from firepred.models.registry import register_model
from firepred.models.zoo.swinunetr3d import SwinUNETR, swinunetr_settings

# Context maps: the VIIRS grid (encoder1) and the five Swin scales.
CONTEXT_LEVELS = 6
DEFAULT_CONTEXT_CHANNELS = 32


class TesseraContextSwinUNETR(SwinUNETR):
    """SwinUNETR-3D whose skip connections are conditioned on learned TESSERA context.

    The TESSERA map has its own small network, which learns features while
    shrinking it to the VIIRS grid and then to every coarser scale. Each
    encoder output is scaled and shifted by the context of its size before the
    decoder uses it, so TESSERA can say "this pixel is forest" at every scale.

    Attributes:
        data_channels (int): VIIRS channels per day (43).
        context (TesseraContextEncoder): the TESSERA network.
        film (torch.nn.ModuleList): one ContextFiLM per conditioned scale, finest first.
    """

    def __init__(self, data_channels, cells_per_pixel, context_channels=DEFAULT_CONTEXT_CHANNELS, **settings):
        """Build SwinUNETR-3D on the VIIRS channels, then the TESSERA network and the FiLM layers.

        The SwinUNETR-3D layers are created first, in their usual order. The
        conditioned widths are F, F, 2F, 4F, 8F and 16F (F = feature_size), for
        the scales 256 down to 8.

        Args:
            data_channels (int): VIIRS channels per day (43).
            cells_per_pixel (int): TESSERA cells per VIIRS pixel side.
            context_channels (int): channels of every context map.
            **settings: SwinUNETR keyword arguments (swinunetr_settings).
        """
        super().__init__(in_channels=data_channels, **settings)
        width = settings["feature_size"]
        self.data_channels = data_channels
        self.context = TesseraContextEncoder(cells_per_pixel, CONTEXT_LEVELS, context_channels)
        self.film = nn.ModuleList(
            ContextFiLM(context_channels, channels)
            for channels in (width, width, 2 * width, 4 * width, 8 * width, 16 * width)
        )

    def forward(self, x_in, static=None):
        """Run SwinUNETR-3D on the VIIRS days, conditioning each scale on TESSERA context.

        With one cell per pixel the batch arrives as one tensor with the map
        appended to every day (batch_inputs); it is split off again here. A finer
        map arrives separately as static.

        Args:
            x_in (torch.Tensor): (B, 43, T, H, W), or (B, 172, T, H, W) when static is None.
            static (torch.Tensor | None): (B, 129, k H, k W), or None.

        Returns:
            torch.Tensor: per-day logits of shape (B, K, T, H, W).
        """
        if static is None:
            # The map was copied onto every day; one day's copy is the whole map.
            x_in, static = x_in[:, :self.data_channels], x_in[:, self.data_channels:, 0]
        context = self.context(static)                                  # 6 maps: (B, Cc, 256/2^i, 256/2^i)
        hidden_states_out = self.swinViT(x_in, self.normalize)          # 5 scales: (B, F*2^i, T, 128/2^i, ...)
        enc0 = self.film[0](self.encoder1(x_in), context[0])            # (B, F,   T, 256, 256)
        enc1 = self.film[1](self.encoder2(hidden_states_out[0]), context[1])  # (B, F, T, 128, 128)
        enc2 = self.film[2](self.encoder3(hidden_states_out[1]), context[2])  # (B, 2F, T, 64, 64)
        enc3 = self.film[3](self.encoder4(hidden_states_out[2]), context[3])  # (B, 4F, T, 32, 32)
        skip4 = self.film[4](hidden_states_out[3], context[4])          # (B, 8F,  T,  16,  16)
        dec4 = self.film[5](self.encoder10(hidden_states_out[4]), context[5])  # (B, 16F, T, 8, 8)
        dec3 = self.decoder5(dec4, skip4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        return self.out(out)                                            # (B, 2, T, 256, 256)


@register_model("swinunetr3d_tessera", takes_tessera=True)
def build(spec):
    """Build SwinUNETR-3D with TESSERA context for a ModelSpec.

    The spec's input channels must include TESSERA (172), because the model
    splits the 129 TESSERA channels off. --model-option context_channels=N sets
    the context width (default 32).

    Args:
        spec (ModelSpec): what the model is built for.

    Returns:
        TesseraContextSwinUNETR: the new model.

    Raises:
        ValueError: if the spec has no TESSERA channels.
    """
    data_channels = spec.in_channels - TESSERA_CHANNELS
    if data_channels <= 0:
        raise ValueError(f"swinunetr3d_tessera needs TESSERA inputs (train with --tessera); "
                         f"got {spec.in_channels} input channels")
    return TesseraContextSwinUNETR(
        data_channels,
        spec.tessera_cells_per_pixel,
        context_channels=spec.options.get("context_channels", DEFAULT_CONTEXT_CHANNELS),
        **swinunetr_settings(spec),
    )
