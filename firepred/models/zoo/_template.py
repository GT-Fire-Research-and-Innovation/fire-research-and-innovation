"""TEMPLATE: start a new model with `python -m firepred.cli.new_model NAME`.

(Files whose name starts with "_" are never registered, so this one is inert.)

A model is one file in firepred/models/zoo/ holding three things, top to bottom:
architecture constants, the nn.Module, and a builder decorated with
@register_model. Dropping the file in zoo/ is all it takes for `-m NAME` to
work in the trainer, the checks, W&B run names and checkpoint names.

Checklist
  1. forward(x): x is (B, C, T, H, W); return logits, no sigmoid:
     (B, K, T, H, W) (one map per input day; the trainer averages over T) or
     (B, K, H, W) (one next-day map).
  2. Create layers in __init__ in the order forward uses them: creation order
     is the order the random initial weights are drawn.
  3. Keep this model's own constants here; read data sizes from `spec`
     (spec.in_channels, spec.num_classes, spec.ts_length, spec.image_size,
     spec.num_heads, spec.feature_size).
  4. New hyperparameters: spec.options.get("name", DEFAULT). Set them with
     `--model-option name=value` (notebook: MODEL_OPTIONS = {"name": value}).
  5. Check: python -m firepred.cli.check model -m NAME
     Train: python -m firepred.cli.train -m NAME ...

Bricks to build with:
  firepred/models/blocks/  window_ops, window_attention (Swin V1), patch_merging,
                           swin_block (one Swin stage), swin_encoder (full Swin encoder)
  monai.networks.blocks    UnetrBasicBlock, UnetrUpBlock, UnetOutBlock, PatchEmbed,
                           Convolution, ResidualUnit, ...
  torch.nn                 nn.Sequential for plain stacks
"""

from torch import nn
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

from firepred.models.registry import register_model

NAME = "__MODEL_NAME__"

# Architecture constants of this model.
WIDTH = 16  # feature channels at full resolution; override with --model-option width=32


class Model(nn.Module):
    """A two-level 3-D U-Net, the smallest useful example of a zoo model.

    One block extracts features, one halves the image (never the days), one
    upsamples and joins the skip connection, and a 1x1x1 convolution gives the
    logits. Copy it with `python -m firepred.cli.new_model NAME` and edit.
    """

    def __init__(self, in_channels, num_classes, width):
        """Build the four blocks of the template model.

        All blocks come from MONAI. Widths are width and 2 x width.

        Args:
            in_channels (int): input channels per day.
            num_classes (int): output channels.
            width (int): base feature width.
        """
        super().__init__()
        self.stem = UnetrBasicBlock(3, in_channels, width, kernel_size=3, stride=1,
                                    norm_name="batch", res_block=True)
        self.down = UnetrBasicBlock(3, width, 2 * width, kernel_size=3, stride=(1, 2, 2),
                                    norm_name="batch", res_block=True)
        self.up = UnetrUpBlock(3, 2 * width, width, kernel_size=3, upsample_kernel_size=(1, 2, 2),
                               norm_name="batch", res_block=True)
        self.head = UnetOutBlock(3, width, num_classes)

    def forward(self, x):                 # (B, C,  T, H,   W)
        """Run the template model on one batch.

        The days are kept throughout, so the output is one map per day.
        The trainer averages them into one next-day map.

        Args:
            x (torch.Tensor): input of shape (B, C, T, H, W).

        Returns:
            torch.Tensor: per-day logits of shape (B, K, T, H, W).
        """
        skip = self.stem(x)               # (B, W,  T, H,   W)    3x3x3 convs, same size
        x = self.down(skip)               # (B, 2W, T, H/2, W/2)  halve space, keep the days
        x = self.up(x, skip)              # (B, W,  T, H,   W)    upsample, concat skip, convs
        return self.head(x)               # (B, K,  T, H,   W)    logits; the trainer averages T


@register_model(NAME)
def build(spec):
    """Build the template model for a ModelSpec.

    Only in_channels and num_classes are used. options["width"] sets the
    feature width.

    Args:
        spec (ModelSpec): what the model is built for.

    Returns:
        Model: the new model.
    """
    return Model(spec.in_channels, spec.num_classes, spec.options.get("width", WIDTH))
