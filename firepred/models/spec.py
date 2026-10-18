"""ModelSpec: everything a model builder needs to know about the data.

Level 2 (models). Built by firepred/training/run.py from the command line;
passed to firepred.models.registry.build_model(name, spec).
"""

import argparse
import ast
from dataclasses import dataclass, field

from firepred.config.dataset import CROP_SIZE


@dataclass(frozen=True)
class ModelSpec:
    """Input/output sizes plus the shared and model-specific hyperparameters.

    in_channels:  channels per day after preprocessing (43, or 172 with TESSERA)
    num_classes:  output channels (2: background, newly burned)
    ts_length:    input days per window (T)
    num_heads:    attention heads per stage (-nh); models without attention ignore it
    feature_size: base feature width (-ed)
    image_size:   (H, W) of every window
    options:      model-specific settings, e.g. {"swin_temporal_downsample": False},
                  from --model-option KEY=VALUE and the model-specific flags
    """

    in_channels: int
    num_classes: int
    ts_length: int
    num_heads: int
    feature_size: int
    image_size: tuple = (CROP_SIZE, CROP_SIZE)
    options: dict = field(default_factory=dict)

    @property
    def input_size(self):
        """(T, H, W) of one input window."""
        return (self.ts_length, *self.image_size)


def parse_model_option(text):
    """'KEY=VALUE' -> (KEY, VALUE) for ModelSpec.options; VALUE as a Python
    literal when it parses (3, 0.5, True, (1, 2)), else as text."""
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key, ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return key, value
