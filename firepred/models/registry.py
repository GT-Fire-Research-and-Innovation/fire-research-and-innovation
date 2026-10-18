"""Model registry: ModelSpec, model names -> builder functions, and the model I/O contract.

Level 2 (models). Used by training/run.py, cli/check.py, cli/train.py and tools/.

ModelSpec is everything a builder needs to know about the data; training
builds one from the command line and passes it to build_model(name, spec).

The contract every registered model follows:
    input   (B, C, T, H, W) float tensor, C = spec.in_channels
    output  logits (B, K, T, H, W) or (B, K, H, W), K = spec.num_classes
forward_model() averages a 5-D output over T, so a model may predict one map
per input day or a single next-day map directly. With k x k TESSERA cells per
pixel (spec.tessera_cells_per_pixel > 1) build_model puts a learned TESSERA stem
in front of the backbone (models/blocks/tessera_stem.py), and the input is the
pair (data, static) instead of one tensor.

Registering is one decorator in the model's own file:

    @register_model("mymodel")
    def build(spec):
        return MyModel(spec.in_channels, spec.num_classes, ...)

Files in firepred/models/zoo/ are imported on the first lookup, so a new file
there is all it takes for `-m mymodel` to work. Lookups construct nothing, so
they never touch the random-number streams.
"""

import argparse
import ast
import importlib
import pkgutil
from dataclasses import dataclass, field

from firepred.config.dataset import CROP_SIZE
from firepred.models import zoo


# ----- What a builder receives ----- #


@dataclass(frozen=True)
class ModelSpec:
    """Describe the data a model is built for: its input and output sizes and settings.

    Every model builder receives one ModelSpec, so models never read the command
    line themselves. Shared settings (heads, feature width) are fields; anything
    model-specific goes in options.

    Attributes:
        in_channels (int): channels per day after preprocessing (43, or 172 with TESSERA).
        num_classes (int): output channels (2: background, newly burned).
        ts_length (int): input days per window (T).
        num_heads (int): attention heads per stage; models without attention ignore it.
        feature_size (int): base feature width.
        image_size (tuple[int, int]): (H, W) of every window.
        options (dict): model-specific settings, e.g. {"attn_version": "v2"}.
        tessera_cells_per_pixel (int): TESSERA cells per VIIRS pixel side; above 1
            a TESSERA stem is built in front of the model.
    """

    in_channels: int
    num_classes: int
    ts_length: int
    num_heads: int
    feature_size: int
    image_size: tuple = (CROP_SIZE, CROP_SIZE)
    options: dict = field(default_factory=dict)
    tessera_cells_per_pixel: int = 1

    @property
    def input_size(self):
        """Give the shape of one input window without the channels.

        Builders that need the full volume size (UNETR's patch grid, SwinUNETR's
        image size) use this. It is (days, height, width).

        Returns:
            tuple[int, int, int]: (T, H, W).
        """
        return (self.ts_length, *self.image_size)


def parse_model_option(text):
    """Parse one --model-option KEY=VALUE flag.

    VALUE is read as a Python literal when it parses (3, 0.5, True, (1, 2)) and
    kept as text otherwise. argparse calls this for every --model-option.

    Args:
        text (str): e.g. "attn_version=v2".

    Returns:
        tuple[str, object]: (KEY, VALUE).

    Raises:
        argparse.ArgumentTypeError: if the text has no KEY= part.
    """
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key, ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return key, value


# ----- Registry ----- #

_BUILDERS = {}


def register_model(name, takes_tessera=False):
    """Make a model builder available under a name, as a decorator.

    Put @register_model("name") on a zoo file's build(spec) function and
    `-m name` builds it. A name can be registered by only one module. A model
    registered with takes_tessera=True reads the TESSERA map itself (one input
    tensor with the map appended, or the pair (data, static)), so build_model
    never puts the generic TESSERA stem in front of it.

    Args:
        name (str): the model name used with -m.
        takes_tessera (bool): the model handles TESSERA inputs of any cells per pixel.

    Returns:
        Callable: the decorator, which registers and returns the builder unchanged.
    """

    def decorator(builder):
        """Register one builder under the decorator's name.

        The builder is returned unchanged, so the zoo file still defines it. A
        second module registering the same name is an error.

        Args:
            builder (Callable[[ModelSpec], nn.Module]): the zoo file's build function.

        Returns:
            Callable[[ModelSpec], nn.Module]: builder.

        Raises:
            ValueError: if another module already registered the name.
        """
        existing = _BUILDERS.get(name)
        if existing is not None and existing.__module__ != builder.__module__:
            raise ValueError(f"Model name {name!r} is already registered by {existing.__module__}")
        builder.takes_tessera = takes_tessera
        _BUILDERS[name] = builder
        return builder

    return decorator


def _import_zoo():
    """Import every public module of models/zoo/, so their registrations run.

    A model is registered when its zoo file is imported. Files starting with
    "_" (the template) are skipped.

    Returns:
        list[str]: the imported module names.
    """
    names = [f"{zoo.__name__}.{module.name}" for module in pkgutil.iter_modules(zoo.__path__)
             if not module.name.startswith("_")]
    for name in names:
        importlib.import_module(name)
    return names


def available_models():
    """List every registered model name.

    The zoo is imported first, so a new file there shows up without any other
    change. These are the names -m accepts.

    Returns:
        list[str]: sorted model names.
    """
    _import_zoo()
    return sorted(_BUILDERS)


def model_takes_tessera(name):
    """Say whether a registered model reads the TESSERA map itself.

    Such a model needs --tessera and builds its own TESSERA layers. The others
    get the generic stem when the map is finer than the VIIRS grid.

    Args:
        name (str): a registered model name.

    Returns:
        bool: True for a model registered with takes_tessera=True.

    Raises:
        NotImplementedError: if no model has that name.
    """
    _import_zoo()
    if name not in _BUILDERS:
        raise NotImplementedError(f"Unsupported model: {name}")
    return getattr(_BUILDERS[name], "takes_tessera", False)


def build_model(name, spec):
    """Build a registered model for a ModelSpec, with a TESSERA stem when the spec asks.

    Looking up a name constructs nothing. Building is therefore the first step
    that draws random numbers (the initial weights): the backbone's first, then
    the stem's, so a run without a stem draws exactly what it always did.

    Args:
        name (str): a registered model name.
        spec (ModelSpec): what the model is built for.

    Returns:
        torch.nn.Module: the new model.

    Raises:
        NotImplementedError: if no model has that name.
    """
    _import_zoo()
    if name not in _BUILDERS:
        raise NotImplementedError(f"Unsupported model: {name}")
    model = _BUILDERS[name](spec)
    if spec.tessera_cells_per_pixel > 1 and not model_takes_tessera(name):
        from firepred.models.blocks.tessera_stem import TesseraStemModel

        model = TesseraStemModel(model, spec.tessera_cells_per_pixel)
    return model


def forward_model(model, batch):
    """Run a model and reduce a per-day output to one next-day map.

    Models may predict one map per input day, (B, K, T, H, W), or one map,
    (B, K, H, W). A per-day output is averaged over the days [released], so
    every model ends with the same (B, K, H, W) logits. A model with a TESSERA
    stem takes the pair (data, static).

    Args:
        model (torch.nn.Module): the model.
        batch (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): input of shape
            (B, C, T, H, W), or (data, static) for a model with a TESSERA stem.

    Returns:
        torch.Tensor: logits of shape (B, K, H, W).
    """
    outputs = model(*batch) if isinstance(batch, tuple) else model(batch)
    if outputs.ndim == 5:
        return outputs.mean(dim=2)
    return outputs
