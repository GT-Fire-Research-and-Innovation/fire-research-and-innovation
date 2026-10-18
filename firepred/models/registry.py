"""Model registry: model names -> builder functions, and the model I/O contract.

Level 2 (models). Used by training/run.py, cli/check.py and tools/.

The contract every registered model follows:
    input   (B, C, T, H, W) float tensor, C = spec.in_channels
    output  logits (B, K, T, H, W) or (B, K, H, W), K = spec.num_classes
forward_model() averages a 5-D output over T, so a model may predict one map
per input day or a single next-day map directly.

Registering is one decorator in the model's own file:

    @register_model("mymodel")
    def build(spec):
        return MyModel(spec.in_channels, spec.num_classes, ...)

Files in firepred/models/zoo/ are imported on the first lookup, so a new file
there is all it takes for `-m mymodel` to work. Lookups construct nothing, so
they never touch the random-number streams.
"""

import importlib
import pkgutil

from firepred.models import zoo

_BUILDERS = {}


def register_model(name):
    """Decorator: make `builder(spec) -> nn.Module` available as `name`."""

    def decorator(builder):
        existing = _BUILDERS.get(name)
        if existing is not None and existing.__module__ != builder.__module__:
            raise ValueError(f"Model name {name!r} is already registered by {existing.__module__}")
        _BUILDERS[name] = builder
        return builder

    return decorator


def _import_zoo():
    for module in pkgutil.iter_modules(zoo.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{zoo.__name__}.{module.name}")


def available_models():
    """Sorted names accepted by build_model (and by `-m`)."""
    _import_zoo()
    return sorted(_BUILDERS)


def build_model(name, spec):
    """Build the registered model `name` for `spec` (a ModelSpec)."""
    _import_zoo()
    if name not in _BUILDERS:
        raise NotImplementedError(f"Unsupported model: {name}")
    return _BUILDERS[name](spec)


def forward_model(model, batch):
    """Run the model and reduce a per-day output to one next-day map.

    (B, K, T, H, W) -> mean over T -> (B, K, H, W). [released]
    """
    outputs = model(batch)
    if outputs.ndim == 5:
        return outputs.mean(dim=2)
    return outputs
