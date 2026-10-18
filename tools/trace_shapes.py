"""Trace real tensor shapes through SwinUNETR-3D and write docs/swinunetr3d/shapes.md.

    python tools/trace_shapes.py                  # paper setup: 43 channels, 6 days, 256x256, F=36, 3 heads
    python tools/trace_shapes.py --tessera        # 172 input channels
    python tools/trace_shapes.py --size 64 --out -   # small, printed instead of written

Every number in the explainer's shape tables comes from this script, which
runs the actual model with forward hooks, so the docs cannot drift from the code.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firepred.config.dataset import CROP_SIZE, model_input_channels  # noqa: E402
from firepred.models.blocks.window_ops import get_window_size  # noqa: E402
from firepred.models.registry import ModelSpec, build_model, forward_model  # noqa: E402

PIXEL_METERS = 375  # VIIRS I-band pixel


def shape(tensor):
    """Write a tensor's shape as text.

    The Markdown tables show shapes like "(1, 43, 6, 256, 256)". Batch comes first.

    Args:
        tensor (torch.Tensor): the tensor.

    Returns:
        str: the shape.
    """
    return "(" + ", ".join(str(n) for n in tensor.shape) + ")"


def parameters(module):
    """Count a module's parameters.

    Every weight and bias counts, trainable or not. The table reports them per part.

    Args:
        module (torch.nn.Module): the module.

    Returns:
        int: the parameter count.
    """
    return sum(p.numel() for p in module.parameters())


def count_macs(model):
    """Attach hooks that add up the model's multiply-accumulate operations (MACs).

    Convolutions and linear layers are counted exactly, and attention's two
    matrix products per window. Norms, activations and additions are skipped,
    so the total is a lower bound.

    Args:
        model (torch.nn.Module): the model.

    Returns:
        dict: {"macs": running total}, updated as the model runs.
    """
    counter = {"macs": 0}

    def conv(module, inputs, output):
        """Add one (transposed) 3-D convolution's MACs to the counter.

        It runs after the layer's forward pass. A transposed convolution is counted
        from its input size.

        Args:
            module (torch.nn.Module): the convolution.
            inputs (tuple[torch.Tensor, ...]): its inputs.
            output (torch.Tensor): its output.
        """
        kernel = module.weight.shape[2:].numel() * (module.in_channels // module.groups)
        if isinstance(module, torch.nn.ConvTranspose3d):
            counter["macs"] += inputs[0].numel() // inputs[0].shape[1] * module.out_channels * kernel
        else:
            counter["macs"] += output.numel() * kernel

    def linear(module, inputs, output):
        """Add one linear layer's MACs to the counter.

        It runs after the layer's forward pass. Each output value costs one MAC per
        input feature.

        Args:
            module (torch.nn.Module): the linear layer.
            inputs (tuple[torch.Tensor, ...]): its inputs.
            output (torch.Tensor): its output.
        """
        counter["macs"] += output.numel() * module.in_features

    def attention(module, inputs, output):
        """Add one window attention's two matrix products to the counter.

        It runs after the attention's forward pass. Scores and the value mix each
        cost tokens x tokens x channels per window.

        Args:
            module (torch.nn.Module): the attention module.
            inputs (tuple[torch.Tensor, ...]): its inputs.
            output (torch.Tensor): its output.
        """
        windows, tokens, channels = inputs[0].shape
        counter["macs"] += 2 * windows * tokens * tokens * channels

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
            module.register_forward_hook(conv)
        elif isinstance(module, torch.nn.Linear):
            module.register_forward_hook(linear)
        elif type(module).__name__ == "WindowAttentionV1":
            module.register_forward_hook(attention)
    return counter


def trace(spec, batch=1):
    """Run one seeded forward pass of SwinUNETR-3D and record every traced module.

    Hooks record each part's input and output shape, parameters and MACs. The
    model is built with seed 0 and run in eval mode.

    Args:
        spec (ModelSpec): what the model is built for.
        batch (int): the batch size of the random input.

    Returns:
        tuple: (model, rows, stage inputs, input shape, output shape, total MACs);
        each row is (name, input shape, output shape, parameters, MACs, note).
    """
    torch.manual_seed(0)
    model = build_model("swinunetr3d", spec).eval()
    counter = count_macs(model)
    rows = []
    stage_inputs = {}
    starts = {}

    def record(name, note=""):
        """Make a forward hook that records one module's row.

        The row holds the module's shapes, parameters and the MACs counted since
        its pre-hook ran. Rows are kept in the order modules finish.

        Args:
            name (str): the module's name in the table.
            note (str): a note for the table.

        Returns:
            Callable: the hook.
        """
        def hook(module, inputs, output):
            """Record this module's row after it runs.

            The MACs are the counter's growth since the matching pre-hook. A tuple
            output is represented by its last element.

            Args:
                module (torch.nn.Module): the module.
                inputs (tuple[torch.Tensor, ...]): its inputs.
                output (torch.Tensor | tuple): its output.
            """
            out = output[-1] if isinstance(output, (list, tuple)) else output
            first = inputs[0]
            macs = counter["macs"] - starts.pop(name)
            rows.append((name, shape(first), shape(out), parameters(module), macs, note))
        return hook

    def start(name):
        """Make a forward pre-hook that remembers the MAC counter before a module runs.

        The matching hook subtracts it to get the module's own MACs. Pre-hooks
        run before the module's forward pass.

        Args:
            name (str): the module's name in the table.

        Returns:
            Callable: the pre-hook.
        """
        def pre_hook(module, inputs):
            """Remember the MAC counter as this module starts.

            It runs before the module's forward pass. Nothing else is changed.

            Args:
                module (torch.nn.Module): the module.
                inputs (tuple[torch.Tensor, ...]): its inputs.
            """
            starts[name] = counter["macs"]
        return pre_hook

    encoder = model.swinViT
    hooks = [encoder.patch_embed.register_forward_pre_hook(start("swinViT.patch_embed")),
             encoder.patch_embed.register_forward_hook(record("swinViT.patch_embed", "Conv3d, kernel = stride = (1, 2, 2)"))]
    for index in range(1, 5):
        layer = getattr(encoder, f"layers{index}")[0]
        hooks.append(layer.register_forward_pre_hook(
            lambda module, inputs, index=index: stage_inputs.__setitem__(index, tuple(inputs[0].shape))))
        hooks.append(layer.register_forward_pre_hook(start(f"swinViT.layers{index}")))
        hooks.append(layer.register_forward_hook(record(
            f"swinViT.layers{index}", f"{len(layer.blocks)} Swin blocks, then PatchMerging")))
    for name in ("encoder1", "encoder2", "encoder3", "encoder4", "encoder10",
                 "decoder5", "decoder4", "decoder3", "decoder2", "decoder1", "out"):
        hooks.append(getattr(model, name).register_forward_pre_hook(start(name)))
        hooks.append(getattr(model, name).register_forward_hook(record(name)))

    x = torch.randn(batch, spec.in_channels, *spec.input_size)
    with torch.no_grad():
        logits = forward_model(model, x)
    for hook in hooks:
        hook.remove()
    return model, rows, stage_inputs, tuple(x.shape), tuple(logits.shape), counter["macs"]


def stage_table(model, spec, stage_inputs):
    """Describe each Swin stage: token grid, channels, windows and what one window covers.

    The ground a window covers shows how far attention can reach at each scale.
    Pixels are 375 m.

    Args:
        model (torch.nn.Module): the traced model.
        spec (ModelSpec): what it was built for.
        stage_inputs (dict): each stage's input shape.

    Returns:
        list[tuple]: one row per stage.
    """
    lines = [
        "| Stage | Token grid (T, H, W) | Channels | Window (tokens) | Windows per sample | "
        "Token = pixels | One window covers | Heads x head dim |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for index in range(1, 5):
        _, channels, t, h, w = stage_inputs[index]
        window, _ = get_window_size((t, h, w), model.swinViT.window_size, tuple(s // 2 for s in model.swinViT.window_size))
        count = (-(-t // window[0])) * (-(-h // window[1])) * (-(-w // window[2]))
        token_pixels = spec.image_size[0] // h
        cover = window[1] * token_pixels
        km = cover * PIXEL_METERS / 1000
        heads = spec.num_heads
        lines.append(
            f"| {index} | ({t}, {h}, {w}) | {channels} | {window[0]}x{window[1]}x{window[2]} = "
            f"{window[0] * window[1] * window[2]} | {count} | {token_pixels}x{token_pixels} | "
            f"{window[0]} days x {cover}x{cover} px ({km:g} km) | {heads} x {channels // heads} |"
        )
    return "\n".join(lines)


def render(spec, model, rows, stage_inputs, input_shape, output_shape, total_macs):
    """Write the shapes.md Markdown from a trace.

    It lists every traced module in order, compute by part, the Swin stages and
    parameters by part. The explainer's shape tables come from it.

    Args:
        spec (ModelSpec): what the model was built for.
        model (torch.nn.Module): the traced model.
        rows (list[tuple]): the traced modules.
        stage_inputs (dict): each stage's input shape.
        input_shape (tuple[int, ...]): the input's shape.
        output_shape (tuple[int, ...]): the output's shape.
        total_macs (int): all counted MACs.

    Returns:
        str: the Markdown.
    """
    total = parameters(model)
    groups = {
        "Swin encoder (swinViT)": parameters(model.swinViT),
        "Convolutional encoder blocks (encoder1-4, encoder10)": sum(
            parameters(getattr(model, n)) for n in ("encoder1", "encoder2", "encoder3", "encoder4", "encoder10")),
        "Decoder (decoder5-1)": sum(parameters(getattr(model, n)) for n in ("decoder5", "decoder4", "decoder3", "decoder2", "decoder1")),
        "Output head (out)": parameters(model.out),
    }
    lines = [
        "# SwinUNETR-3D tensor shapes",
        "",
        "_Generated by `python tools/trace_shapes.py`; do not edit by hand._",
        "",
        f"Setup: in_channels={spec.in_channels}, T={spec.ts_length} days, "
        f"{spec.image_size[0]}x{spec.image_size[1]} pixels, feature_size F={spec.feature_size}, "
        f"{spec.num_heads} heads per stage, patch (1, 2, 2), window ({spec.ts_length}, 4, 4), "
        f"temporal downsampling {spec.options.get('swin_temporal_downsample', False)}. "
        "Shapes are (batch, channels, T, H, W).",
        "",
        f"Input {input_shape} -> model logits (B, 2, T, H, W) -> mean over T -> {output_shape}.",
        "",
        "## Every module, in execution order",
        "",
        "| Module | Input | Output | Parameters | Compute (GMAC) | Share of compute | Note |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, inp, out, params, macs, note in rows:
        lines.append(f"| `{name}` | {inp} | {out} | {params:,} | {macs / 1e9:.2f} | "
                     f"{100 * macs / total_macs:.1f}% | {note} |")
    lines += [
        "",
        f"Compute for one window (B = 1): {total_macs / 1e9:.1f} GMAC "
        "(multiply-accumulates in convolutions, linear layers and attention's two",
        "matrix products; a training step costs roughly 3x this per window).",
    ]
    compute = {}
    for name, _, _, _, macs, _ in rows:
        part = ("Swin encoder (swinViT)" if name.startswith("swinViT") else
                "Convolutional encoder blocks (encoder1-4, encoder10)" if name.startswith("encoder") else
                "Decoder (decoder5-1)" if name.startswith("decoder") else "Output head (out)")
        compute[part] = compute.get(part, 0) + macs
    full_resolution = sum(macs for name, *_, macs, _ in rows if name in ("encoder1", "decoder1"))
    lines += [
        "",
        "## Compute by part",
        "",
        "| Part | Compute (GMAC) | Share |",
        "|---|---|---|",
    ]
    for part, macs in compute.items():
        lines.append(f"| {part} | {macs / 1e9:.1f} | {100 * macs / total_macs:.1f}% |")
    lines += [
        "",
        f"The two full-resolution blocks, `encoder1` and `decoder1`, do "
        f"{100 * full_resolution / total_macs:.0f}% of the work.",
    ]
    lines += [
        "",
        "## Swin stages: what attention sees",
        "",
        f"A VIIRS pixel is {PIXEL_METERS} m. Attention runs only inside a window; every second",
        "block shifts the windows by half a window so neighbouring windows exchange information.",
        "",
        stage_table(model, spec, stage_inputs),
        "",
        "## Parameters",
        "",
        "| Part | Parameters | Share |",
        "|---|---|---|",
    ]
    for name, count in groups.items():
        lines.append(f"| {name} | {count:,} | {100 * count / total:.1f}% |")
    lines += [f"| **Total** | **{total:,}** | 100% |", ""]
    return "\n".join(lines)


def main(argv=None):
    """Trace SwinUNETR-3D with real shapes and write (or print) the Markdown.

    The defaults are the paper's: 6 days, 256 x 256, feature size 36, 3 heads.
    With --out - the Markdown is printed instead of written.

    Args:
        argv (list[str] | None): the arguments (default: sys.argv).

    Returns:
        str: the Markdown.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tessera", action="store_true", help="172 input channels instead of 43")
    parser.add_argument("--ts", type=int, default=6)
    parser.add_argument("--size", type=int, default=CROP_SIZE)
    parser.add_argument("--feature-size", type=int, default=36)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "swinunetr3d" / "shapes.md"),
                        help="Markdown file to write, or - for stdout")
    args = parser.parse_args(argv)

    spec = ModelSpec(
        in_channels=model_input_channels(args.tessera),
        num_classes=2,
        ts_length=args.ts,
        num_heads=args.heads,
        feature_size=args.feature_size,
        image_size=(args.size, args.size),
    )
    model, rows, stage_inputs, input_shape, output_shape, total_macs = trace(spec)
    text = render(spec, model, rows, stage_inputs, input_shape, output_shape, total_macs)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")
    return text


if __name__ == "__main__":
    main()
