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

from firepred.config.data import CROP_SIZE, model_input_channels  # noqa: E402
from firepred.models.blocks.window_ops import get_window_size  # noqa: E402
from firepred.models.registry import ModelSpec, build_model, forward_model  # noqa: E402

PIXEL_METERS = 375  # VIIRS I-band pixel


def shape(tensor):
    return "(" + ", ".join(str(n) for n in tensor.shape) + ")"


def parameters(module):
    return sum(p.numel() for p in module.parameters())


def count_macs(model):
    """Attach hooks that add each layer's multiply-accumulates to a counter.

    Counts Conv3d / ConvTranspose3d / Linear exactly and attention's two
    matrix products (q k^T and attn v) per window; skips norms, activations and
    elementwise ops, which are a small share.
    """
    counter = {"macs": 0}

    def conv(module, inputs, output):
        kernel = module.weight.shape[2:].numel() * (module.in_channels // module.groups)
        if isinstance(module, torch.nn.ConvTranspose3d):
            counter["macs"] += inputs[0].numel() // inputs[0].shape[1] * module.out_channels * kernel
        else:
            counter["macs"] += output.numel() * kernel

    def linear(module, inputs, output):
        counter["macs"] += output.numel() * module.in_features

    def attention(module, inputs, output):
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
    torch.manual_seed(0)
    model = build_model("swinunetr3d", spec).eval()
    counter = count_macs(model)
    rows = []
    stage_inputs = {}
    starts = {}

    def record(name, note=""):
        def hook(module, inputs, output):
            out = output[-1] if isinstance(output, (list, tuple)) else output
            first = inputs[0]
            macs = counter["macs"] - starts.pop(name)
            rows.append((name, shape(first), shape(out), parameters(module), macs, note))
        return hook

    def start(name):
        def pre_hook(module, inputs):
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
    """Per Swin stage: token grid, channels, windows, what one window sees."""
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
