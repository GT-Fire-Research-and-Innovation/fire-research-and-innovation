"""Every registered model honors the I/O contract; SwinUNETR-3D keeps its checkpoint keys.

New files in firepred/models/zoo/ are picked up here automatically.
"""

import pytest
import torch
from torch import nn

from firepred.models.registry import (
    ModelSpec,
    available_models,
    build_model,
    forward_model,
    model_takes_tessera,
    parse_model_option,
)

TINY = dict(in_channels=5, num_classes=2, ts_length=2, num_heads=3, feature_size=6, image_size=(32, 32))
# Broken since the original TS-SatFire code; see docs/KNOWN_ISSUES.md.
KNOWN_BROKEN = set()


@pytest.mark.parametrize("name", available_models())
def test_model_contract(name):
    """Every registered model maps (B, C, T, H, W) to (B, 2, H, W) logits after forward_model and trains one step."""
    if name in KNOWN_BROKEN:
        pytest.xfail(f"{name} cannot be built (docs/KNOWN_ISSUES.md)")
    torch.manual_seed(0)
    # A model that reads TESSERA itself gets the map appended to every day, as batch_inputs does.
    spec = dict(TINY, in_channels=TINY["in_channels"] + 129) if model_takes_tessera(name) else TINY
    model = build_model(name, ModelSpec(**spec))
    x = torch.randn(2, spec["in_channels"], TINY["ts_length"], 32, 32)
    logits = forward_model(model, x)
    assert logits.shape == (2, TINY["num_classes"], 32, 32)
    logits.mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_registry_knows_the_paper_models():
    """The paper's models are registered and unknown names are refused."""
    assert {"swinunetr3d", "unet3d", "attunet", "unetr3d"} <= set(available_models())
    with pytest.raises(NotImplementedError):
        build_model("no_such_model", ModelSpec(**TINY))


def test_swinunetr3d_checkpoint_keys_are_stable():
    """Old checkpoints load only if these keys never change."""
    spec = ModelSpec(in_channels=43, num_classes=2, ts_length=6, num_heads=3, feature_size=36)
    keys = list(nn.DataParallel(build_model("swinunetr3d", spec)).state_dict())
    assert len(keys) == 289
    assert keys[0] == "module.swinViT.patch_embed.proj.weight"
    assert "module.swinViT.layers1.0.blocks.0.attn.relative_position_bias_table" in keys
    assert keys[-1] == "module.out.0.conv.conv.bias"


def test_swin_temporal_downsample_option():
    """--swin-temporal-downsample turns on time merging in the first patch-merging layer."""
    spec = ModelSpec(in_channels=5, num_classes=2, ts_length=2, num_heads=3, feature_size=6, image_size=(32, 32),
                     options={"swin_temporal_downsample": True})
    model = build_model("swinunetr3d", spec)
    assert model.swinViT.layers1[0].downsample.merge_D


def test_parse_model_option():
    """KEY=VALUE options parse VALUE as a Python literal when possible, else as text."""
    assert parse_model_option("width=32") == ("width", 32)
    assert parse_model_option("kind=conv") == ("kind", "conv")
    assert parse_model_option("sizes=(1, 2)") == ("sizes", (1, 2))


def test_tpool_starts_as_the_plain_mean_over_days():
    """swinunetr3d_tpool's documented promise: at initialization it equals mean-over-T."""
    torch.manual_seed(0)
    model = build_model("swinunetr3d_tpool", ModelSpec(**TINY)).eval()
    x = torch.randn(1, TINY["in_channels"], TINY["ts_length"], 32, 32)
    with torch.no_grad():
        pooled = forward_model(model, x)
        plain_mean = model.backbone(x).mean(dim=2)
    assert torch.allclose(pooled, plain_mean, atol=1e-6)


def test_swin_v2_attention_option():
    """--model-option attn_version=v2 builds SwinUNETR-3D with Swin V2 attention; v1 stays the default."""
    def spec(**options):
        """A small SwinUNETR-3D spec with the given options."""
        return ModelSpec(in_channels=5, num_classes=2, ts_length=2, num_heads=3, feature_size=6, image_size=(32, 32),
                         options=options)

    v2 = build_model("swinunetr3d", spec(attn_version="v2"))
    assert type(v2.swinViT.layers1[0].blocks[0].attn).__name__ == "WindowAttentionV2"
    assert type(build_model("swinunetr3d", spec()).swinViT.layers1[0].blocks[0].attn).__name__ == "WindowAttentionV1"
    out = forward_model(v2, torch.randn(1, 5, 2, 32, 32))
    assert out.shape == (1, 2, 32, 32)
    with pytest.raises(ValueError, match="v1 or v2"):
        build_model("swinunetr3d", spec(attn_version="ar"))


def test_tessera_context_model_starts_as_plain_swinunetr3d():
    """swinunetr3d_tessera with zero FiLM gives exactly a 43-channel SwinUNETR-3D's output, for any k."""
    from firepred.models.zoo.swinunetr3d_tessera import TesseraContextSwinUNETR

    base = dict(num_classes=2, ts_length=2, num_heads=3, feature_size=6, image_size=(32, 32))
    torch.manual_seed(0)
    plain = build_model("swinunetr3d", ModelSpec(in_channels=5, **base)).eval()
    data = torch.randn(1, 5, 2, 32, 32)
    for cells in (1, 2, 5):
        model = build_model("swinunetr3d_tessera", ModelSpec(in_channels=5 + 129, tessera_cells_per_pixel=cells, **base))
        assert isinstance(model, TesseraContextSwinUNETR)  # no generic stem around it
        missing, unexpected = model.load_state_dict(plain.state_dict(), strict=False)
        assert not unexpected and all(key.startswith(("context.", "film.")) for key in missing)
        model.eval()
        static = torch.randn(1, 129, 32 * cells, 32 * cells)
        with torch.no_grad():
            expected = plain(data)
            if cells == 1:  # one tensor, map copied onto every day (batch_inputs)
                got = model(torch.cat((data, static.unsqueeze(2).expand(-1, -1, 2, -1, -1)), dim=1))
            else:
                got = model(data, static.half())
        assert torch.allclose(got, expected, atol=1e-6)


def test_tessera_context_learns_from_the_map():
    """After one Adam step the FiLM layers move, so the prediction starts to depend on TESSERA."""
    base = dict(num_classes=2, ts_length=2, num_heads=3, feature_size=6, image_size=(32, 32))
    torch.manual_seed(1)
    model = build_model("swinunetr3d_tessera", ModelSpec(in_channels=134, tessera_cells_per_pixel=4,
                                                         options={"context_channels": 8}, **base))
    data, static = torch.randn(2, 5, 2, 32, 32), torch.randn(2, 129, 128, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    forward_model(model, (data, static)).mean().backward()
    assert model.context.patchify[0].weight.grad is None or not model.context.patchify[0].weight.grad.any()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        a = forward_model(model, (data, static))
        b = forward_model(model, (data, torch.zeros_like(static)))
    assert not torch.allclose(a, b)
    assert len(model.context.reduce) == 1  # k = 4: two halvings


def test_tessera_context_model_needs_tessera():
    """swinunetr3d_tessera refuses a spec without TESSERA channels."""
    with pytest.raises(ValueError, match="--tessera"):
        build_model("swinunetr3d_tessera", ModelSpec(in_channels=43, num_classes=2, ts_length=2, num_heads=3,
                                                     feature_size=6, image_size=(32, 32)))
