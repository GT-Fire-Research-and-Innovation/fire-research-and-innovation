"""Every registered model honors the I/O contract; SwinUNETR-3D keeps its checkpoint keys.

New files in firepred/models/zoo/ are picked up here automatically.
"""

import pytest
import torch
from torch import nn

from firepred.models.registry import available_models, build_model, forward_model
from firepred.models.spec import ModelSpec, parse_model_option

TINY = dict(in_channels=5, num_classes=2, ts_length=2, num_heads=3, feature_size=6, image_size=(32, 32))
# Broken since the original TS-SatFire code; see docs/KNOWN_ISSUES.md.
KNOWN_BROKEN = {"unetr3d"}


@pytest.mark.parametrize("name", available_models())
def test_model_contract(name):
    if name in KNOWN_BROKEN:
        pytest.xfail(f"{name} cannot be built (docs/KNOWN_ISSUES.md)")
    torch.manual_seed(0)
    model = build_model(name, ModelSpec(**TINY))
    x = torch.randn(2, TINY["in_channels"], TINY["ts_length"], 32, 32)
    logits = forward_model(model, x)
    assert logits.shape == (2, TINY["num_classes"], 32, 32)
    logits.mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_registry_knows_the_paper_models():
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
    spec = ModelSpec(in_channels=5, num_classes=2, ts_length=2, num_heads=3, feature_size=6, image_size=(32, 32),
                     options={"swin_temporal_downsample": True})
    model = build_model("swinunetr3d", spec)
    assert model.swinViT.layers1[0].downsample.merge_D


def test_parse_model_option():
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
