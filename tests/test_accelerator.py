"""--device: auto keeps the old CUDA-else-CPU choice; tpu goes through torch_xla."""

import sys
import types

import pytest
import torch

from firepred.training import accelerator


def test_auto_is_the_old_choice_and_cpu_needs_no_xla():
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert accelerator.resolve_device("auto") == expected
    cpu = accelerator.resolve_device("cpu")
    assert not accelerator.is_xla(cpu)
    assert accelerator.checkpoint_map_location(cpu) == cpu
    accelerator.mark_step(cpu)  # no-ops: must not import torch_xla
    accelerator.seed_device(cpu, 42)


def test_tpu_without_torch_xla_explains_itself(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_xla", None)  # makes the import fail
    with pytest.raises(RuntimeError, match="torch_xla"):
        accelerator.resolve_device("tpu")


def test_unknown_device_is_refused():
    with pytest.raises(ValueError):
        accelerator.resolve_device("gpu")


@pytest.fixture
def fake_xla(monkeypatch):
    calls = []
    xm = types.ModuleType("torch_xla.core.xla_model")
    xm.xla_device = lambda: torch.device("xla", 0)
    xm.mark_step = lambda: calls.append("mark_step")
    xm.set_rng_state = lambda seed: calls.append(("seed", seed))
    xm.save = lambda obj, path: calls.append(("save", path))
    core = types.ModuleType("torch_xla.core")
    core.xla_model = xm
    package = types.ModuleType("torch_xla")
    package.core = core
    monkeypatch.setitem(sys.modules, "torch_xla", package)
    monkeypatch.setitem(sys.modules, "torch_xla.core", core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", xm)
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    return calls


def test_tpu_path_uses_xla_calls(fake_xla, tmp_path):
    import os

    device = accelerator.resolve_device("tpu")
    assert accelerator.is_xla(device) and os.environ["PJRT_DEVICE"] == "TPU"
    accelerator.seed_device(device, 7)
    accelerator.mark_step(device)
    accelerator.save({"x": 1}, tmp_path / "c.pth", device)
    assert accelerator.checkpoint_map_location(device) == torch.device("cpu")
    assert fake_xla == [("seed", 7), "mark_step", ("save", tmp_path / "c.pth")]


def test_cpu_checkpoint_is_a_plain_torch_save(tmp_path):
    from firepred.training.checkpoints import save_checkpoint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(model, optimizer, 0, 1.5, tmp_path / "c.pth", extra={"use_tessera": False})
    loaded = torch.load(tmp_path / "c.pth", weights_only=True)
    assert loaded["loss"] == 1.5 and loaded["use_tessera"] is False
    assert torch.equal(loaded["model_state_dict"]["weight"], model.weight.detach())


def test_cli_device_flag():
    from firepred.cli.train import parse_args

    required = ["-m", "swinunetr3d", "-mode", "pred", "-b", "1", "-r", "1", "-lr", "0.001",
                "-nh", "3", "-ed", "36", "-ts", "6", "-it", "1"]
    assert parse_args(required).device == "auto"
    assert parse_args(required + ["--device", "tpu"]).device == "tpu"
