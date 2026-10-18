"""--device: auto keeps the old CUDA-else-CPU choice; tpu goes through torch_xla."""

import sys
import types

import pytest
import torch

from firepred.training.accelerator import Device


def test_auto_is_the_old_choice_and_cpu_needs_no_xla():
    """--device auto picks CUDA when visible, else the CPU, and CPU runs never import torch_xla."""
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert Device.resolve("auto").torch_device == expected
    cpu = Device.resolve("cpu")
    assert not cpu.is_xla and Device.of(cpu) is cpu and Device.of("cpu") == cpu
    assert cpu.map_location == torch.device("cpu")
    cpu.mark_step()  # no-ops: must not import torch_xla
    cpu.seed(42)


def test_tpu_without_torch_xla_explains_itself(monkeypatch):
    """--device tpu without torch_xla fails with a message that names the missing package."""
    monkeypatch.setitem(sys.modules, "torch_xla", None)  # makes the import fail
    with pytest.raises(RuntimeError, match="torch_xla"):
        Device.resolve("tpu")


def test_unknown_device_is_refused():
    """Device names outside DEVICES are refused."""
    with pytest.raises(ValueError):
        Device.resolve("gpu")


@pytest.fixture
def fake_xla(monkeypatch):
    """Fake torch_xla modules that record mark_step, seeding and saving."""
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
    """On a TPU, seeding, mark_step and saving go through torch_xla."""
    import os

    device = Device.resolve("tpu")
    assert device.is_xla and os.environ["PJRT_DEVICE"] == "TPU"
    device.seed(7)
    device.mark_step()
    device.save({"x": 1}, tmp_path / "c.pth")
    assert device.map_location == torch.device("cpu")
    assert fake_xla == [("seed", 7), "mark_step", ("save", tmp_path / "c.pth")]


def test_cpu_checkpoint_is_a_plain_torch_save(tmp_path):
    """On the CPU a checkpoint is a plain torch.save that loads with weights_only=True."""
    from firepred.training.checkpoints import CheckpointStore

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    CheckpointStore(tmp_path, "c.pth", {"use_tessera": False}).save(tmp_path / "c.pth", 0, 1.5, model, optimizer)
    loaded = torch.load(tmp_path / "c.pth", weights_only=True)
    assert loaded["loss"] == 1.5 and loaded["use_tessera"] is False
    assert torch.equal(loaded["model_state_dict"]["weight"], model.weight.detach())


def test_cli_device_flag():
    """The training CLI accepts --device and defaults to auto."""
    from firepred.cli.train import parse_args

    required = ["-m", "swinunetr3d", "-mode", "pred", "-b", "1", "-r", "1", "-lr", "0.001",
                "-nh", "3", "-ed", "36", "-ts", "6", "-it", "1"]
    assert parse_args(required).device == "auto"
    assert parse_args(required + ["--device", "tpu"]).device == "tpu"
