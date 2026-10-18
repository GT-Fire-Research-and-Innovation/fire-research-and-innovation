"""Which device trains the model: the CPU, a CUDA GPU, or a TPU through PyTorch/XLA.

Level 3 (training). Used by training/run.py, training/loop.py and
training/checkpoints.py; cli/train.py exposes it as --device.

    auto   CUDA when a GPU is visible, else the CPU (the only behaviour before --device)
    cpu    the CPU
    cuda   the GPU(s); fails when none is visible
    tpu    one TPU core (v3) or chip (v5e) through torch_xla. [ours]

The TPU path is opt-in and leaves every other path unchanged. It differs from a
GPU run in three ways:
  * XLA records operations lazily and compiles them into one graph per step,
    run at mark_step(); the first steps (and a smaller last batch) are slow
    while graphs compile.
  * Results are close to a GPU run's but not identical (other kernels and
    matrix-multiply precision).
  * One process drives one TPU device: the other cores/chips of a TPU v3-8 or
    v5e-8 stay idle (nn.DataParallel only splits batches across GPUs).
"""

import os

import torch

DEVICES = ("auto", "cpu", "cuda", "tpu")


def xla_model():
    """torch_xla's xla_model module, imported only when a TPU is used."""
    # PJRT is PyTorch/XLA's runtime; it must be told the device kind.
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    try:
        import torch_xla.core.xla_model as xm
    except ImportError as error:
        raise RuntimeError(
            "--device tpu needs PyTorch/XLA (the torch_xla package), which the "
            "Kaggle and Colab TPU runtimes provide; it is not installed here"
        ) from error
    return xm


def resolve_device(name="auto"):
    """The torch.device for --device `name` (one of DEVICES)."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "tpu":
        return xla_model().xla_device()
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda, but no CUDA GPU is visible")
    if name not in DEVICES:
        raise ValueError(f"Unknown device {name!r}; choose from {DEVICES}")
    return torch.device(name)


def is_xla(device):
    """True when `device` is a PyTorch/XLA (TPU) device."""
    return torch.device(device).type == "xla"


def seed_device(device, seed):
    """Seed the TPU's own random generator (numpy, CPU and CUDA are seeded by the caller)."""
    if is_xla(device):
        xla_model().set_rng_state(seed)


def mark_step(device):
    """On a TPU, compile and run the operations recorded so far; elsewhere a no-op.

    Called after the forward pass (so the scores read from its outputs do not
    recompute it) and after each optimizer step (one graph per training step).
    """
    if is_xla(device):
        xla_model().mark_step()


def checkpoint_map_location(device):
    """Where torch.load puts a checkpoint's tensors for a run on `device`.

    TPU runs load onto the CPU; load_state_dict then copies the values onto the
    TPU parameters, and the optimizer moves its state to them.
    """
    return torch.device("cpu") if is_xla(device) else device


def save(obj, path, device):
    """torch.save, except that TPU tensors are first copied to the CPU (xm.save),
    so the file loads on machines without torch_xla."""
    if is_xla(device):
        xla_model().save(obj, path)
    else:
        torch.save(obj, path)
