"""Which device trains the model: the CPU, a CUDA GPU, or a TPU through PyTorch/XLA.

Level 3 (training). Device.resolve(--device) is built by training/experiment.py
(and cli/check.py) and handed to the Tracker, Trainer, Evaluator and the
checkpoint classes.

    auto   CUDA when a GPU is visible, else the CPU (the only behaviour before --device)
    cpu    the CPU
    cuda   the GPU(s); fails when none is visible
    tpu    one TPU core (v3) or chip (v5e) through torch_xla. [ours]

The TPU path is opt-in and leaves every other path unchanged. It differs from a
GPU run in three ways:
  * XLA records operations lazily and compiles them into one graph per step,
    run at Device.mark_step(); the first steps (and a smaller last batch) are slow
    while graphs compile.
  * Results are close to a GPU run's but not identical (other kernels and
    matrix-multiply precision).
  * One process drives one TPU device: the other cores/chips of a TPU v3-8 or
    v5e-8 stay idle (nn.DataParallel only splits batches across GPUs).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch

DEVICES = ("auto", "cpu", "cuda", "tpu")


@dataclass(frozen=True)
class Device:
    """Hold the device a run trains on and do the few things that differ on a TPU.

    A run makes one Device from --device and hands it to every class that
    touches tensors. On a CPU or GPU most methods do nothing special; on a TPU
    they go through PyTorch/XLA, which is imported only then.

    Attributes:
        torch_device (torch.device): the torch device tensors and models are moved to.
    """

    torch_device: torch.device

    @classmethod
    def resolve(cls, name="auto"):
        """Turn a --device name into the Device to train on.

        "auto" picks a GPU when one is visible, else the CPU. "tpu" is one TPU
        core or chip through PyTorch/XLA.

        Args:
            name (str): "auto", "cpu", "cuda" or "tpu".

        Returns:
            Device: the device.

        Raises:
            RuntimeError: for "cuda" without a visible GPU, or "tpu" without torch_xla.
            ValueError: for an unknown name.
        """
        if name == "auto":
            return cls(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if name == "tpu":
            return cls(cls.xla_model().xla_device())
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda, but no CUDA GPU is visible")
        if name not in DEVICES:
            raise ValueError(f"Unknown device {name!r}; choose from {DEVICES}")
        return cls(torch.device(name))

    @classmethod
    def of(cls, device):
        """Wrap a torch device (or its name) as a Device; a Device is returned as it is.

        Classes that take a device accept either kind through this. Unlike
        resolve, a name is used literally ("cpu" stays the CPU, "auto" is refused).

        Args:
            device (Device | torch.device | str): the device.

        Returns:
            Device: the device.
        """
        return device if isinstance(device, cls) else cls(torch.device(device))

    @staticmethod
    def xla_model():
        """Import PyTorch/XLA's device module, only when a TPU is used.

        GPU and CPU runs never import torch_xla, so it need not be installed. The
        PJRT runtime is told the device kind first.

        Returns:
            module: torch_xla.core.xla_model.

        Raises:
            RuntimeError: if torch_xla is not installed.
        """
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

    @property
    def type(self):
        """Give the device kind: "cpu", "cuda" or "xla".

        The gradient scaler and the memory report check it. It is the torch
        device's own type.

        Returns:
            str: the device type.
        """
        return self.torch_device.type

    @property
    def is_xla(self):
        """Say whether this is a TPU (PyTorch/XLA).

        TPU runs need a few extra calls (mark_step, xm.save). Everything else is
        identical.

        Returns:
            bool: True for an XLA device.
        """
        return self.type == "xla"

    @property
    def map_location(self):
        """Choose where a loaded checkpoint's tensors are placed.

        TPU runs load onto the CPU; load_state_dict then copies the values onto the
        TPU parameters. Other runs load straight onto the device.

        Returns:
            torch.device: where to load the tensors.
        """
        return torch.device("cpu") if self.is_xla else self.torch_device

    def seed(self, seed):
        """Seed the TPU's own random generator; other devices need nothing here.

        numpy, the CPU and CUDA are seeded by the experiment. A TPU has a separate
        generator.

        Args:
            seed (int): the run's seed.

        Returns:
            int: seed.
        """
        if self.is_xla:
            self.xla_model().set_rng_state(seed)
        return seed

    def mark_step(self):
        """On a TPU, run the operations recorded so far; elsewhere, do nothing.

        XLA records operations lazily and compiles them into one graph per step.
        Calling this after the forward pass and after the optimizer step gives one
        graph per training step.

        Returns:
            Device: this device.
        """
        if self.is_xla:
            self.xla_model().mark_step()
        return self

    def save(self, obj, path):
        """Save an object with torch.save, copying TPU tensors to the CPU first.

        A checkpoint saved from a TPU run must load on machines without torch_xla.
        On other devices this is plain torch.save.

        Args:
            obj (object): what to save (a checkpoint dict).
            path (str | Path): the file.

        Returns:
            Path: path.
        """
        if self.is_xla:
            self.xla_model().save(obj, path)
        else:
            torch.save(obj, path)
        return Path(path)
