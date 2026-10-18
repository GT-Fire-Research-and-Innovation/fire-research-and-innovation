"""Turn computed values into JSON-safe golden values, and compare two of them.

encode() keeps small arrays byte for byte and reduces large ones to a hash
plus a float64 sum and the first values, so a golden file stays small and a
mismatch is still easy to read. differences() compares bit for bit; with
tolerant=True a hashed array may instead match on its sum and leading values
(for a machine with different BLAS or thread scheduling).
"""

import base64
import hashlib
import math
from pathlib import Path

HASH_ABOVE = 64    # arrays with more elements are stored as a hash
HEAD = 8           # leading values kept next to a hash, for diagnosis
RTOL = 1e-5        # relative tolerance of the tolerant comparison


def _b64(raw):
    """Base64 text of raw bytes."""
    return base64.b64encode(raw).decode("ascii")


def encode(value, normalize=None):
    """A JSON-safe, comparable form of `value` (tensors, arrays, containers, scalars).

    `normalize` rewrites strings and paths (e.g. a scratch dir to "<WORK>").
    Containers keep their type name, so a tuple and a list never compare equal.
    """
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype == object:
            return ["object-array", [encode(item, normalize) for item in array.ravel().tolist()]]
        if array.size > HASH_ABOVE:
            numeric = array.dtype.kind in "fiub"
            flat = array.ravel()
            return [
                "array-hash",
                str(array.dtype),
                list(array.shape),
                hashlib.sha256(array.tobytes()).hexdigest(),
                float(array.astype(np.float64).sum()) if numeric else None,
                [float(item) for item in flat[:HEAD]] if numeric else None,
            ]
        return ["array", str(array.dtype), list(array.shape), _b64(array.tobytes())]
    if isinstance(value, np.generic):
        return ["np", str(value.dtype), _b64(value.tobytes())]
    if isinstance(value, dict):
        return ["dict", [[encode(k, normalize), encode(v, normalize)] for k, v in value.items()]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [encode(item, normalize) for item in value]]
    if isinstance(value, Path):
        return ["path", normalize(str(value)) if normalize else str(value)]
    if isinstance(value, str):
        return normalize(value) if normalize else value
    if isinstance(value, bytes):
        return ["bytes", _b64(value)]
    if isinstance(value, torch.device):
        return ["device", str(value)]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return ["repr", type(value).__name__, repr(value)]


def _same_float(a, b):
    """Bit-for-bit float equality (NaN equals NaN, 0.0 differs from -0.0)."""
    import struct

    return struct.pack("<d", a) == struct.pack("<d", b)


def _close(a, b):
    """Floats equal within RTOL (NaN equals NaN)."""
    if a is None or b is None:
        return a is b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=RTOL)


def differences(expected, actual, path="", tolerant=False, limit=20):
    """Human-readable list of where `actual` differs from `expected` (empty when equal)."""
    found = []

    def walk(a, b, where):
        """Compare a and b recursively, appending to `found` until `limit`."""
        if len(found) >= limit:
            return
        if isinstance(a, float) and isinstance(b, float):
            if not _same_float(a, b):
                found.append(f"{where}: {a!r} != {b!r}")
            return
        if type(a) is not type(b):
            found.append(f"{where}: type {type(a).__name__} != {type(b).__name__} ({a!r:.80} vs {b!r:.80})")
            return
        if isinstance(a, dict):
            for key in a.keys() | b.keys():
                if key not in a or key not in b:
                    found.append(f"{where}.{key}: only in {'expected' if key in a else 'actual'}")
                else:
                    walk(a[key], b[key], f"{where}.{key}")
            return
        if isinstance(a, list):
            if tolerant and len(a) == 6 and a and a[0] == "array-hash" and b[0] == "array-hash":
                same_meta = a[1:3] == b[1:3]
                close_sum = _close(a[4], b[4])
                close_head = a[5] is None or all(map(_close, a[5], b[5] or []))
                if not (same_meta and close_sum and close_head):
                    found.append(f"{where}: array differs (sum {a[4]!r} vs {b[4]!r})")
                return
            if len(a) != len(b):
                found.append(f"{where}: length {len(a)} != {len(b)}")
                return
            for index, (left, right) in enumerate(zip(a, b)):
                walk(left, right, f"{where}[{index}]")
            return
        if a != b:
            found.append(f"{where}: {a!r:.120} != {b!r:.120}")

    walk(expected, actual, path)
    return found
