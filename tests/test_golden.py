"""Every deterministic output still equals its stored golden value.

Each group in tests/golden/cases.py is recomputed in its own process and
compared with tests/golden/values/<group>.json, bit for bit. On a machine whose
library versions differ from _meta.json (or with GOLDEN_TOLERANT=1), large
arrays may instead match on their sum and leading values.
"""

import json
import os
import sys
from pathlib import Path

import pytest

GOLDEN = Path(__file__).resolve().parent / "golden"
sys.path.insert(0, str(GOLDEN))

import capture  # noqa: E402
from encode import differences  # noqa: E402

GROUP_NAMES = sorted(path.stem for path in (GOLDEN / "values").glob("*.json") if not path.stem.startswith("_"))


def tolerant():
    """True when exact float equality cannot be expected on this machine."""
    if os.environ.get("GOLDEN_TOLERANT") == "1":
        return True
    meta = json.loads((GOLDEN / "values" / "_meta.json").read_text())
    stored = {key: value for key, value in meta["environment"].items() if key != "platform"}
    current = {key: value for key, value in capture.environment().items() if key != "platform"}
    return stored != current


def test_every_case_group_has_stored_values():
    """A group added to cases.py must be captured; a stored group must still have a case."""
    import cases

    assert sorted(cases.GROUPS) == GROUP_NAMES


@pytest.mark.parametrize("name", GROUP_NAMES)
def test_group_matches_golden(name, tmp_path):
    """Recompute one group and compare every key with the stored value."""
    fresh = json.loads(capture.run_group(name, tmp_path).read_text())
    stored = json.loads((GOLDEN / "values" / f"{name}.json").read_text())
    problems = [f"missing key {key}" for key in stored if key not in fresh]
    problems += [f"new key {key} (capture it with --reason)" for key in fresh if key not in stored]
    for key in stored.keys() & fresh.keys():
        problems += differences(stored[key], fresh[key], path=key, tolerant=tolerant())
    assert not problems, f"{name}: {len(problems)} differences\n" + "\n".join(problems[:30])
