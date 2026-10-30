"""The dataset pipeline: functional rules for the dataset-building modules of data/, and the masters against an offline fake hub."""

import ast
import json
import sys
from pathlib import Path

import pytest

import fake_hub

DATA = Path(__file__).resolve().parents[1] / "firepred" / "data"
# The dataset-building modules (functional style); the training-side modules
# (archive's readers, stats, fire_dataset) are partly object-oriented.
BUILD = ("files", "hub", "windows", "tessera", "tessera_master", "windows_master")
MASTERS = {"windows_master.py", "tessera_master.py"}
GOLDEN = Path(__file__).resolve().parent / "golden"


def modules():
    """(path, parsed tree) of every dataset-building module."""
    return [(DATA / f"{name}.py", ast.parse((DATA / f"{name}.py").read_text())) for name in BUILD]


def functions(tree):
    """Every function definition in a module, nested ones included."""
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def returns_a_value(function):
    """True when the function has a `return <value>` of its own (not counting nested functions)."""
    stack = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return) and node.value is not None:
            if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def test_every_build_function_returns_a_value():
    """Functional style: no function works only by side effect."""
    silent = [
        f"{path.name}:{fn.lineno} {fn.name}"
        for path, tree in modules()
        for fn in functions(tree)
        if not returns_a_value(fn)
    ]
    assert not silent, "functions without a returned value:\n" + "\n".join(silent)


def test_only_the_masters_print():
    """Helpers return notes and reports; the masters decide what to show."""
    prints = [
        f"{path.name}:{node.lineno}"
        for path, tree in modules()
        if path.name not in MASTERS
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"
    ]
    assert not prints, prints


def test_build_modules_have_no_mutable_state():
    """No `global`, classes are frozen dataclasses, module-level containers are UPPER_CASE constants."""
    problems = []
    for path, tree in modules():
        problems += [f"{path.name}:{n.lineno} global" for n in ast.walk(tree) if isinstance(n, ast.Global)]
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                frozen = any(
                    isinstance(d, ast.Call) and getattr(d.func, "id", "") == "dataclass"
                    and any(k.arg == "frozen" and getattr(k.value, "value", False) for k in d.keywords)
                    for d in node.decorator_list
                )
                if not frozen:
                    problems.append(f"{path.name}:{node.lineno} class {node.name} is not a frozen dataclass")
            if isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                problems += [f"{path.name}:{node.lineno} {n}" for n in names if not n.isupper()]
    assert not problems, "\n".join(problems)


# ----- masters against the fake hub ----- #


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """An offline fake huggingface_hub with a source repo of 4 fires (see golden/cases.py)."""
    sys.path.insert(0, str(GOLDEN))
    import cases

    monkeypatch.setattr(fake_hub, "ROOT", str(tmp_path / "hub"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    fake_hub.CALLS.clear()
    cases.build_source_repo(fake_hub, "owner/src")
    return tmp_path


def test_process_windows_only_fires_builds_just_those(hub):
    """only_fires limits a build to the listed fires; the report and the repo agree."""
    from firepred.data import windows_master

    report = windows_master.process_windows(
        "token", src_repo="owner/src", window_repo="owner/w", length=6, interval=1,
        only_fires=["20000019"], roi_dir=hub / "roi", work_dir=hub / "work",
    )
    assert report.built == {"train": ("20000019",), "val": (), "test": ()}
    assert report.uploaded == ("train/p20000019.npz",)
    files = sorted(p.name for p in fake_hub._repo("owner/w").rglob("*") if p.is_file())
    assert files == ["p20000019.npz", "window_config.json"]
    assert not any(call[0] == "login" for call in fake_hub.CALLS)  # tokens are passed, never a global login


def test_rebuild_zip_and_stage_training_data(hub):
    """rebuild_data_zip publishes data.zip; stage_training_data downloads and extracts it, or uses an attached zip."""
    from firepred.data import windows_master

    windows_master.process_windows("token", src_repo="owner/src", window_repo="owner/w", length=6, interval=1,
                                   roi_dir=hub / "roi", work_dir=hub / "work")
    upload = windows_master.rebuild_data_zip("token", window_repo="owner/w", src_repo="owner/src",
                                             work_dir=hub / "work", roi_dir=hub / "roi")
    assert upload.path_in_repo == "data.zip" and (fake_hub._repo("owner/w") / "data.zip").is_file()

    work = windows_master.workspace(hub / "kaggle", 6, 1, use_tessera=False, window_repo="owner/w")
    staged = windows_master.stage_training_data(work, token="token")
    assert staged.train_root == staged.data_root and staged.counts == {"train": 2, "val": 1, "test": 1}

    attached = windows_master.workspace(hub / "attached", 6, 1)
    staged_zip = windows_master.stage_training_data(attached, attached_zip=upload.zip_path)
    assert staged_zip.counts == staged.counts


def test_workspace_secret_and_caches(tmp_path, monkeypatch):
    """workspace names every notebook path; read_secret falls back to the environment; pin_caches sets env vars."""
    from firepred.data import windows_master

    work = windows_master.workspace(tmp_path, 10, 3, use_tessera=True)
    assert work.window_repo.endswith("-10-3") and work.data_root.name == "data_ts10_it3_tessera"
    assert work.view_dir.name == "data-ts10-it3-tessera-view"
    monkeypatch.setenv("MY_TOKEN", "abc")
    assert windows_master.read_secret("MY_TOKEN") == "abc"
    for name in ("HF_HOME", "XDG_CACHE_HOME", "TMPDIR"):
        monkeypatch.setenv(name, "unchanged")
    settings = windows_master.pin_caches(tmp_path / "cache")
    assert settings["TMPDIR"] == str(tmp_path / "cache" / "tmp") and (tmp_path / "cache" / "tmp").is_dir()


def test_tessera_view_refuses_to_clear_real_files(data_root, tmp_path):
    """build_tessera_view replaces an old symlink view but never deletes a folder holding real data."""
    from firepred.data import tessera

    root, _ = data_root
    (root / "tessera" / "excluded_npz.txt").write_text("train/p20000002.npz\n")
    view = tessera.build_tessera_view(root, root / "tessera", tmp_path / "view")
    assert view.dropped == ("train/p20000002.npz",)
    again = tessera.build_tessera_view(root, root / "tessera", tmp_path / "view")
    assert again.dropped == view.dropped
    with pytest.raises(RuntimeError, match="real files"):
        tessera.build_tessera_view(root, root / "tessera", root)
    (root / "tessera" / "excluded_npz.txt").unlink()


def test_window_config_decision_is_pure():
    """decide_window_config needs no hub: new repo -> write, conflict -> error, older file -> keep."""
    from firepred.data import hub

    settings = hub.window_settings(6, 1)
    assert hub.decide_window_config(None, settings, False, "r").written
    with pytest.raises(ValueError, match="ts_interval"):
        hub.decide_window_config({**settings, "ts_interval": 3}, settings, False, "r")
    kept = hub.decide_window_config({"ts_length": 6, "firepred_grid": "viirs_day"}, settings, False, "r")
    assert not kept.written and "ts_interval" in kept.missing
    # Without firepred_grid the file holds index-cropped FirePred windows: a conflict, not a gap.
    with pytest.raises(ValueError, match="firepred_grid"):
        hub.decide_window_config({"ts_length": 6}, settings, False, "r")
    assert json.loads(json.dumps(settings)) == settings


def test_window_plan_skips_built_and_missing_fires():
    """plan_windows builds every fire with GeoTIFFs not yet in the repo, with its label rule, in split order."""
    from firepred.data.files import RoiTable
    from firepred.data.windows_master import plan_windows

    roi = RoiTable(("a", "b"), ("c",), ("d",))
    locations = {"a": {"VIIRS_Day": ["a/VIIRS_Day/1.tif"]}, "c": {}, "d": {}}
    plan = plan_windows(roi, {"d": 1}, locations, {"val/pc.npz"}, ("train", "val", "test"))
    assert [(job.split, job.fire_id, job.label_sel) for job in plan.jobs] == [("train", "a", 0), ("test", "d", 1)]
    assert plan.already_built == ("val/pc.npz",) and plan.without_geotiffs == ("b",)
    assert [job.fire_id for job in plan_windows(roi, {}, locations, {"val/pc.npz"}, ("val",), overwrite=True).jobs] == ["c"]
    assert plan_windows(roi, {}, locations, set(), ("train",), only_fires=["b"]).jobs == ()


def test_tessera_plan_gives_each_skipped_fire_a_reason():
    """plan_tessera keeps fires with a prior-year embedding and says why every other fire is skipped."""
    from firepred.data.files import RoiTable
    from firepred.data.tessera_master import plan_tessera

    roi = RoiTable(("a", "b", "c"), (), ("d",), start_years={"a": 2019, "b": 2017, "d": 2021})
    days = {"VIIRS_Day": ["x/1.tif", "x/2.tif", "x/3.tif"]}
    locations = {"a": days, "b": days, "c": days, "d": days}
    plan = plan_tessera(roi, ("train", "test"), locations, {"tessera/pd.npz"}, range(2017, 2025))
    assert [(job.fire_id, job.day_files, job.start_year) for job in plan.jobs] == [("a", ("x/1.tif", "x/3.tif"), 2019)]
    assert plan.skipped == {"b": "no TESSERA for prior year 2016", "c": "no start_date in ROI CSVs"}
    assert plan.already_built == ("d",) and plan.fire_ids == ("a", "b", "c", "d")
