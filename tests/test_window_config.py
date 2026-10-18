"""window_config.json: every build setting recorded; rewritten only with --overwrite."""

import json

import pytest

pytest.importorskip("huggingface_hub")

import fake_hub  # noqa: E402  (tests/support, on sys.path via conftest)
from firepred.prep import repos  # noqa: E402

REPO = "owner/windows-6-1"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setattr(fake_hub, "ROOT", str(tmp_path / "hub"))
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)
    fake_hub.CALLS.clear()
    api = fake_hub.HfApi()
    api.create_repo(REPO)
    return api


def stored(api):
    return json.loads((fake_hub._repo(REPO) / "window_config.json").read_text())


def uploads():
    return [call for call in fake_hub.CALLS if call[0] == "upload_file"]


def test_new_repo_records_every_setting(hub):
    settings = repos.window_settings(6, 1, train_val_label_sel=1, use_roi_label_sel=False)
    repos.ensure_window_config(REPO, settings)
    config = stored(hub)
    assert config["train_val_label_sel"] == 1 and config["test_label_sel"] == 1
    assert config["ts_length"] == 6 and config["crop_size"] == 256 and config["stored_channels"] == 27
    assert "NIFC" in config["notes"]["label_sel"]["0"] and config["notes"]["npz_compresslevel"] == 1
    assert repos.window_settings(6, 1)["test_label_sel"] == "roi"


def test_existing_config_is_kept_without_overwrite(hub):
    repos.ensure_window_config(REPO, repos.window_settings(6, 1))
    repos.ensure_window_config(REPO, repos.window_settings(6, 1))
    assert len(uploads()) == 1


def test_older_smaller_config_is_kept_without_overwrite(hub):
    hub.upload_file(json.dumps({"ts_length": 6, "ts_interval": 1}).encode(), "window_config.json", REPO)
    decision = repos.ensure_window_config(REPO, repos.window_settings(6, 1))
    assert stored(hub) == {"ts_length": 6, "ts_interval": 1}
    assert not decision.written and "predates" in decision.message
    repos.ensure_window_config(REPO, repos.window_settings(6, 1), overwrite=True)
    assert stored(hub)["train_val_label_sel"] == 0


def test_conflicting_setting_needs_overwrite(hub):
    repos.ensure_window_config(REPO, repos.window_settings(6, 1, train_val_label_sel=0))
    with pytest.raises(ValueError, match="train_val_label_sel"):
        repos.ensure_window_config(REPO, repos.window_settings(6, 1, train_val_label_sel=1))
    assert stored(hub)["train_val_label_sel"] == 0
    repos.ensure_window_config(REPO, repos.window_settings(6, 1, train_val_label_sel=1), overwrite=True)
    assert stored(hub)["train_val_label_sel"] == 1
