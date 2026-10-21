"""window_config.json: every build setting recorded; rewritten only with --overwrite."""

import json

import pytest

pytest.importorskip("huggingface_hub")

import fake_hub  # noqa: E402  (tests/support, on sys.path via conftest)
from firepred.data import hub as data_hub  # noqa: E402

REPO = "owner/windows-6-1"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake Hugging Face with one empty window repo."""
    monkeypatch.setattr(fake_hub, "ROOT", str(tmp_path / "hub"))
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)
    fake_hub.CALLS.clear()
    api = fake_hub.HfApi()
    api.create_repo(REPO)
    return api


def stored(api):
    """The window_config.json stored in the fake repo."""
    return json.loads((fake_hub._repo(REPO) / "window_config.json").read_text())


def uploads():
    """The upload_file calls made so far."""
    return [call for call in fake_hub.CALLS if call[0] == "upload_file"]


def test_new_repo_records_every_setting(hub):
    """A new repo records every build setting plus the human-readable notes."""
    settings = data_hub.window_settings(6, 1, train_val_label_sel=1, use_roi_label_sel=False)
    data_hub.ensure_window_config(REPO, settings)
    config = stored(hub)
    assert config["train_val_label_sel"] == 1 and config["test_label_sel"] == 1
    assert config["ts_length"] == 6 and config["crop_size"] == 256 and config["stored_channels"] == 27
    assert "NIFC" in config["notes"]["label_sel"]["0"] and config["notes"]["npz_compresslevel"] == 6
    assert config["notes"]["npz_compression"] == "zstd"
    assert data_hub.window_settings(6, 1)["test_label_sel"] == "roi"


def test_existing_config_is_kept_without_overwrite(hub):
    """An existing, matching config is never uploaded again."""
    data_hub.ensure_window_config(REPO, data_hub.window_settings(6, 1))
    data_hub.ensure_window_config(REPO, data_hub.window_settings(6, 1))
    assert len(uploads()) == 1


def test_older_smaller_config_is_kept_without_overwrite(hub):
    """A config that predates newer settings is kept (with a note) until --overwrite."""
    hub.upload_file(json.dumps({"ts_length": 6, "ts_interval": 1}).encode(), "window_config.json", REPO)
    decision = data_hub.ensure_window_config(REPO, data_hub.window_settings(6, 1))
    assert stored(hub) == {"ts_length": 6, "ts_interval": 1}
    assert not decision.written and "predates" in decision.message
    data_hub.ensure_window_config(REPO, data_hub.window_settings(6, 1), overwrite=True)
    assert stored(hub)["train_val_label_sel"] == 0


def test_conflicting_setting_needs_overwrite(hub):
    """A conflicting setting raises unless --overwrite rewrites the config."""
    data_hub.ensure_window_config(REPO, data_hub.window_settings(6, 1, train_val_label_sel=0))
    with pytest.raises(ValueError, match="train_val_label_sel"):
        data_hub.ensure_window_config(REPO, data_hub.window_settings(6, 1, train_val_label_sel=1))
    assert stored(hub)["train_val_label_sel"] == 0
    data_hub.ensure_window_config(REPO, data_hub.window_settings(6, 1, train_val_label_sel=1), overwrite=True)
    assert stored(hub)["train_val_label_sel"] == 1
