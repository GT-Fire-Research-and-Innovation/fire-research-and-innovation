"""The command-line entry points: flags, the model scaffold, the checks."""

import importlib.util

import pytest

from firepred.cli import check, new_model
from firepred.cli import train as train_cli
from firepred.models import registry

TRAIN_FLAGS = {
    "-m", "-mode", "-b", "-r", "-lr", "-nh", "-ed", "-nc", "-ts", "-it", "-test", "-seed", "--epochs",
    "--num-workers", "--data-root", "--unzip-dir", "--unetr-version", "--checkpoint", "--load-epoch",
    "--test-csv", "--plot-dir", "--test-every", "--test-after-train", "--tessera", "--tessera-dir", "--loss",
    "--swin-temporal-downsample", "--ce-weight", "--lambda-dice", "--lambda-ce", "--lr-schedule",
    "--lr-decay-factor", "--lr-patience", "--lr-min", "--upload-checkpoints", "--tessera-min-coverage",
    "--model-option", "--wandb-user", "--lambda-focal", "--focal-gamma", "--focal-alpha",
}
REQUIRED = ["-m", "swinunetr3d", "-mode", "pred", "-b", "8", "-r", "1", "-lr", "0.001",
            "-nh", "3", "-ed", "36", "-ts", "6", "-it", "1"]


def test_train_flags(capsys):
    with pytest.raises(SystemExit):
        train_cli.parse_args(["--help"])
    text = capsys.readouterr().out
    assert all(flag in text for flag in TRAIN_FLAGS)


def test_train_defaults_and_model_options():
    args = train_cli.parse_args(REQUIRED + ["--model-option", "width=32", "--model-option", "kind=conv"])
    assert args.loss is None and args.lr_schedule == "none" and args.epochs == 100
    assert args.unzip_dir == "off"
    assert args.test_csv.endswith("roi/us_fire_2021_out_new.csv")
    assert args.model_option == {"width": 32, "kind": "conv"}
    assert train_cli.parse_args(REQUIRED).model_option is None


def test_new_model_scaffold_registers_and_trains(tmp_path):
    path = new_model.create("scaffold_test_model", directory=tmp_path)
    spec = importlib.util.spec_from_file_location("scaffold_test_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        check.main(["model", "-m", "scaffold_test_model", "--size", "32", "--batch", "2", "-ts", "2",
                    "--device", "cpu"])
    finally:
        registry._BUILDERS.pop("scaffold_test_model", None)
    with pytest.raises(FileExistsError):
        new_model.create("scaffold_test_model", directory=tmp_path)


def test_check_data_on_synthetic_root(data_root, capsys):
    root, _ = data_root
    check.main(["data", "--data-root", str(root), "--batch", "2", "--tessera"])
    assert "model channels: 172" in capsys.readouterr().out


def test_zip_with_escaping_member_is_refused(tmp_path):
    import zipfile

    from firepred.prep.staging import check_members_inside

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("train/ok.npz", b"x")
        archive.writestr("../escape.txt", b"x")
    with zipfile.ZipFile(bad) as archive, pytest.raises(ValueError, match="escapes"):
        check_members_inside(archive, tmp_path / "out")
