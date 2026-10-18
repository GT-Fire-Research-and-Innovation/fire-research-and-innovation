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
    """--help lists every training flag the notebooks rely on."""
    with pytest.raises(SystemExit):
        train_cli.parse_args(["--help"])
    text = capsys.readouterr().out
    assert all(flag in text for flag in TRAIN_FLAGS)


def test_train_defaults_and_model_options():
    """Training defaults, and --model-option pairs parsed as Python literals."""
    args = train_cli.parse_args(REQUIRED + ["--model-option", "width=32", "--model-option", "kind=conv"])
    assert args.loss is None and args.lr_schedule == "none" and args.epochs == 100
    assert args.unzip_dir == "off"
    assert args.test_csv.endswith("roi/us_fire_2021_out_new.csv")
    assert args.model_option == [("width", 32), ("kind", "conv")]  # parsed flags stay as given
    assert train_cli.config_from_args(args).model_options["width"] == 32
    assert train_cli.parse_args(REQUIRED).model_option is None


def test_new_model_scaffold_registers_and_trains(tmp_path):
    """A scaffolded model registers, builds, trains one step, and its name cannot be scaffolded twice."""
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
    """`check data` reads a synthetic TESSERA batch and reports 172 model channels."""
    root, _ = data_root
    check.main(["data", "--data-root", str(root), "--batch", "2", "--tessera"])
    assert "model channels: 172" in capsys.readouterr().out


def test_zip_with_escaping_member_is_refused(tmp_path):
    """A zip member that would land outside the extraction folder is refused."""
    import zipfile

    from firepred.data.files import check_members_inside

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("train/ok.npz", b"x")
        archive.writestr("../escape.txt", b"x")
    with zipfile.ZipFile(bad) as archive, pytest.raises(ValueError, match="escapes"):
        check_members_inside(archive, tmp_path / "out")


def test_every_dataset_command_reaches_its_master_with_its_flags():
    """Each subcommand calls its master function with the parsed flags under the same parameter names."""
    import inspect

    from firepred.cli import dataset as dataset_cli

    for command in dataset_cli.COMMANDS:
        args = dataset_cli.parse_args(["--token", "t", command.name])
        kwargs = dataset_cli.call_arguments(command, args)
        parameters = inspect.signature(command.function).parameters
        assert set(kwargs) <= set(parameters), (command.name, set(kwargs) - set(parameters))
        required = {name for name, p in parameters.items() if p.default is inspect.Parameter.empty}
        assert required <= set(kwargs), (command.name, required - set(kwargs))
        assert kwargs.get("token", "t") == "t"
    upload = dataset_cli.call_arguments(dataset_cli.BY_NAME["upload-zip"],
                                        dataset_cli.parse_args(["upload-zip", "--zip-repo", "tessera", "-ts", "10"]))
    assert upload["repo_id"] == "SamuelWu318/ts-tesserafire"
    pull = dataset_cli.call_arguments(dataset_cli.BY_NAME["pull-zip"], dataset_cli.parse_args(["pull-zip", "-ts", "10", "-it", "3"]))
    assert pull["repo_id"].endswith("-10-3") and pull["data_root"] == "./data"
