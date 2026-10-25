"""End-to-end training runs through the CLI: they finish and produce sane results.

Exact numbers of a full run are not pinned (random-draw order may change in a
refactor); what is pinned is the shape of a run: what gets logged, which
checkpoint is kept, that -test reproduces the final scores, and that a run is
deterministic for a given seed. Default runs use the one-layer "tiny3d" model
(tests/support/tiny_model.py); `pytest -m slow` repeats one with SwinUNETR-3D.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

RUN_CLI = Path(__file__).resolve().parent / "support" / "run_cli.py"
EPOCH = "Training Epoch (Starting at 0)"
LEARNING_RATE = "Learning Rate"
VAL_LOSS_DICECE = "Validation Weighted Dice + Cross-Entropy Loss — Next-Day Fire Growth"
PERIODIC_F1 = "Test F1 — Next-Day Fire Growth (periodic, current weights)"
CHECKPOINT_FIELDS = {
    "epoch", "model_state_dict", "optimizer_state_dict", "loss", "use_tessera", "model_input_channels",
    "swin_temporal_downsample", "tessera_stats", "loss_config", "lr_config", "normalization",
}


def base_flags(model="tiny3d", width="12"):
    """Common flags of every run: 2 epochs, batch 2, seed 42."""
    return ["-m", model, "-mode", "pred", "-b", "2", "-r", "1", "-lr", "0.001", "-nh", "3", "-ed", width,
            "-ts", "6", "-it", "1", "-seed", "42", "--epochs", "2"]


class Run:
    """What one CLI run left behind: exit code, W&B calls, checkpoints, plots."""

    def __init__(self, case_dir, result):
        """Read the call log and list the files of a finished run."""
        self.dir = case_dir
        self.returncode = result.returncode
        self.stdout = result.stdout
        self.stderr = result.stderr
        calls_path = case_dir / "calls.json"
        self.calls = json.loads(calls_path.read_text()) if calls_path.exists() else []
        self.checkpoints = sorted(case_dir.glob("*.pth"))
        self.plots = sorted(case_dir.rglob("*.png"))

    def named(self, name):
        """Every call of one kind (e.g. "log", "init")."""
        return [call for call in self.calls if call[0] == name]

    def logs(self):
        """The dicts passed to wandb.log, in order."""
        return [call[1][0] for call in self.named("log")]

    def train_logs(self):
        """Per-epoch training logs."""
        return [log for log in self.logs() if EPOCH in log and LEARNING_RATE not in log and PERIODIC_F1 not in log]

    def val_logs(self):
        """Per-epoch validation logs (they carry the learning rate)."""
        return [log for log in self.logs() if LEARNING_RATE in log]

    def periodic_logs(self):
        """Periodic test-fire logs."""
        return [log for log in self.logs() if PERIODIC_F1 in log]

    def final_log(self):
        """The final test log (the one with test_f1), or None."""
        finals = [log for log in self.logs() if "test_f1" in log]
        return finals[-1] if finals else None

    def checkpoint(self, index=0):
        """Fully load one saved checkpoint."""
        return torch.load(self.checkpoints[index], map_location="cpu", weights_only=False)


def run(case_dir, flags, check=True):
    """Run the training CLI with `flags` in `case_dir` and return a Run."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", PYTHONHASHSEED="0",
               OMP_NUM_THREADS="4", PYTHONWARNINGS="ignore")
    result = subprocess.run([sys.executable, str(RUN_CLI), str(case_dir), "--", *flags],
                            capture_output=True, text=True, env=env)
    finished = Run(Path(case_dir), result)
    if check:
        assert finished.returncode == 0, finished.stderr[-3000:]
    return finished


@pytest.fixture(scope="module")
def e2e_root(tmp_path_factory):
    """Synthetic data root for end-to-end runs (2 val windows, 2 test fires) and its test CSV."""
    import synth

    return synth.data_root(tmp_path_factory.mktemp("e2e") / "data", synth.E2E_FIRES)


def data_flags(e2e_root):
    """Flags pointing a run at the synthetic data."""
    root, csv_path = e2e_root
    return ["--data-root", str(root), "--test-csv", str(csv_path)]


@pytest.fixture(scope="module")
def dicece_run(e2e_root, tmp_path_factory):
    """Dice+CE run with checkpoint upload, periodic test every epoch and plots."""
    flags = base_flags() + data_flags(e2e_root) + [
        "--loss", "dicece", "--upload-checkpoints", "--test-every", "1", "--plot-dir", "plots"]
    finished = run(tmp_path_factory.mktemp("dicece"), flags)
    finished.data_root = e2e_root[0]
    return finished


def assert_finite_scores(log, low=0.0, high=1.0, score_words=("F1", "IoU", "Dice")):
    """Every value in a log is finite; F1/IoU/Dice values lie in [low, high]."""
    for key, value in log.items():
        assert math.isfinite(value), (key, value)
        if any(word in key for word in score_words) and "Loss" not in key:
            assert low <= value <= high, (key, value)


def test_training_run_logs_every_epoch(dicece_run):
    """W&B is set up and finished, and every epoch logs train, validation and periodic test scores."""
    init = dicece_run.named("init")
    assert len(init) == 1 and init[0][2]["project"]
    assert dicece_run.named("finish")
    assert dicece_run.named("define_metric")
    train, val, periodic = dicece_run.train_logs(), dicece_run.val_logs(), dicece_run.periodic_logs()
    assert [log[EPOCH] for log in train] == [0, 1]
    assert [log[EPOCH] for log in val] == [0, 1]
    assert [log[EPOCH] for log in periodic] == [0, 1]
    for log in train + val + periodic:
        assert_finite_scores(log)
    for log in train + val:
        assert all(value > 0 for key, value in log.items() if "Loss" in key)
    assert all(log[LEARNING_RATE] == pytest.approx(1e-3) for log in val)


def test_training_run_charts_fire_and_background_scores(dicece_run):
    """Train, validation and periodic test logs chart F1 and IoU for both classes; Dice+CE charts its parts."""
    for split, logs in (("Training", dicece_run.train_logs()), ("Validation", dicece_run.val_logs()),
                        ("Test", dicece_run.periodic_logs())):
        suffix = " (periodic, current weights)" if split == "Test" else ""
        for score in ("F1", "IoU"):
            for target in ("Next-Day Fire Growth", "Background"):
                assert all(f"{split} {score} — {target}{suffix}" in log for log in logs), (split, score, target)
    for log in dicece_run.train_logs():
        parts = (log["Training Weighted Dice Loss — Next-Day Fire Growth"]
                 + log["Training Weighted Cross-Entropy Loss — Next-Day Fire Growth"])
        assert parts == pytest.approx(log["Training Weighted Dice + Cross-Entropy Loss — Next-Day Fire Growth"])
    assert not any("Mean" in key for log in dicece_run.val_logs() for key in log)


def test_training_run_logs_false_positives_and_negatives(dicece_run):
    """Train, validation, periodic and final test logs each chart both error counts, in pixels per window."""
    window_pixels = 256 * 256
    logs = dicece_run.train_logs() + dicece_run.val_logs() + dicece_run.periodic_logs() + [dicece_run.final_log()]
    for log in logs:
        for kind in ("False Positives", "False Negatives"):
            values = [value for key, value in log.items() if kind in key]
            assert len(values) == 1, (kind, sorted(log))
            assert 0.0 <= values[0] <= window_pixels, (kind, values[0])
    final = dicece_run.final_log()
    assert final["test_false_positives"] == final["Test False Positives — Next-Day Fire Growth (pixels per window)"]
    assert final["test_false_negatives"] == final["Test False Negatives — Next-Day Fire Growth (pixels per window)"]


def test_training_run_keeps_the_best_checkpoint(dicece_run):
    """One checkpoint is kept: the lowest validation loss, with every recorded setting."""
    assert len(dicece_run.checkpoints) == 1
    checkpoint = dicece_run.checkpoint()
    val_losses = [log[VAL_LOSS_DICECE] for log in dicece_run.val_logs()]
    assert checkpoint["loss"] == pytest.approx(min(val_losses))
    assert CHECKPOINT_FIELDS <= set(checkpoint)
    assert checkpoint["loss_config"]["loss"] == "dicece"
    # Inputs were standardized with the data root's own training statistics, not the paper's.
    window_config = json.loads((Path(dicece_run.data_root) / "window_config.json").read_text())
    assert checkpoint["normalization"] == {"source": "window-config", **{
        key: window_config["normalization"][key] for key in ("mean", "std")}}
    assert dicece_run.checkpoints[0].name.startswith("model_tiny3d_mode_pred_")
    assert f"epoch_{checkpoint['epoch'] + 1}" in dicece_run.checkpoints[0].name


def test_training_run_uploads_and_tests_the_best_checkpoint(dicece_run):
    """The checkpoint is uploaded with best/latest/epoch aliases, and the final test covers both fires."""
    aliases = [call[2]["aliases"] for call in dicece_run.named("run.log_artifact")]
    assert aliases and all(set(a[:2]) == {"best", "latest"} and a[2].startswith("epoch-") for a in aliases)
    final = dicece_run.final_log()
    assert final is not None and final["test_fires"] == 2
    assert_finite_scores(final)
    assert dicece_run.named("run.summary.update")
    assert len(dicece_run.plots) == 2  # one per test window


def test_tessera_plateau_run(e2e_root, tmp_path):
    """Dice loss + TESSERA inputs + plateau schedule: 172 channels and training-fire statistics saved."""
    flags = base_flags() + data_flags(e2e_root) + [
        "--loss", "dice", "--tessera", "--lr-schedule", "plateau", "--lr-patience", "0"]
    finished = run(tmp_path, flags)
    checkpoint = finished.checkpoint()
    assert checkpoint["use_tessera"] is True
    assert checkpoint["model_input_channels"] == 172
    stats = checkpoint["tessera_stats"]
    assert len(stats["mean"]) == len(stats["std"]) == 128
    assert "_tessera" in finished.checkpoints[0].name
    assert finished.final_log() is not None


def test_auto_ce_weight_run(e2e_root, tmp_path):
    """--ce-weight auto trains with (1, background / fire) of the training labels and records it everywhere."""
    from firepred.data.stats import count_labels

    root, _ = e2e_root
    counts = count_labels(root / "train")
    flags = base_flags() + data_flags(e2e_root) + [
        "--loss", "dicece", "--ce-weight", "auto", "--epochs", "1", "--no-test-after-train"]
    finished = run(tmp_path, flags)
    expected = [1.0, counts.background / counts.fire]
    recorded = finished.named("config.update")[0][1][0]
    assert recorded["ce_weight"] == expected
    assert recorded["ce_weight_counts"] == [counts.background, counts.fire]
    assert finished.checkpoint()["loss_config"]["ce_weight"] == expected
    assert all(math.isfinite(log[VAL_LOSS_DICECE]) for log in finished.val_logs())


def test_worker_cosine_run_without_final_test(e2e_root, tmp_path):
    """Two loader workers and a cosine schedule; --no-test-after-train skips the final test."""
    flags = base_flags() + data_flags(e2e_root) + [
        "--num-workers", "2", "--lr-schedule", "cosine", "--no-test-after-train"]
    finished = run(tmp_path, flags)
    rates = [log[LEARNING_RATE] for log in finished.val_logs()]
    assert rates[0] == pytest.approx(1e-3) and rates[1] < rates[0]
    assert finished.final_log() is None
    assert len(finished.checkpoints) == 1


def test_test_mode_reproduces_final_scores(dicece_run, e2e_root, tmp_path):
    """-test on the kept checkpoint gives exactly the scores the training run logged at the end."""
    flags = base_flags() + data_flags(e2e_root) + [
        "--loss", "dicece", "-test", "--checkpoint", str(dicece_run.checkpoints[0])]
    finished = run(tmp_path, flags)
    assert not finished.named("define_metric")
    assert finished.named("init")
    names = [call[1][0] for call in finished.named("run.name")]
    assert names and names[0].startswith("test_")
    trained, tested = dicece_run.final_log(), finished.final_log()
    assert tested["test_f1"] == trained["test_f1"]
    assert tested["test_iou"] == trained["test_iou"]
    assert not finished.checkpoints


def test_test_mode_finds_the_newest_checkpoint_in_the_wandb_run_folders(dicece_run, e2e_root, tmp_path,
                                                                        monkeypatch):
    """-test without --checkpoint uses the newest matching file in ./wandb/*/files, where training saves."""
    import shutil

    monkeypatch.delenv("WANDB_DIR", raising=False)
    trained = dicece_run.checkpoints[0]
    older = tmp_path / "wandb" / "run-20260101_000000-old" / "files" / trained.name
    newer = tmp_path / "wandb" / "run-20260102_000000-new" / "files" / trained.name
    for path in (older, newer):
        path.parent.mkdir(parents=True)
    older.write_bytes(b"not a checkpoint")  # would fail to load if it were picked
    os.utime(older, (1, 1))
    shutil.copy(trained, newer)
    finished = run(tmp_path, base_flags() + data_flags(e2e_root) + ["--loss", "dicece", "-test"])
    assert f"Testing checkpoint {newer.resolve()}" in finished.stdout
    assert finished.final_log()["test_f1"] == dicece_run.final_log()["test_f1"]


def test_test_mode_without_a_checkpoint_stops_before_wandb(e2e_root, tmp_path, monkeypatch):
    """-test with nothing to find fails with a clear message and starts no W&B run."""
    monkeypatch.delenv("WANDB_DIR", raising=False)
    finished = run(tmp_path, base_flags() + data_flags(e2e_root) + ["--loss", "dicece", "-test"], check=False)
    assert finished.returncode != 0
    assert "pass --checkpoint" in finished.stderr
    assert not finished.named("init")


def test_run_without_window_config_uses_official_values(e2e_root, tmp_path):
    """A data root without window_config.json trains with the official TS-SatFire values and says so in W&B."""
    from firepred.config.dataset import NORMALIZATION_MEAN
    from firepred.config.training import PAPER_CE_WEIGHT

    root, csv_path = e2e_root
    bare = tmp_path / "bare"
    bare.mkdir()
    for split in ("train", "val", "test"):
        (bare / split).symlink_to(root / split, target_is_directory=True)
    flags = base_flags() + ["--data-root", str(bare), "--test-csv", str(csv_path), "--loss", "dicece",
                            "--ce-weight", "auto", "--epochs", "1", "--no-test-after-train"]
    finished = run(tmp_path / "case", flags)
    assert "WARNING: no window_config.json" in finished.stdout
    recorded = finished.named("config.update")[0][1][0]
    assert sorted(recorded["defaults_used"]) == ["ce_weight", "normalization", "window_config"]
    assert recorded["normalization"] == "paper" and recorded["ce_weight"] == list(PAPER_CE_WEIGHT)
    checkpoint = finished.checkpoint()
    assert checkpoint["normalization"]["mean"] == [float(value) for value in NORMALIZATION_MEAN]
    assert checkpoint["loss_config"]["ce_weight"] == list(PAPER_CE_WEIGHT)


def test_test_mode_refuses_a_tessera_mismatch(dicece_run, e2e_root, tmp_path):
    """A checkpoint trained without TESSERA cannot be tested with --tessera."""
    flags = base_flags() + data_flags(e2e_root) + [
        "--loss", "dicece", "--tessera", "-test", "--checkpoint", str(dicece_run.checkpoints[0])]
    finished = run(tmp_path, flags, check=False)
    assert finished.returncode != 0
    assert "tessera" in finished.stderr.lower()


def weights(finished):
    """The kept checkpoint's model weights."""
    return finished.checkpoint()["model_state_dict"]


def assert_same_weights(left, right):
    """Two state dicts hold bit-identical tensors under the same names."""
    assert list(left) == list(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_runs_are_deterministic_and_periodic_tests_change_nothing(dicece_run, e2e_root, tmp_path):
    """The same seed gives the same weights, with or without periodic test scoring."""
    common = base_flags() + data_flags(e2e_root) + ["--loss", "dicece", "--no-test-after-train"]
    again = run(tmp_path / "again", common + ["--test-every", "1"])
    without = run(tmp_path / "without", common + ["--test-every", "0"])
    assert_same_weights(weights(dicece_run), weights(again))
    assert_same_weights(weights(again), weights(without))


@pytest.mark.slow
def test_swinunetr3d_run(e2e_root, tmp_path):
    """The real model trains, keeps a checkpoint and scores the test fires."""
    flags = base_flags("swinunetr3d", "12") + data_flags(e2e_root) + ["--loss", "dicece", "--test-every", "1"]
    finished = run(tmp_path, flags)
    assert len(finished.checkpoints) == 1
    final = finished.final_log()
    assert final is not None and final["test_fires"] == 2
    assert_finite_scores(final)
