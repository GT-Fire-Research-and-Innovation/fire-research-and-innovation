"""TESSERA storage: float16 by default, int8 with a per-band scale and offset on request, and conversion."""

import filecmp
import sys

import numpy as np
import pytest

from firepred.config.dataset import TESSERA_QUANTIZATION
from firepred.data.archive import (
    compute_tessera_stats,
    convert_tessera_npz,
    dequantize_int8,
    open_npz,
    quantize_int8,
    read_tessera,
    save_tessera_npz,
    tessera_quantization_of,
)


def smooth_map(seed=0, side=256):
    """A (128, side, side) float32 embedding with NaN where coverage is 0, and that coverage."""
    rng = np.random.default_rng(seed)
    centres = np.linspace(-1.0, 1.0, 128, dtype=np.float32)[:, None, None]
    embedding = centres + rng.normal(0.0, 0.5, size=(128, side, side)).astype(np.float32)
    coverage = rng.random((side, side)).astype(np.float32)
    coverage[:7] = 0
    embedding[:, coverage == 0] = np.nan
    return embedding, coverage


def test_default_stores_float16_as_before(tmp_path):
    """Without a quantization the file holds the float16 embedding, 0 where empty, and no int8 members."""
    assert TESSERA_QUANTIZATION == "none"
    embedding, coverage = smooth_map()
    path = save_tessera_npz(tmp_path / "p1.npz", embedding, coverage, {})
    with open_npz(path) as archive:
        assert archive["embedding"].dtype == np.float16
        assert not {"quantization", "embedding_int8", "embedding_scale", "embedding_offset"} & set(archive.files)
    read, _ = read_tessera(path)
    assert np.array_equal(read, np.nan_to_num(embedding, nan=0.0).astype(np.float16))


def test_int8_reads_back_within_half_a_step(tmp_path):
    """An int8 file reads back as float16 within half a step, 0 where empty, and is much smaller."""
    embedding, coverage = smooth_map()
    plain = save_tessera_npz(tmp_path / "plain.npz", embedding, coverage, {"fire_id": "x"})
    path = save_tessera_npz(tmp_path / "p1.npz", embedding, coverage, {"fire_id": "x"}, quantization="int8")
    with open_npz(path) as archive:
        # No "embedding" member: a reader that predates int8 fails instead of using codes as values.
        assert "embedding" not in archive.files
        assert archive["embedding_int8"].dtype == np.int8 and str(archive["quantization"]) == "int8"
        assert str(archive["fire_id"]) == "x"
        scale = archive["embedding_scale"]
    read, read_coverage = read_tessera(path)
    assert read.dtype == np.float16 and read.shape == embedding.shape
    has_data = coverage > 0
    error = np.abs(read.astype(np.float32) - embedding)[:, has_data]
    assert (error <= scale[:, None] / 2 + 4e-3).all()  # half a step, plus float16 rounding
    assert not read[:, ~has_data].any()
    assert np.array_equal(read_coverage, coverage.astype(np.float16))
    assert path.stat().st_size < 0.6 * plain.stat().st_size


def test_int8_statistics_match_float16(tmp_path):
    """Training statistics from int8 files match those from float16 files of the same maps."""
    plain, quantized = [], []
    for seed in range(2):
        embedding, coverage = smooth_map(seed)
        plain.append(read_tessera(save_tessera_npz(tmp_path / f"a{seed}.npz", embedding, coverage, {})))
        quantized.append(read_tessera(save_tessera_npz(tmp_path / f"b{seed}.npz", embedding, coverage, {},
                                                       quantization="int8")))
    expected, got = compute_tessera_stats(plain), compute_tessera_stats(quantized)
    assert np.allclose(got["mean"], expected["mean"], atol=1e-3)
    assert np.allclose(got["std"], expected["std"], rtol=1e-2)


def test_constant_and_empty_bands_survive():
    """A constant band reads back exactly (no division by a zero range); a map without data is all zero."""
    embedding = np.full((2, 4, 4), 0.3, dtype=np.float32)
    embedding[1] = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    codes, scale, offset = quantize_int8(embedding, np.ones((4, 4), dtype=bool))
    assert not codes[0].any() and scale[0] == 1.0
    assert np.allclose(dequantize_int8(codes, scale, offset)[0], 0.3, atol=1e-3)
    # The band's whole range is used and nothing is clipped: its ends map to -127 and 127.
    assert codes[1].min() == -127 and codes[1].max() == 127
    assert np.allclose(dequantize_int8(codes, scale, offset, np.float32)[1], embedding[1], atol=scale[1] / 2 + 1e-6)
    codes, scale, offset = quantize_int8(np.full((2, 4, 4), np.nan, np.float32), np.zeros((4, 4), dtype=bool))
    assert not codes.any() and not dequantize_int8(codes, scale, offset).any()


def test_bad_input_is_refused(tmp_path):
    """An unknown quantization, or a missing value inside a covered cell, is an error."""
    embedding, coverage = smooth_map()
    with pytest.raises(ValueError, match="quantization"):
        save_tessera_npz(tmp_path / "p1.npz", embedding, coverage, {}, quantization="int4")
    embedding[3, 100, 100] = np.nan
    with pytest.raises(ValueError, match="band 3"):
        save_tessera_npz(tmp_path / "p2.npz", embedding, coverage, {}, quantization="int8")


def test_build_fire_tessera_writes_int8(tmp_path, monkeypatch):
    """build_fire_tessera(quantization="int8") writes an int8 file that reads like the float16 one."""
    import importlib.metadata

    import synth

    real_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "0.0-fake" if name == "geotessera" else real_version(name))

    from firepred.data.tessera import build_fire_tessera

    local_map = synth.fire_geotiffs(tmp_path / "geotiffs", n_days=2)
    days = {name: path for name, path in local_map.items() if "/VIIRS_Day/" in name}
    plain = build_fire_tessera(synth.FakeGeoTessera(), "1", days, 2020, tmp_path / "plain.npz")
    quantized = build_fire_tessera(synth.FakeGeoTessera(), "1", days, 2020, tmp_path / "int8.npz",
                                   quantization="int8")
    with open_npz(quantized.path) as archive:
        assert "embedding_int8" in archive.files and int(archive["tessera_year"]) == 2019
        scale = archive["embedding_scale"]
    (expected, coverage), (got, _) = read_tessera(plain.path), read_tessera(quantized.path)
    has_data = coverage.astype(np.float32) > 0
    error = np.abs(got.astype(np.float32) - expected.astype(np.float32))[:, has_data]
    assert (error.max(axis=1) <= scale / 2 + 4e-3).all()  # half a step, plus float16 rounding twice


def test_build_step_and_cli_pass_the_option_on(tmp_path, monkeypatch):
    """`dataset tessera --tessera-quantization int8` reaches process_tessera; build_one hands it to the build."""
    from firepred.cli import dataset as dataset_cli
    from firepred.data import tessera_master

    command = dataset_cli.BY_NAME["tessera"]
    assert dataset_cli.call_arguments(command, dataset_cli.parse_args(["tessera"]))["quantization"] == "none"
    args = dataset_cli.parse_args(["tessera", "--tessera-quantization", "int8"])
    assert dataset_cli.call_arguments(command, args)["quantization"] == "int8"
    with pytest.raises(SystemExit):
        dataset_cli.parse_args(["tessera", "--tessera-quantization", "int4"])

    seen = {}
    monkeypatch.setattr(tessera_master.hub, "download_file",
                        lambda repo, name, token, cache_dir: f"{cache_dir}/{name}")
    monkeypatch.setattr(tessera_master, "build_fire_tessera", lambda *args, **kwargs: seen.update(kwargs))
    tessera_master.build_one(None, "1", ("a.tif",), 2020, "src", None, tmp_path, 128, tmp_path,
                             cells_per_pixel=5, quantization="int8")
    assert seen["quantization"] == "int8" and seen["cells_per_pixel"] == 5


# ----- Converting files and repos ----- #


def test_convert_file_keeps_provenance_and_round_trips(tmp_path):
    """float16 -> int8 keeps every provenance member and adds the given ones; the same storage is a plain copy."""
    embedding, coverage = smooth_map()
    source = save_tessera_npz(tmp_path / "p7.npz", embedding, coverage,
                              {"fire_id": "7", "tessera_year": np.int16(2019), "code_version": "abc"})
    quantized = convert_tessera_npz(source, tmp_path / "q.npz", "int8", {"converted_from": "owner/src"})
    assert tessera_quantization_of(quantized) == "int8"
    with open_npz(quantized) as archive:
        assert (str(archive["fire_id"]), int(archive["tessera_year"]), str(archive["code_version"])) == ("7", 2019, "abc")
        assert str(archive["converted_from"]) == "owner/src"
        assert "embedding" not in archive.files
        scale = archive["embedding_scale"]
    (expected, _), (got, got_coverage) = read_tessera(source), read_tessera(quantized)
    has_data = coverage > 0
    error = np.abs(got.astype(np.float32) - expected.astype(np.float32))[:, has_data]
    assert (error <= scale[:, None] / 2 + 4e-3).all()
    assert np.array_equal(got_coverage, coverage.astype(np.float16))

    back = convert_tessera_npz(quantized, tmp_path / "back.npz", "none")
    assert tessera_quantization_of(back) == "none" and np.array_equal(read_tessera(back)[0], got)
    copy = convert_tessera_npz(quantized, tmp_path / "copy.npz", "int8", {"converted_from": "ignored"})
    assert filecmp.cmp(quantized, copy, shallow=False)  # nothing rounded twice
    with pytest.raises(ValueError, match="quantization"):
        convert_tessera_npz(source, tmp_path / "x.npz", "int4")


def test_plan_conversion_picks_fire_files_not_yet_converted():
    """Only tessera/p<id>.npz files are converted; ones in the destination are skipped unless overwrite."""
    from firepred.data.tessera_master import plan_conversion

    source = ["tessera/p2.npz", "tessera/p1.npz", "tessera.zip", "README.md", "tessera/p3.npz", "tessera/notes.npz"]
    dest = ["tessera/p2.npz", ".gitattributes"]
    assert plan_conversion(source, dest) == (("tessera/p1.npz", "tessera/p3.npz"), ("tessera/p2.npz",))
    assert plan_conversion(source, dest, overwrite=True) == (("tessera/p1.npz", "tessera/p2.npz", "tessera/p3.npz"), ())
    assert plan_conversion(source, dest, only_fires=["3", "2"]) == (("tessera/p3.npz",), ("tessera/p2.npz",))


@pytest.fixture
def hub_with_tessera(tmp_path, monkeypatch):
    """A fake Hugging Face with a float16 TESSERA repo of four fires (one broken), tessera.zip and a README."""
    import fake_hub

    monkeypatch.setattr(fake_hub, "ROOT", str(tmp_path / "hub"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    api = fake_hub.HfApi()
    api.create_repo("owner/src")
    for seed, fire_id in enumerate(("1", "2", "3")):
        embedding, coverage = smooth_map(seed)
        path = save_tessera_npz(tmp_path / f"p{fire_id}.npz", embedding, coverage, {"fire_id": fire_id})
        api.upload_file(str(path), f"tessera/p{fire_id}.npz", "owner/src")
    api.upload_file(b"not a zip", "tessera/p4.npz", "owner/src")
    api.upload_file(b"zip", "tessera.zip", "owner/src")
    api.upload_file(b"readme", "README.md", "owner/src")
    fake_hub.CALLS.clear()  # the log starts after this setup
    return fake_hub


def test_convert_repo_works_in_batches_and_keeps_disk_small(hub_with_tessera, tmp_path, monkeypatch):
    """Batches of 2: each upload sees at most 2 staged files and no download left; the source stays unchanged."""
    from firepred.data import tessera_master

    fake_hub = hub_with_tessera
    source_before = {path.name: path.read_bytes() for path in fake_hub._repo("owner/src").rglob("*") if path.is_file()}
    work = tmp_path / "work"
    on_disk = []
    real_upload = tessera_master.upload_staged

    def watched_upload(staging_dir, repo, token):
        """Record what is on disk at each upload, then upload as usual."""
        on_disk.append((sorted(p.name for p in staging_dir.iterdir()), list((work / "temp").rglob("*"))))
        return real_upload(staging_dir, repo, token)

    monkeypatch.setattr(tessera_master, "upload_staged", watched_upload)
    report = tessera_master.convert_tessera_repo("t", "owner/int8", tessera_repo="owner/src", batch_size=2,
                                                 work_dir=work)

    assert report.converted == ("tessera/p1.npz", "tessera/p2.npz", "tessera/p3.npz")
    assert list(report.failed) == ["tessera/p4.npz"]  # the broken file is reported, the run goes on
    assert on_disk == [(["p1.npz", "p2.npz"], []), (["p3.npz"], [])]
    assert 0 < report.bytes_after < 0.6 * report.bytes_before
    dest = fake_hub._repo("owner/int8")
    assert sorted(p.name for p in (dest / "tessera").iterdir()) == ["p1.npz", "p2.npz", "p3.npz"]
    assert not (dest / "tessera.zip").exists() and not (dest / "README.md").exists()
    for fire_id in ("1", "2", "3"):
        path = dest / "tessera" / f"p{fire_id}.npz"
        assert tessera_quantization_of(path) == "int8"
        with open_npz(path) as archive:
            assert str(archive["fire_id"]) == fire_id and str(archive["converted_from"]) == "owner/src"
    assert not any((work / "staging" / "tessera-convert").iterdir()) and not any((work / "temp").iterdir())
    source_after = {path.name: path.read_bytes() for path in fake_hub._repo("owner/src").rglob("*") if path.is_file()}
    assert source_after == source_before

    # A second run converts nothing: every good file is already there, the broken one fails again.
    again = tessera_master.convert_tessera_repo("t", "owner/int8", tessera_repo="owner/src", work_dir=work)
    assert again.converted == () and len(again.already) == 3 and list(again.failed) == ["tessera/p4.npz"]


def test_convert_repo_refuses_a_missing_or_same_destination(hub_with_tessera, tmp_path):
    """The source repo is never the destination, and the settings are checked before anything is downloaded."""
    from firepred.data import tessera_master

    for dest in (None, "", "owner/src"):
        with pytest.raises(ValueError, match="destination"):
            tessera_master.convert_tessera_repo("t", dest, tessera_repo="owner/src", work_dir=tmp_path)
    with pytest.raises(ValueError, match="quantization"):
        tessera_master.convert_tessera_repo("t", "owner/q", tessera_repo="owner/src", quantization="int4")
    with pytest.raises(ValueError, match="batch_size"):
        tessera_master.convert_tessera_repo("t", "owner/q", tessera_repo="owner/src", batch_size=0)
    assert not hub_with_tessera.CALLS  # nothing was listed, created or downloaded


def test_convert_cli_reaches_the_master():
    """`dataset --tessera-repo A tessera-convert --dest-tessera-repo B` converts A into B as int8 by default."""
    from firepred.cli import dataset as dataset_cli

    command = dataset_cli.BY_NAME["tessera-convert"]
    args = dataset_cli.parse_args(["--tessera-repo", "owner/a", "tessera-convert", "--dest-tessera-repo", "owner/b",
                                   "--batch-size", "4"])
    kwargs = dataset_cli.call_arguments(command, args)
    assert (kwargs["tessera_repo"], kwargs["dest_repo"], kwargs["quantization"], kwargs["batch_size"]) == (
        "owner/a", "owner/b", "int8", 4)
    args = dataset_cli.parse_args(["tessera-convert", "--dest-tessera-repo", "b", "--tessera-quantization", "none"])
    assert dataset_cli.call_arguments(command, args)["quantization"] == "none"
