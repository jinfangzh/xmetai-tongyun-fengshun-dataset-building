from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import legacy_zarr_refresh as refresh
import numpy as np
import pytest
import zarr
from legacy_zarr_refresh import (
    LegacyZarrSource,
    correct_statistics,
    create_refreshed_store,
    load_profiles,
    read_manifest,
    refresh_legacy_zarr,
    select_profiles,
)

from tests.dataset_building.samples.legacy import (
    profile as _profile,
)
from tests.dataset_building.samples.legacy import (
    snapshot_files as _snapshot_files,
)
from tests.dataset_building.samples.legacy import (
    write_legacy_store as _write_legacy_store,
)


def test_profiles_are_external_and_all_selects_every_profile(tmp_path: Path) -> None:
    config = tmp_path / "datasets.json"
    config.write_text(
        json.dumps(
            {
                "datasets": {
                    "sample": {
                        "input": "/external/sample.zarr",
                        "output_name": "sample.refreshed.zarr",
                        "cadence_hours": 6,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    profiles = load_profiles(config)
    assert [profile.key for profile in select_profiles(profiles, ["all"])] == ["sample"]
    assert profiles["sample"].already_refreshed == "error"


def test_statistics_corrections_are_channel_aligned() -> None:
    mean, std = correct_statistics(
        ("q1000", "ssrd"),
        np.asarray([2.0, 4.0], dtype="f4"),
        np.asarray([0.5, 2.0], dtype="f4"),
        {"q1000": 2.0},
        {"ss*": 3.0},
    )
    np.testing.assert_allclose(mean, [1.0, 12.0])
    np.testing.assert_allclose(std, [0.25, 6.0])


def test_statistics_factor_one_is_a_no_op() -> None:
    original_mean = np.asarray([2.0, 4.0], dtype="f4")
    original_std = np.asarray([0.5, 2.0], dtype="f4")
    mean, std = correct_statistics(
        ("q1000", "ssrd"),
        original_mean,
        original_std,
        {"q1000": 1.0},
        {"ss*": 1.0},
    )
    np.testing.assert_array_equal(mean, original_mean)
    np.testing.assert_array_equal(std, original_std)


def test_already_refreshed_source_is_rejected_before_open(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    source.mkdir()
    (source / "refresh_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already contains a refresh manifest"):
        LegacyZarrSource(_profile(source))


def test_refresh_store_uses_zarr_v2_and_dot_chunk_keys(tmp_path: Path) -> None:
    store = tmp_path / "refreshed.zarr"
    create_refreshed_store(
        store,
        dataset_id="synthetic",
        data_revision="test",
        times=np.asarray(["2024-01-01"], dtype="datetime64[D]"),
        channels=("t2m", "q1000"),
        lat=np.asarray([1.0, -1.0]),
        lon=np.asarray([0.0, 180.0]),
        mean=np.asarray([0.0, 0.0]),
        std=np.asarray([1.0, 1.0]),
        land_mask=np.asarray([[True, False], [False, True]]),
        radiation_seconds=None,
    )
    zarr.consolidate_metadata(str(store))

    root = json.loads((store / ".zgroup").read_text(encoding="utf-8"))
    data = json.loads((store / "data" / ".zarray").read_text(encoding="utf-8"))
    assert root["zarr_format"] == 2
    assert data["dimension_separator"] == "."
    assert (store / ".zmetadata").is_file()


def test_invalid_statistics_pattern_is_rejected() -> None:
    with pytest.raises(ValueError, match="matched no channels"):
        correct_statistics(("t2m",), np.asarray([0.0]), np.asarray([1.0]), {}, {"q*": 2.0})


def test_end_to_end_refresh_aligns_channels_latitude_statistics_and_mask(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "legacy.zarr"
    source_values = _write_legacy_store(source_path)
    output_root = tmp_path / "output"

    output, manifest = refresh_legacy_zarr(
        _profile(source_path),
        output_root,
        full_data_check=True,
    )

    group = zarr.open_group(str(output), mode="r")
    assert group["channel"][:].tolist() == ["q1000", "ssrd"]
    np.testing.assert_array_equal(group["lat"][:], [1.0, -1.0])
    np.testing.assert_array_equal(group["lon"][:], [0.0, 180.0])
    np.testing.assert_allclose(group["mean"][:], [1.0, 12.0])
    np.testing.assert_allclose(group["std"][:], [0.25, 6.0])

    expected = source_values[:, [1, 0], ::-1, :].astype("f2")
    np.testing.assert_array_equal(group["data"][:], expected)
    np.testing.assert_array_equal(
        group["mask"][0],
        np.asarray([[1, 0], [0, 1]], dtype="u1"),
    )
    assert manifest["validation"]["status"] == "ok"
    assert manifest["validation"]["full_data_check"] is True
    assert len(manifest["validation"]["checked_data_chunks"]) == 2
    assert (output / "refresh_manifest.json").is_file()
    reopened_manifest = read_manifest(output)
    assert reopened_manifest["validation"]["status"] == "ok"
    assert reopened_manifest["source_indices"] == {
        "start": 0,
        "end": 1,
        "count": 2,
    }


@pytest.mark.parametrize("as_module", [False, True])
def test_single_file_cli_help(tmp_path: Path, as_module: bool) -> None:
    script = Path(refresh.__file__).resolve()
    assert script.name == "legacy_zarr_refresh.py"
    assert not list(script.with_suffix("").glob("*.py"))
    entry = (
        ["-m", "examples.dataset_building.legacy_zarr_refresh"]
        if as_module else [str(script)]
    )
    result = subprocess.run(
        [sys.executable, *entry, "--help"],
        cwd=script.parents[2] if as_module else tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    for command in ("inspect", "pilot", "convert", "validate", "auxiliary"):
        assert command in result.stdout


def test_cli_inspection_and_dry_run_are_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.zarr"
    _write_legacy_store(source)
    config = tmp_path / "datasets.json"
    config.write_text(
        json.dumps({"datasets": {"synthetic": _profile(source).serializable()}}),
        encoding="utf-8",
    )
    before = _snapshot_files(source)
    refresh.main(["--config", str(config), "inspect", "--dataset", "all"])
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["synthetic"]["channels"] == ["q1000", "ssrd"]
    output = tmp_path / "output"
    refresh.main([
        "--config", str(config), "pilot", "--dataset", "synthetic",
        "--output-root", str(output), "--start-index", "1", "--count", "1",
        "--dry-run",
    ])
    plan = json.loads(capsys.readouterr().out)
    assert plan["source_indices"] == {"start": 1, "end": 1, "count": 1}
    assert plan["actions"]["channel_reorder"] is True
    assert plan["actions"]["latitude_flip"] is True
    assert not output.exists()
    assert _snapshot_files(source) == before


def test_single_file_cli_refresh_validate_and_auxiliary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.zarr"
    values = _write_legacy_store(source)
    config = tmp_path / "datasets.json"
    config.write_text(
        json.dumps({"datasets": {"synthetic": _profile(source).serializable()}}),
        encoding="utf-8",
    )
    before = _snapshot_files(source)
    output_root = tmp_path / "output"
    common = ["--config", str(config)]
    selection = ["--dataset", "synthetic", "--output-root", str(output_root)]
    subprocess.run(
        [
            sys.executable, str(Path(refresh.__file__).resolve()), *common,
            "convert", *selection, "--workers", "2", "--full-data-check",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    output = output_root / "synthetic.refreshed.zarr"
    manifest = read_manifest(output)
    assert manifest["validation"]["full_data_check"] is True
    assert len(manifest["validation"]["checked_data_chunks"]) == 2
    group = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(group["data"][:], values[:, [1, 0], ::-1, :].astype("f2"))
    embedded_mask = np.asarray(group["mask"][:])

    refresh.main([
        *common, "validate", "--dataset", "synthetic",
        "--zarr", str(output), "--full-data-check",
    ])
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    target_before = _snapshot_files(output)
    refresh.main([*common, "auxiliary", *selection, "--dry-run", "--strict"])
    assert json.loads(capsys.readouterr().out)[0]["status"] == "planned"
    assert _snapshot_files(output) == target_before
    refresh.main([*common, "auxiliary", *selection, "--strict"])
    assert json.loads(capsys.readouterr().out)[0]["status"] == "ok"
    assert (output / "mask.npy").is_file()
    assert (output / "auxiliary_refresh_manifest.json").is_file()
    np.testing.assert_array_equal(group["mask"][:], embedded_mask)

    refresh.main([
        *common, "pilot", *selection, "--start-index", "1",
        "--count", "1", "--full-data-check",
    ])
    capsys.readouterr()
    pilot = output_root / "synthetic.refreshed.zarr.pilot.1_1"
    assert read_manifest(pilot)["source_indices"] == {"start": 1, "end": 1, "count": 1}
    assert _snapshot_files(source) == before


def test_refresh_resumes_after_interrupted_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.zarr"
    values = _write_legacy_store(source)
    output_root = tmp_path / "output"
    original = LegacyZarrSource.read_data

    def interrupt(self: LegacyZarrSource, index: int) -> np.ndarray:
        if index == 1:
            raise RuntimeError("simulated interrupted source read")
        return original(self, index)

    monkeypatch.setattr(LegacyZarrSource, "read_data", interrupt)
    with pytest.raises(RuntimeError, match="simulated interrupted source read"):
        refresh_legacy_zarr(_profile(source), output_root)
    state = output_root / ".synthetic.refreshed.zarr.refresh_state.json"
    assert json.loads(state.read_text(encoding="utf-8"))["completed"] == 1
    monkeypatch.setattr(LegacyZarrSource, "read_data", original)
    output, manifest = refresh_legacy_zarr(
        _profile(source), output_root, full_data_check=True,
    )
    assert manifest["validation"]["status"] == "ok"
    assert not state.exists()
    assert not (output_root / ".synthetic.refreshed.zarr.staging").exists()
    group = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(group["data"][:], values[:, [1, 0], ::-1, :].astype("f2"))


def external_store(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable} to run this external-data check")
    return Path(value)


def assert_dataset_store(path: Path) -> None:
    assert (path / ".zgroup").is_file(), f"not a Zarr v2 store: {path}"
    group = zarr.open_group(str(path), mode="r")
    assert {
        "data",
        "time",
        "channel",
        "lat",
        "lon",
        "mean",
        "std",
        "mask",
        "mask_channel",
    } <= set(group)
    assert group["data"].ndim == 4
    assert group["data"].shape[1] == group["channel"].shape[0]
    assert group["mean"].shape == group["std"].shape == group["channel"].shape
    assert group["mask"].shape[1:] == group["data"].shape[2:]


@pytest.mark.integration
def test_external_refreshed_store_contract() -> None:
    assert_dataset_store(external_store("XMETAI_REFRESH_TEST_STORE"))
