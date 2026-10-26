"""Known samples exercise the real legacy refresh and sidecar workflow."""

from __future__ import annotations

import json
from dataclasses import replace

import legacy_zarr_refresh as refresh
import numpy as np
import pytest
import xarray as xr
import zarr

from pipelines.dataset_building import validate_standard_dataset

from .samples.legacy import snapshot_files


@pytest.mark.parametrize("workers", [1, 2])
def test_refresh_workflow_matches_known_results(legacy_sample, tmp_path, workers):
    profile, auxiliary_source, expected = legacy_sample
    original = snapshot_files(profile.input)
    original_auxiliary = snapshot_files(auxiliary_source)
    output_root = tmp_path / "output"

    output, manifest = refresh.refresh_legacy_zarr(
        profile,
        output_root,
        workers=workers,
        full_data_check=True,
    )
    group = zarr.open_group(str(output), mode="r")
    assert group["channel"][:].tolist() == expected["channel"]
    for name in ("data", "lat", "lon", "mean", "std"):
        np.testing.assert_array_equal(group[name][:], expected[name])
    np.testing.assert_array_equal(group["time"][:], np.asarray(expected["time"], "datetime64[ns]"))
    np.testing.assert_array_equal(group["mask"][0], expected["land_mask"])
    np.testing.assert_array_equal(group["mask"][1], 1 - np.asarray(expected["land_mask"]))
    assert group["data"].dtype == np.dtype("float16")
    assert group["data"].shape == (2, 2, 2, 2)
    assert group["data"].chunks == (1, 2, 2, 2)
    assert json.loads((output / ".zgroup").read_text(encoding="utf-8"))["zarr_format"] == 2
    assert {file.name for file in (output / "data").iterdir() if not file.name.startswith(".")} == {
        "0.0.0.0",
        "1.0.0.0",
    }
    assert manifest["validation"]["status"] == "ok"
    assert manifest["validation"]["full_data_check"] is True
    assert len(manifest["validation"]["checked_data_chunks"]) == 2
    assert refresh.read_manifest(output) == json.loads(json.dumps(manifest))

    auxiliary = refresh.migrate_auxiliary_files(
        profile.key,
        {"source": str(auxiliary_source), "outputs": [output.name]},
        output_root,
        dry_run=False,
        overwrite=False,
        strict=True,
    )
    assert auxiliary["status"] == "ok"
    assert auxiliary["artifacts"]["mask.npy"]["source_format"] == "netcdf"
    assert auxiliary["artifacts"]["embedded_mask"]["status"] == "preserved"
    np.testing.assert_array_equal(np.load(output / "weight.npy"), expected["weight"])
    np.testing.assert_array_equal(np.load(output / "mask.npy"), expected["external_mask"])
    with xr.open_dataset(output / "const.nc") as const:
        assert const.channel.values.tolist() == expected["const_channels"]
        np.testing.assert_array_equal(const["data"].values, expected["const"])
        np.testing.assert_array_equal(const.lat.values, expected["lat"])
        np.testing.assert_array_equal(const.lon.values, expected["lon"])
    np.testing.assert_array_equal(group["mask"][0], expected["land_mask"])
    assert (output / "auxiliary_refresh_manifest.json").is_file()
    assert snapshot_files(profile.input) == original
    assert snapshot_files(auxiliary_source) == original_auxiliary
    assert not (output_root / ".synthetic.refreshed.zarr.staging").exists()
    assert not (output_root / ".synthetic.refreshed.zarr.refresh_state.json").exists()


def test_interrupted_refresh_resumes_only_unfinished_frames(legacy_sample, tmp_path, monkeypatch):
    profile, _, expected = legacy_sample
    original_files = snapshot_files(profile.input)
    read_data = refresh.LegacyZarrSource.read_data

    def interrupt(source, index):
        if index == 1:
            raise RuntimeError("test interruption after one completed frame")
        return read_data(source, index)

    monkeypatch.setattr(refresh.LegacyZarrSource, "read_data", interrupt)
    output_root = tmp_path / "output"
    with pytest.raises(RuntimeError, match="test interruption after one completed frame"):
        refresh.refresh_legacy_zarr(profile, output_root)
    state = output_root / ".synthetic.refreshed.zarr.refresh_state.json"
    assert json.loads(state.read_text(encoding="utf-8"))["completed"] == 1
    assert not (output_root / profile.output_name).exists()
    assert snapshot_files(profile.input) == original_files

    resumed_reads = []

    def record_read(source, index):
        resumed_reads.append(index)
        return read_data(source, index)

    monkeypatch.setattr(refresh.LegacyZarrSource, "read_data", record_read)
    output, manifest = refresh.refresh_legacy_zarr(profile, output_root, full_data_check=True)
    # Final verification also rereads the source; only the write phase must skip frame zero.
    assert resumed_reads == [1, 0, 1]
    group = zarr.open_group(str(output), mode="r")
    for name in ("data", "mean", "std"):
        np.testing.assert_array_equal(group[name][:], expected[name])
    assert manifest["validation"]["status"] == "ok"
    assert not state.exists()
    assert snapshot_files(profile.input) == original_files


def test_refreshed_output_is_not_silently_processed_twice(legacy_sample, tmp_path):
    profile, _, _ = legacy_sample
    output, _ = refresh.refresh_legacy_zarr(profile, tmp_path / "output")
    original = snapshot_files(output)
    with pytest.raises(ValueError, match="already contains a refresh manifest"):
        refresh.refresh_legacy_zarr(replace(profile, input=output), tmp_path / "second")
    assert not (tmp_path / "second").exists()
    assert snapshot_files(output) == original


def test_current_refresh_output_is_rejected_by_v3_standard(
    legacy_sample, published_schema, tmp_path
):
    profile, _, _ = legacy_sample
    _, spec, schema, _ = published_schema
    frozen_yaml = schema.read_bytes()
    output, _ = refresh.refresh_legacy_zarr(profile, tmp_path / "output", full_data_check=True)
    before = snapshot_files(output)

    report = validate_standard_dataset(output, spec, mode="full")

    checks = {check.check_id: check for check in report.checks}
    assert checks["storage.zarr_format"].status == "fail"
    assert checks["storage.zarr_format"].expected == 3
    assert checks["storage.zarr_format"].observed == 2
    assert not report.release_passed
    assert snapshot_files(output) == before
    assert schema.read_bytes() == frozen_yaml
