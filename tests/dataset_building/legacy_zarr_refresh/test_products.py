"""Ten product orders and semantics survive a real v2-to-v3 refresh."""

from __future__ import annotations

import json
from pathlib import Path

import legacy_zarr_refresh as refresh
import numpy as np
import pytest
import xarray as xr
import zarr

from datakit.schema.datasets import load_schema
from pipelines.dataset_building import validate_standard_dataset
from pipelines.dataset_building.support.inputs import load_validation_schema
from tests.dataset_building.samples.legacy import snapshot_files, write_product_sample

ROOT = Path(__file__).resolve().parents[3]
SCHEMAS = tuple(sorted((ROOT / "datakit/schema/datasets").glob("*.yaml")))


@pytest.mark.parametrize("standard", SCHEMAS, ids=lambda path: path.stem)
def test_each_product_refresh_matches_frozen_order_and_passes_quick(tmp_path, standard):
    frozen_yaml = standard.read_bytes()
    profile, schema, expected = write_product_sample(tmp_path, standard)
    frozen_sample = profile.target_schema.read_bytes()
    before = snapshot_files(profile.input)
    baseline = json.loads((ROOT / "docs/validation/PRODUCT_BASELINE_2026-10-13.json").read_text())
    observed = next(item for item in baseline if item["dataset_id"] == schema.dataset_id)
    assert schema.product.channel_names == tuple(observed["channels"])

    output, manifest = refresh.refresh_legacy_zarr(
        profile, tmp_path / "output", full_data_check=True
    )
    group = zarr.open_group(str(output), mode="r")
    assert group["channel"][:].tolist() == observed["channels"]
    for name in ("data", "time", "lat", "lon", "mean", "std"):
        np.testing.assert_array_equal(group[name][:], expected[name])
    np.testing.assert_array_equal(group["mask"][0], expected["mask"])
    assert group.attrs["dataset_id"] == schema.dataset_id != output.name
    assert group.attrs["data_revision"] == schema.data_revision
    assert manifest["target"]["schema_sha256"] == schema.schema_sha256
    assert group["data"].chunks == (1, len(observed["channels"]), 3, 4)
    assert bool((np.diff(group["lat"][:]) > 0).all()) == (schema.product.product_id == "cra.c71")
    for definition in schema.product.channels:
        info = group["channel"].attrs["channel_info"][definition.name]
        assert all(info[key] == value for key, value in definition.semantic_fields().items())
    if schema.product.product_id == "oras5.c88":
        assert group["channel"].attrs["channel_info"]["os19978"]["units"] is None
        assert group["channel"].attrs["channel_info"]["os19978"]["level"] == 199.78

    migrated = refresh.migrate_auxiliary_files(
        profile.key,
        {
            "source": str(profile.input),
            "outputs": [output.name],
            "target_schema": str(profile.target_schema),
        },
        output.parent,
        dry_run=False,
        overwrite=False,
        strict=True,
    )
    assert migrated["status"] == "ok"
    weight = next(item for item in schema.variant.auxiliaries if item.role == "weight")
    if weight.format == "netcdf":
        assert not (output / "weight.npy").exists()
        with xr.open_dataset(output / weight.filename) as dataset:
            np.testing.assert_array_equal(next(iter(dataset.data_vars.values())).values, [1, 2, 3])
            np.testing.assert_array_equal(dataset.lat.values, expected["lat"])
    else:
        np.testing.assert_array_equal(
            np.load(output / weight.filename), np.array([1, 2, 3], dtype="f4").reshape(1, 3, 1)
        )

    spec, _ = load_validation_schema(profile.target_schema)
    frozen_output = snapshot_files(output)
    quick = validate_standard_dataset(output, spec, mode="quick")
    assert quick.quick_passed, [
        (check.check_id, check.observed)
        for check in quick.checks
        if check.check_id in quick.quick_blocking_checks
    ]
    full = validate_standard_dataset(output, spec, mode="full")
    assert full.coverage["value_scan_complete"]
    assert full.coverage["storage_audit_complete"]
    assert not [check for check in full.checks if check.status == "fail"]
    # No production receipt is manufactured by the converter or the test.
    assert not full.release_passed
    assert snapshot_files(profile.input) == before
    assert snapshot_files(output) == frozen_output
    assert standard.read_bytes() == frozen_yaml
    assert profile.target_schema.read_bytes() == frozen_sample


def test_example_selects_all_ten_packaged_standards():
    profiles = refresh.load_profiles(
        ROOT / "examples/dataset_building/configs/legacy_zarr_refresh.example.json"
    )
    assert set(profiles) == {path.stem for path in SCHEMAS}
    assert len(profiles) == 10
    for profile in profiles.values():
        schema, _ = load_schema(profile.target_schema)
        assert profile.output_name == schema.directory_name
        assert profile.expected_channels == len(schema.product.channels)
        assert profile.expected_grid == schema.variant.grid.shape


def test_invalid_precipitation_effective_statistics_are_not_relabelled(tmp_path):
    standard = next(path for path in SCHEMAS if path.stem == "era5_c78_p25_h6")
    profile, _, _ = write_product_sample(tmp_path, standard)
    group = zarr.open_group(str(profile.input), mode="r")
    index = group["level"][:].tolist().index("tp")
    mean = np.load(profile.input / "mean.npy")
    mean[index] = 12
    np.save(profile.input / "mean.npy", mean)
    with pytest.raises(ValueError, match="effective values: tp"):
        refresh.refresh_legacy_zarr(profile, tmp_path / "output")
    assert not (tmp_path / "output").exists()
