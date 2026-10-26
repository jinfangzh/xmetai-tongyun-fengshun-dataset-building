"""Regression checks for standard dataset layout."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import zarr

from pipelines.dataset_building import validate_standard_dataset

from ._helpers import _assert_counts, _checks


@pytest.mark.parametrize(
    "fault",
    [
        "channel_order",
        "level_alias",
        "latitude_direction",
        "mask_nonbinary",
        "std_inf",
        "time_missing_first",
        "processing",
        "unknown_unit",
        "boolean_coefficient",
    ],
)
def test_strict_contract_rejects_compatibility_shortcuts(published, fault):
    path, spec = published
    group = zarr.open_group(str(path), mode="r+")
    if fault == "channel_order":
        group["channel"][:] = ["tp", "q1000", "t2m"]
    elif fault == "level_alias":
        raw = json.loads((path / "data" / "zarr.json").read_text())
        raw["dimension_names"][1] = "level"
        (path / "data" / "zarr.json").write_text(json.dumps(raw))
    elif fault == "latitude_direction":
        group["lat"][:] = [87, 88.5, 90]
    elif fault == "mask_nonbinary":
        group["mask"][0, 0, 0] = 2
    elif fault == "std_inf":
        group["std"][0] = np.inf
    elif fault == "time_missing_first":
        group["time"][0] = np.datetime64("2024-02-28T06", "ns")
    else:
        attrs = dict(group["channel"].attrs)
        info = attrs["channel_info"]
        if fault == "processing":
            info["q1000"]["preprocessing"] = [{"operation": "zscore"}]
        elif fault == "boolean_coefficient":
            info["tp"]["add_offset"] = False
        else:
            info["q1000"]["units"] = "unknown"
        group["channel"].attrs.update(attrs)
    zarr.consolidate_metadata(str(path))
    full = validate_standard_dataset(path, spec, mode="full")
    assert not full.release_passed
    assert any(c.status == "fail" for c in full.checks)
    _assert_counts(full)


def test_south_to_north_grid_is_explicit_and_range_remains_min_max(published):
    path, spec = published
    ascending = replace(spec.variant.grid, latitude_direction="south-to-north")
    spec = replace(
        spec, product=replace(spec.product, variants=(replace(spec.variant, grid=ascending),))
    )
    group = zarr.open_group(str(path), mode="r+")
    group["lat"][:] = ascending.coordinates()[0]
    zarr.consolidate_metadata(str(path))
    full = validate_standard_dataset(path, spec, mode="full")
    assert full.release_passed
    assert _checks(full)["attributes.root.geospatial_lat_range"] == "pass"
    wrong = replace(
        spec,
        product=replace(
            spec.product,
            variants=(
                replace(spec.variant, grid=replace(ascending, latitude_direction="north-to-south")),
            ),
        ),
    )
    assert (
        _checks(validate_standard_dataset(path, wrong, mode="quick"))["coordinates.lat.complete"]
        == "fail"
    )
    with pytest.raises(ValueError):
        replace(ascending, latitude_direction="unspecified")
