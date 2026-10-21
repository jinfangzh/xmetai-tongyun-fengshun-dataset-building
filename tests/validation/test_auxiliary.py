"""Regression checks for standard dataset auxiliary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import zarr

from datakit.schema.standard_dataset import AuxiliaryDefinition
from pipelines.dataset import validate_standard_dataset

from ._helpers import _checks


def test_auxiliary_roles_missing_files_and_raw_coordinates(published):
    import xarray as xr

    path, spec = published
    weight = AuxiliaryDefinition(
        "weight.npy", "npy", None, ("singleton", "lat", "singleton"), "float32", "weight"
    )
    const = AuxiliaryDefinition(
        "const.nc",
        "netcdf",
        "z",
        ("channel", "lat", "lon"),
        "float32",
        "const",
        ("z", "lsm"),
        check_variable_name=True,
    )
    variant = replace(spec.variant, auxiliaries=(weight, const))
    spec = replace(spec, product=replace(spec.product, variants=(variant,)))
    report = validate_standard_dataset(path, spec, mode="full")
    assert _checks(report)["auxiliary.weight.npy.exists"] == "fail"
    np.save(path / "weight.npy", np.ones((1, 3, 1), dtype="f4"))
    values = np.ones((2, 3, 4), dtype="f4")
    coords = {
        "channel": ["z", "lsm"],
        "lat": np.array([90, 88.5, 87], dtype="f4"),
        "lon": np.array([0, 1.5, 3, 4.5], dtype="f4"),
    }
    xr.Dataset({"wrong": (("channel", "lat", "lon"), values)}, coords=coords).to_netcdf(
        path / "const.nc"
    )
    report = validate_standard_dataset(path, spec, mode="full")
    assert _checks(report)["auxiliary.weight.npy.grid_association"] == "needs-review"
    assert _checks(report)["auxiliary.const.nc.variable"] == "fail"
    xr.Dataset({"z": (("channel", "lat", "lon"), values)}, coords=coords).to_netcdf(
        path / "const.nc", mode="w"
    )
    spec = replace(spec, auxiliary_provenance={"weight.npy": "approved grid/axis/source linkage"})
    assert validate_standard_dataset(path, spec, mode="full").release_passed
    coords["lat"] = coords["lat"][::-1]
    xr.Dataset({"z": (("channel", "lat", "lon"), values)}, coords=coords).to_netcdf(
        path / "const.nc", mode="w"
    )
    report = validate_standard_dataset(path, spec, mode="full")
    assert _checks(report)["auxiliary.const.nc.lat"] == "fail"


def test_auxiliary_checks_survive_channel_failure_and_precision_is_explicit(published):
    path, spec = published
    group = zarr.open_group(str(path), mode="r+")
    group["channel"][:] = ["tp", "q1000", "t2m"]
    (path / "unapproved.nc").write_bytes(b"not a declared auxiliary")
    report = validate_standard_dataset(path, spec, mode="full")
    assert _checks(report)["auxiliary.declared_files"] == "warning"
    assert report.coverage["value_checked"] == 0
    (path / "unapproved.nc").unlink()
    group["channel"][:] = ["t2m", "q1000", "tp"]
    definition = AuxiliaryDefinition(
        "weight.npy", "npy", None, ("singleton", "lat", "singleton"), "float32", "weight"
    )
    spec = replace(
        spec,
        product=replace(spec.product, variants=(replace(spec.variant, auxiliaries=(definition,)),)),
        auxiliary_provenance={"weight.npy": "independently approved weight grid and precision"},
    )
    weights = np.ones((1, 3, 1), dtype="f4")
    weights[0, 0, 0] = -1e-8
    np.save(path / "weight.npy", weights)
    assert (
        _checks(validate_standard_dataset(path, spec, mode="full"))[
            "auxiliary.weight.npy.weight_nonnegative"
        ]
        == "fail"
    )
    approved = replace(spec, auxiliary_tolerances={"weight.npy": 1e-7})
    assert validate_standard_dataset(path, approved, mode="full").release_passed
    assert np.load(path / "weight.npy")[0, 0, 0] < 0  # no silent clipping/repair


def test_unregistered_auxiliaries_warn_without_blocking_release(published):
    path, spec = published
    (path / "extra.nc").write_bytes(b"unregistered file is not consumed as an approved role")
    full = validate_standard_dataset(path, spec, mode="full")
    assert full.release_passed
    assert _checks(full)["layout.declared_root_objects"] == "warning"
    assert _checks(full)["auxiliary.declared_files"] == "warning"
    document = full.to_dict()
    assert len(document["warnings"]) == 2
    assert "auxiliary.declared_files" not in document["blocking_checks"]
    assert validate_standard_dataset(path, spec, mode="quick").to_dict()["quick_passed"]
    (path / "unexpected.txt").write_text("not an NC/NPY auxiliary")
    assert (
        _checks(validate_standard_dataset(path, spec, mode="full"))["layout.declared_root_objects"]
        == "fail"
    )


def test_auxiliary_variable_name_is_deferred_but_layout_is_checked(published):
    import xarray as xr

    path, spec = published
    definition = AuxiliaryDefinition(
        "weight.nc", "netcdf", None, ("lat",), "float32", "weight", weight_nonnegative_atol=1e-7
    )
    spec = replace(
        spec,
        product=replace(spec.product, variants=(replace(spec.variant, auxiliaries=(definition,)),)),
    )
    coords = {"lat": spec.variant.grid.coordinates()[0]}
    xr.Dataset({"any_product_name": (("lat",), np.ones(3, dtype="f4"))}, coords=coords).to_netcdf(
        path / "weight.nc"
    )
    full = validate_standard_dataset(path, spec, mode="full")
    assert full.release_passed
    assert _checks(full)["auxiliary.weight.nc.variable_name"] == "not-applicable"
    assert _checks(full)["auxiliary.weight.nc.header_layout"] == "pass"
    xr.Dataset({"other_name": (("lat",), np.ones(3, dtype="f8"))}, coords=coords).to_netcdf(
        path / "weight.nc", mode="w"
    )
    assert (
        _checks(validate_standard_dataset(path, spec, mode="full"))[
            "auxiliary.weight.nc.header_layout"
        ]
        == "fail"
    )


def test_ambiguous_auxiliary_variables_are_not_silently_selected(published):
    import xarray as xr

    path, spec = published
    definition = AuxiliaryDefinition("weight.nc", "netcdf", None, ("lat",), "float32", "weight")
    spec = replace(
        spec,
        product=replace(spec.product, variants=(replace(spec.variant, auxiliaries=(definition,)),)),
    )
    xr.Dataset(
        {name: (("lat",), np.ones(3, dtype="f4")) for name in ("first", "second")},
        coords={"lat": spec.variant.grid.coordinates()[0]},
    ).to_netcdf(path / "weight.nc")
    full = validate_standard_dataset(path, spec, mode="full")
    assert _checks(full)["auxiliary.weight.nc.variable_selection"] == "warning"
    assert "auxiliary.weight.nc.weight_nonnegative" not in _checks(full)


@pytest.mark.parametrize(
    "minimum,passes", [(-1e-8, True), (-1e-7, True), (-1.01e-7, False), (-1e-4, False)]
)
def test_product_weight_tolerance_has_a_real_failure_boundary(published, minimum, passes):
    from datakit.schema._standard_products import WEIGHT_NPY

    path, spec = published
    spec = replace(
        spec,
        product=replace(spec.product, variants=(replace(spec.variant, auxiliaries=(WEIGHT_NPY,)),)),
        auxiliary_provenance={"weight.npy": "independently approved fixture grid"},
    )
    values = np.ones((1, 3, 1), dtype="f4")
    values[0, 0, 0] = minimum
    np.save(path / "weight.npy", values)
    full = validate_standard_dataset(path, spec, mode="full")
    assert (_checks(full)["auxiliary.weight.npy.weight_nonnegative"] == "pass") == passes
    assert full.release_passed == passes
    assert np.array_equal(np.load(path / "weight.npy"), values)
    if passes:
        assert _checks(full)["auxiliary.weight.npy.weight_roundoff"] == "warning"


def test_polar_float32_roundoff_is_reproduced_without_clipping():
    latitude = np.linspace(-90, 90, 721, dtype="f4")
    weights = np.cos(np.deg2rad(latitude))
    weights /= weights.mean()
    assert np.flatnonzero(weights < 0).tolist() == [0, 720]
    assert -1e-7 <= float(weights.min()) < 0
    accurate = np.cos(np.deg2rad(latitude.astype("f8")))
    accurate /= accurate.mean()
    assert (accurate >= 0).all()
