"""Regression checks for standard dataset masks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import zarr

from pipelines.dataset_building.definitions.standard_dataset import (
    STANDARD_LAND_SEA_MASK,
    AuxiliaryDefinition,
    MaskDefinition,
)
from pipelines.dataset_building import validate_standard_dataset

from ._helpers import _checks


def test_external_mask_files_cannot_block_embedded_mask_acceptance(published):
    path, spec = published
    auxiliaries = (
        AuxiliaryDefinition(
            "mask.npy",
            "npy",
            None,
            ("lat", "lon"),
            "bool",
            "external_mask",
            semantics_confirmed=False,
        ),
        AuxiliaryDefinition(
            "mask.nc",
            "netcdf",
            None,
            ("lat", "lon"),
            "float32",
            "external_mask",
            semantics_confirmed=False,
        ),
        AuxiliaryDefinition(
            "land_mask.nc",
            "netcdf",
            None,
            ("lat", "lon"),
            "bool",
            "land_copy",
            equals_mask="land_mask",
        ),
        AuxiliaryDefinition(
            "sst_mask.nc",
            "netcdf",
            None,
            ("lat", "lon"),
            "float32",
            "sst_mask",
            semantics_confirmed=False,
        ),
    )
    spec = replace(
        spec,
        mask_provenance=None,
        product=replace(spec.product, variants=(replace(spec.variant, auxiliaries=auxiliaries),)),
    )
    # Both invalid external files and a missing external file are out of scope.
    for definition in auxiliaries[:-1]:
        (path / definition.filename).write_bytes(b"invalid external mask file; must not be opened")
    full = validate_standard_dataset(path, spec, mode="full")
    assert full.release_passed
    assert _checks(full)["mask.provenance"] == "not-applicable"
    assert all(
        _checks(full)[f"auxiliary.{a.filename}.scope"] == "not-applicable" for a in auxiliaries
    )
    assert not any(
        check.check_id.endswith((".role", ".grid_association", ".mask_equal"))
        for check in full.checks
    )
    # A real defect in the authoritative Zarr mask still fails.
    group = zarr.open_group(str(path), mode="r+")
    group["mask"][0, 0, 0] = 2
    report = validate_standard_dataset(path, spec, mode="full")
    assert not report.release_passed
    assert _checks(report)["mask.binary_complement"] == "fail"


def test_embedded_mask_labels_remain_strict_without_external_masks(published):
    path, spec = published
    group = zarr.open_group(str(path), mode="r+")
    group["mask_channel"][:] = ["sea_mask", "land_mask"]
    report = validate_standard_dataset(path, spec, mode="quick")
    assert _checks(report)["mask.labels"] == "fail"
    assert not report.to_dict()["quick_passed"]


def test_mask_provenance_can_only_be_required_explicitly(published):
    path, spec = published
    report = validate_standard_dataset(
        path,
        replace(
            spec,
            mask_provenance=None,
            product=replace(spec.product, mask=replace(spec.product.mask, require_provenance=True)),
        ),
        mode="full",
    )
    assert _checks(report)["mask.provenance"] == "needs-review"
    assert not report.release_passed


def test_mask_validation_and_quality_follow_the_product_schema(published):
    path, spec = published
    group = zarr.open_group(str(path), mode="r+")
    data = group["data"][:]
    data[:, 0, :, 2:] = np.nan  # outside the declared land region
    group["data"][:] = data
    reversed_masks = group["mask"][:][::-1].astype(bool)
    del group["mask"]
    group.create_array(
        "mask",
        data=reversed_masks,
        chunks=reversed_masks.shape,
        dimension_names=("mask_channel", "lat", "lon"),
    )
    group["mask_channel"][:] = ["sea_mask", "land_mask"]
    zarr.consolidate_metadata(str(path))
    definition = MaskDefinition(
        schema_id="fixture.reversed-mask", channels=("sea_mask", "land_mask"), dtype="bool"
    )
    spec = replace(
        spec,
        product=replace(spec.product, mask=definition),
        quality=(replace(spec.quality[0], valid_region="land"), *spec.quality[1:]),
    )
    report = validate_standard_dataset(path, spec, mode="full")
    assert report.release_passed, report.to_dict()
    assert _checks(report)["layout.mask"] == "pass"
    assert _checks(report)["mask.labels"] == "pass"
    assert _checks(report)["quality.t2m"] == "pass"
    assert report.identity["mask_schema_id"] == "fixture.reversed-mask"
    # The same data still fails the original canonical declaration.
    canonical = replace(spec, product=replace(spec.product, mask=STANDARD_LAND_SEA_MASK))
    assert (
        _checks(validate_standard_dataset(path, canonical, mode="quick"))["layout.mask"] == "fail"
    )
