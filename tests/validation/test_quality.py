"""Regression checks for standard dataset quality."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import zarr

from datakit.schema.standard_dataset import EvidenceReference
from pipelines.dataset import validate_standard_dataset

from ._helpers import _checks


def test_masked_quality_and_unclipped_physical_inverse(published):
    path, spec = published
    group = zarr.open_group(str(path), mode="r+")
    group["data"][1, 0, :, 2:] = np.nan  # outside land domain
    rules = (replace(spec.quality[0], valid_region="land"), *spec.quality[1:])
    spec = replace(spec, quality=rules)
    assert validate_standard_dataset(path, spec, mode="full").release_passed
    group["data"][1, 2, 0, 0] = 10
    rules = (*rules[:2], replace(rules[2], domain="physical", bounds=(0, 2)))
    report = validate_standard_dataset(path, replace(spec, quality=rules), mode="full")
    assert _checks(report)["quality.tp"] == "fail"  # no core clip(0,7) hiding error
    assert report.quality["tp"]["maximum"] > 20


def test_tp_processing_domain_is_required_even_without_optional_bounds(published):
    path, spec = published
    group = zarr.open_group(str(path), mode="r+")
    group["data"][1, 2, 0, 0] = -0.5
    rules = tuple(replace(rule, bounds=None) for rule in spec.quality)
    report = validate_standard_dataset(path, replace(spec, quality=rules), mode="full")
    assert _checks(report)["quality.tp.processing_domain"] == "fail"
    assert not report.release_passed


def test_bad_missing_frame_cannot_be_diluted_by_global_rate(published):
    path, spec = published
    group = zarr.open_group(str(path), mode="r+")
    group["data"][1, 0, :, :] = np.nan
    rules = (replace(spec.quality[0], max_nan_fraction=0.1), *spec.quality[1:])
    report = validate_standard_dataset(path, replace(spec, quality=rules), mode="full")
    assert report.quality["t2m"]["nan_fraction"] < 0.1
    assert report.quality["t2m"]["maximum_block_nan_fraction"] == 1
    assert _checks(report)["quality.t2m"] == "fail"
    permitted = (replace(rules[0], max_block_nan_fraction=1.0), *rules[1:])
    assert validate_standard_dataset(
        path, replace(spec, quality=permitted), mode="full"
    ).release_passed


def test_custom_region_requires_frozen_raw_mask_and_binding(published):
    path, spec = published
    mask = np.zeros((3, 4), dtype="bool")
    mask[:, 0] = True
    file = path.parent / "approved-wet-region.npy"
    np.save(file, mask)
    reference = EvidenceReference(file, hashlib.sha256(file.read_bytes()).hexdigest())
    rules = (replace(spec.quality[0], valid_region="custom:wet-level"), *spec.quality[1:])
    spec = replace(
        spec,
        quality=rules,
        valid_regions={"wet-level": reference},
        valid_region_provenance={"wet-level": "independent frozen grid/axis/depth mapping"},
    )
    group = zarr.open_group(str(path), mode="r+")
    group["data"][:, 0, :, 1:] = np.nan
    report = validate_standard_dataset(path, spec, mode="full")
    assert report.release_passed, report.to_dict()
    assert report.quality["t2m"]["count"] == 36 and report.quality["t2m"]["nan"] == 0

    def mutate(_entry):
        # In-place bytes work on Windows too: truncating a mapped file is
        # rejected there before its contents change.
        with file.open("r+b") as stream:
            stream.seek(-mask.size, 2)
            stream.write(b"\x01" * mask.size)

    report = validate_standard_dataset(path, spec, mode="full", on_block=mutate)
    assert not report.release_passed and not report.coverage["scan_complete"]
    assert _checks(report)["metadata.valid_region.wet-level.unchanged"] == "fail"


def test_successful_reads_without_quality_policy_are_not_full_completion(published):
    path, spec = published
    report = validate_standard_dataset(path, replace(spec, quality=()), mode="full")
    assert report.coverage["value_checked"] == 12 and report.coverage["physically_present"] == 12
    assert not report.coverage["quality_checks_complete"] and not report.coverage["scan_complete"]


def test_badly_typed_quality_settings_cannot_silently_relax_validation(published):
    _path, spec = published
    for fields in (
        {"allow_constant": "false"},
        {"max_nan_fraction": True},
        {"max_block_nan_fraction": True},
        {"bounds": (0,)},
        {"bounds": (False, 10)},
    ):
        with pytest.raises(ValueError):
            replace(spec.quality[0], **fields)
