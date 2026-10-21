"""Regression checks for standard dataset comparison."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace

import pytest
import zarr

from datakit.schema.standard_dataset import (
    ConversionChannelRule,
    ConversionValidationSpec,
    EvidenceReference,
)
from pipelines.dataset import validate_standard_dataset


def test_conversion_full_mapping_and_attached_receipt(published):
    from pipelines.dataset import validate_standard_conversion

    path, spec = published
    source = path.parent / "source.zarr"
    shutil.copytree(path, source)
    group = zarr.open_group(str(source), mode="r+")
    group["mean"][2] = 4  # approved effective tp statistics correction
    conversion = ConversionValidationSpec(
        "old-fixture",
        "old-snapshot",
        "rules-1",
        spec.first_time,
        spec.last_time,
        6,
        spec.product.channel_names,
        tuple(
            ConversionChannelRule(n, n, 0, 0, expected_mean=0, expected_std=1)
            for n in spec.product.channel_names
        ),
    )
    receipt = validate_standard_conversion(source, path, spec, conversion)
    assert receipt["comparison_complete"] and all(
        c["status"] == "pass" for c in receipt["checks"].values()
    )
    file = path.parent / "conversion.json"
    file.write_text(json.dumps(receipt))
    confirmed = replace(
        spec,
        conversion_required=True,
        conversion_evidence=EvidenceReference(file, hashlib.sha256(file.read_bytes()).hexdigest()),
    )
    assert validate_standard_dataset(path, confirmed, mode="full").release_passed
    receipt = validate_standard_conversion(source, path, spec, conversion, mode="quick")
    assert not receipt["comparison_complete"]
    group["data"][1, 0, 0, 0] = 5
    receipt = validate_standard_conversion(source, path, spec, conversion)
    assert receipt["checks"]["values"]["status"] == "fail"


def test_combination_requires_full_reports_and_exact_time(published):
    from datakit.validation import check_standard_time_alignment
    from pipelines.dataset import validate_standard_combination

    path, spec = published
    other = path.parent / "other.zarr"
    shutil.copytree(path, other)
    reports = [validate_standard_dataset(p, spec, mode="full").to_dict() for p in (path, other)]
    result = validate_standard_combination(
        [path, other], [spec, spec], individual_reports=reports, shared_arrays=("mask",)
    )
    assert result["passed"] and result["time_alignment"]["lost_counts"] == [0, 0]
    assert not validate_standard_combination([path, other], [spec, spec])["passed"]
    time = spec.expected_times()
    assert not check_standard_time_alignment([time, time[1:]], policy="exact")["passed"]
    result = check_standard_time_alignment([time, time[1:]], policy="intersection")
    assert result["passed"] and result["lost_counts"] == [1, 0]
    with pytest.raises(ValueError):
        check_standard_time_alignment([time, time[::-1]])
