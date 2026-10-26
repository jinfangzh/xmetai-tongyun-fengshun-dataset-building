"""Regression checks for standard dataset comparison."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import replace

import pytest
import zarr

from datakit.schema.datasets.specifications import EvidenceReference
from pipelines.dataset_building import validate_standard_dataset
from pipelines.dataset_building.support.specifications import (
    ConversionChannelRule,
    ConversionValidationSpec,
)


@pytest.mark.parametrize("fixture_name", ["published", "published_schema"])
def test_conversion_full_mapping_and_attached_receipt(request, fixture_name):
    from pipelines.dataset_building import validate_standard_conversion

    path, spec = request.getfixturevalue(fixture_name)[:2]
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
    for field in ("config_id", "template_version", "schema_sha256"):
        stale = deepcopy(receipt)
        stale[field] = "different-standard"
        file.write_text(json.dumps(stale))
        mismatched = replace(
            confirmed,
            conversion_evidence=EvidenceReference(
                file, hashlib.sha256(file.read_bytes()).hexdigest(),
            ),
        )
        report = validate_standard_dataset(path, mismatched, mode="full")
        assert not report.release_passed
        assert any(
            item.check_id == "conversion_fidelity.identity" and item.status == "fail"
            for item in report.checks
        )
    receipt = validate_standard_conversion(source, path, spec, conversion, mode="quick")
    assert not receipt["comparison_complete"]
    group["data"][1, 0, 0, 0] = 5
    receipt = validate_standard_conversion(source, path, spec, conversion)
    assert receipt["checks"]["values"]["status"] == "fail"


@pytest.mark.parametrize("during_scan", [False, True])
def test_comparison_rejects_a_yaml_changed_after_loading(published_schema, during_scan):
    from pipelines.dataset_building import validate_standard_conversion

    path, spec, schema, _ = published_schema
    other = path.parent / "other.zarr"
    shutil.copytree(path, other)
    conversion = ConversionValidationSpec(
        "fixture", "snapshot", "rules-1", spec.first_time, spec.last_time, 6,
        spec.product.channel_names,
        tuple(ConversionChannelRule(n, n, 0, 0) for n in spec.product.channel_names),
    )
    def change_schema(_block=None):
        with schema.open("a", encoding="utf-8") as stream:
            stream.write("\n# changed after loading\n")

    if not during_scan:
        change_schema()
    with pytest.raises(ValueError, match="changed"):
        validate_standard_conversion(
            other, path, spec, conversion, on_block=change_schema if during_scan else None,
        )
