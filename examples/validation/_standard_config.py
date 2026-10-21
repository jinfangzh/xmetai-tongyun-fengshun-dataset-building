"""Example-only JSON configuration; the datakit library accepts typed specs."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import Any

from datakit.schema import (
    DatasetValidationSpec,
    EvidenceReference,
    QualityRule,
    get_dataset_product,
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def load_profile(path: Path) -> tuple[DatasetValidationSpec, dict[str, Any]]:
    document = read_json(path)
    allowed = {f.name for f in fields(DatasetValidationSpec)} - {
        "product",
        "first_time",
        "last_time",
        "variant_name",
    }
    allowed |= {
        "product_id",
        "product_version",
        "variant_id",
        "expected_time",
        "dataset_path",
        "auxiliary_directory",
        "profile_status",
        "notes",
    }
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"Unrecognized profile fields (possible typo): {sorted(unknown)}")
    product = get_dataset_product(document["product_id"], document["product_version"])
    value = {
        k: v for k, v in document.items() if k in {f.name for f in fields(DatasetValidationSpec)}
    }
    value.update(
        product=product,
        variant_name=document["variant_id"],
        first_time=document["expected_time"]["first"],
        last_time=document["expected_time"]["last"],
    )
    value["quality"] = tuple(
        QualityRule(
            **{**rule, "bounds": tuple(rule["bounds"]) if rule.get("bounds") is not None else None}
        )
        for rule in document.get("quality", [])
    )
    value["allowed_omitted"] = tuple(tuple(i) for i in document.get("allowed_omitted", []))
    if "additional_metadata_files" in value:
        value["additional_metadata_files"] = tuple(value["additional_metadata_files"])
    for key in ("production_evidence", "consumer_evidence", "conversion_evidence"):
        evidence = document.get(key)
        if evidence:
            evidence_path = Path(evidence["path"])
            if not evidence_path.is_absolute():
                evidence_path = path.parent / evidence_path
            value[key] = EvidenceReference(evidence_path, evidence["sha256"])
    regions = {}
    for name, reference in document.get("valid_regions", {}).items():
        source = Path(reference["path"])
        if not source.is_absolute():
            source = path.parent / source
        regions[name] = EvidenceReference(source, reference["sha256"])
    value["valid_regions"] = regions
    spec = DatasetValidationSpec(**value)
    spec.validate_configuration()
    if document["expected_time"].get("count") != spec.expected_shape[0]:
        raise ValueError("Frozen expected count differs from independently generated time coverage")
    return spec, document
