"""Publication policy and production/conversion evidence checks."""

from __future__ import annotations

from typing import Any

import xarray as xr

from datakit.schema.specifications import AuxiliaryDefinition, DatasetSchema

from .configuration import DatasetValidationSpec
from .reports import ValidationReport

STANDARD = "standard_conformance"
INTEGRITY = "data_integrity"
CONVERSION_CHECKS = frozenset(
    {"mapping", "coordinates", "channels", "statistics", "values", "coverage"}
)


def check_dataset_identity(
    root: str,
    spec: DatasetSchema,
    report: ValidationReport,
) -> None:
    report.add(
        "publication.schema_approval",
        "pass" if spec.schema_status == "approved" else "needs-review",
        STANDARD,
        "approved",
        spec.schema_status,
        "Approval applies to the declared rules, not to the data being inspected",
    )
    if spec.directory_name:
        report.require(
            "publication.directory_name",
            root == spec.directory_name,
            STANDARD,
            spec.directory_name,
            root,
        )
        report.add(
            "publication.identity_mapping",
            "pass"
            if spec.directory_name == spec.dataset_id or spec.identity_mapping_evidence
            else "needs-review",
            STANDARD,
            spec.dataset_id,
            spec.directory_name,
            spec.identity_mapping_evidence,
        )


def check_mask_provenance(spec: DatasetSchema, report: ValidationReport) -> None:
    mask_schema = spec.product.mask
    report.add(
        "mask.provenance",
        (
            ("pass" if spec.mask_provenance else "needs-review")
            if mask_schema.require_provenance
            else "not-applicable"
        ),
        STANDARD,
        evidence=(
            spec.mask_provenance
            if mask_schema.require_provenance
            else "Acceptance uses the embedded Zarr mask layout, labels, binary values and complement; source derivation is outside this scope"
        ),
    )


def check_auxiliary_role(
    definition: AuxiliaryDefinition,
    spec: DatasetSchema,
    report: ValidationReport,
) -> None:
    confirmed = definition.semantics_confirmed or bool(
        spec.confirmed_optional_roles.get(definition.filename)
    )
    report.add(
        f"auxiliary.{definition.filename}.role",
        "pass" if confirmed else "needs-review",
        STANDARD,
        definition.role,
        spec.confirmed_optional_roles.get(definition.filename),
    )


def check_production_receipt(
    spec: DatasetValidationSpec,
    report: ValidationReport,
    receipt: dict[str, Any] | None,
    observed_hash: str | None = None,
) -> bool:
    reference = spec.production_evidence
    if reference is None or not spec.snapshot_id:
        report.add(
            "publication.frozen_snapshot",
            "not-run"
            if report.mode == "quick" and not spec.allowed_omitted
            else "needs-review",
            "publication_readiness",
            evidence="Require independently frozen production receipt and stopped writes/immutable snapshot",
        )
        return False
    if receipt is None:
        raise ValueError("Production receipt contents are required")
    expected = {
        "dataset_id": spec.dataset_id,
        "data_revision": spec.data_revision,
        "snapshot_id": spec.snapshot_id,
        "shape": list(spec.expected_shape),
        "logical_blocks_expected": spec.expected_shape[0],
        "complete": True,
        "allowed_omitted": [list(index) for index in spec.allowed_omitted],
    }
    good = all(
        receipt.get(key) == value for key, value in expected.items()
    ) and receipt.get("stability") in ("immutable-snapshot", "writes-stopped")
    return report.require(
        "publication.frozen_snapshot",
        good,
        "publication_readiness",
        expected,
        receipt,
        {
            "path": str(reference.path),
            "sha256": observed_hash,
            "limitation": "Producer attests stability; metadata digest alone cannot prove absence of concurrent value writes",
        },
    )


def attach_conversion_evidence(
    report: ValidationReport,
    evidence: dict[str, Any],
) -> None:
    """Check a previously read receipt, not the underlying source/target data."""
    category = "conversion_fidelity"
    required = CONVERSION_CHECKS
    expected = {
        "dataset_id": report.dataset_id,
        "data_revision": report.identity["data_revision"],
        "snapshot_id": report.identity["snapshot_id"],
        "product_id": report.identity["product_id"],
        "product_version": report.identity["product_version"],
        "config_id": report.identity["config_id"],
        "template_version": report.identity["template_version"],
        "schema_sha256": report.identity["schema_sha256"],
    }
    checks = evidence.get("checks", {})
    bound = bool(expected["snapshot_id"]) and all(
        evidence.get(k) == v for k, v in expected.items()
    )
    report.require(
        f"{category}.identity",
        bound,
        category,
        expected,
        {k: evidence.get(k) for k in expected},
    )
    if not bound:
        return
    report.require(
        f"{category}.check_coverage",
        isinstance(checks, dict) and set(checks) == required,
        category,
        sorted(required),
        sorted(checks) if isinstance(checks, dict) else checks,
    )
    if not isinstance(checks, dict):
        return
    metadata_fields = ("source_identity", "rules_version", "mapping")
    report.require(
        f"{category}.provenance",
        all(evidence.get(k) for k in metadata_fields),
        category,
        metadata_fields,
        {k: evidence.get(k) for k in metadata_fields},
    )
    report.require(
        "conversion_fidelity.full_comparison",
        evidence.get("comparison_complete") is True,
        category,
        True,
        evidence.get("comparison_complete"),
    )
    for name in sorted(required):
        item = checks.get(name)
        if not isinstance(item, dict):
            report.add(f"{category}.{name}", "not-run", category)
            continue
        status = item.get("status")
        if status not in ("pass", "fail", "needs-review", "blocked", "not-run"):
            status = "needs-review"
        if status == "pass" and not item.get("evidence"):
            status = "needs-review"
        report.add(
            f"{category}.{name}",
            status,
            category,
            item.get("expected"),
            item.get("observed"),
            item.get("evidence"),
        )


def check_product_confirmation(spec: DatasetSchema, report: ValidationReport) -> None:
    semantic_status = "pass"
    if spec.product.semantic_review:
        if spec.validation_scope == "confirmed-format":
            semantic_status = "not-applicable"
        elif not spec.semantic_confirmation:
            semantic_status = "needs-review"
    report.add(
        "semantics.product_confirmation",
        semantic_status,
        STANDARD,
        spec.product.semantic_review,
        evidence=spec.semantic_confirmation,
    )


def check_auxiliary_provenance(obj, definition, spec, report):
    if not isinstance(obj, xr.Dataset):
        prefix = f"auxiliary.{definition.filename}"
        report.add(
            f"{prefix}.grid_association",
            "pass"
            if spec.auxiliary_provenance.get(definition.filename)
            else "needs-review",
            STANDARD,
            evidence=spec.auxiliary_provenance.get(definition.filename),
        )
