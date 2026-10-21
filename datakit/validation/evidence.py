"""Bind independently executed consumer and conversion gates to one revision."""

from __future__ import annotations

from typing import Any

from .reports import ValidationReport

CONSUMER_CHECKS = frozenset(
    {
        "environment",
        "metadata",
        "channel_layout",
        "model_parameters",
        "time_windows",
        "spatial_alignment",
        "channel_semantics",
        "multi_dataset",
        "batch",
        "forward",
        "inverse",
        "output",
    }
)
CONVERSION_CHECKS = frozenset(
    {"mapping", "coordinates", "channels", "statistics", "values", "coverage"}
)


def attach_standard_evidence(
    report: ValidationReport, evidence: dict[str, Any], *, category: str
) -> None:
    """Validate a receipt's binding and mandatory check coverage; no file IO.

    A receipt is an external test's result, not generated from dataset metadata.
    The path pipeline separately verifies its independently frozen SHA256.
    """
    required = (
        CONSUMER_CHECKS
        if category == "consumer_compatibility"
        else CONVERSION_CHECKS if category == "conversion_fidelity" else None
    )
    if required is None:
        raise ValueError("Only consumer or conversion evidence can be attached")
    expected = {
        "dataset_id": report.dataset_id,
        "data_revision": report.identity["data_revision"],
        "snapshot_id": report.identity["snapshot_id"],
        "product_id": report.identity["product_id"],
        "product_version": report.identity["product_version"],
        "profile_version": report.identity["profile_version"],
    }
    checks = evidence.get("checks", {})
    bound = bool(expected["snapshot_id"]) and all(evidence.get(k) == v for k, v in expected.items())
    report.require(
        f"{category}.identity", bound, category, expected, {k: evidence.get(k) for k in expected}
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
    metadata_fields = (
        ("consumer_version", "environment", "config_sha256", "mapping")
        if category == "consumer_compatibility"
        else ("source_identity", "rules_version", "mapping")
    )
    report.require(
        f"{category}.provenance",
        all(evidence.get(k) for k in metadata_fields),
        category,
        metadata_fields,
        {k: evidence.get(k) for k in metadata_fields},
    )
    if category == "conversion_fidelity":
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
        if status not in ("pass", "fail", "needs-review", "blocked", "not-run") and not (
            name == "multi_dataset" and status == "not-applicable"
        ):
            status = "needs-review"
        # Passing checks must identify what was actually exercised.
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
