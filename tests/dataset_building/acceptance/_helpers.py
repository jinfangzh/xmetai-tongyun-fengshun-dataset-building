"""Shared receipt binding and report assertions for independent fixtures."""

from __future__ import annotations

from pathlib import Path

from tests.dataset_building.samples.standard import write_production_evidence as _receipt

__all__ = ["SCHEMA_DIRECTORY", "_assert_counts", "_checks", "_receipt"]

SCHEMA_DIRECTORY = Path(__file__).resolve().parents[3] / "config" / "dataset_specs"


def _checks(report):
    return {check.check_id: check.status for check in report.checks}


def _assert_counts(report):
    c = report.coverage
    assert c["value_checked"] + c["read_failed"] + c["not_scanned"] == c["logical_blocks_expected"]
    assert (
        c["physically_present"]
        + c["allowed_omitted"]
        + c["unapproved_missing"]
        + c["storage_unknown"]
        == c["logical_blocks_expected"]
    )


def validate_fixture(target, spec, **options):
    """Run a rule-focused quick/full check, retaining each call's YAML report."""
    from uuid import uuid4

    from pipelines.dataset import validate_zarr

    options.setdefault("mode", "quick")
    options.setdefault("output", Path(target).parent / f"check-{uuid4().hex}.yaml")
    return validate_zarr(target, spec, **options)
