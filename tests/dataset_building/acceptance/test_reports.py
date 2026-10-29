"""Public report serialization and completion boundaries."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.dataset_building.support.reports import (
    ValidationReport, report_exit_code, to_jsonable, write_reports,
)
from pipelines.dataset_building.checks.conversion import attach_conversion_evidence
from pipelines.dataset_building.workflows.dataset import validate_standard_dataset


def test_json_conversion_preserves_nonfinite_diagnostics_without_invalid_json():
    values = {
        "numbers": np.array([1.0, np.nan, np.inf, -np.inf]),
        "count": np.int64(2),
        "path": Path("dataset/report.json"),
        "nested": (np.bool_(True), {"missing": None}),
    }
    converted = to_jsonable(values)
    assert converted["numbers"] == [1.0, "nan", "inf", "-inf"]
    assert converted["count"] == 2
    assert converted["path"] == str(values["path"])
    assert converted["nested"] == [True, {"missing": None}]
    assert json.loads(json.dumps(converted, allow_nan=False)) == converted


@pytest.mark.parametrize("status,passed", [("not-run", True), ("fail", False)])
def test_future_receipt_is_optional_for_quick_but_invalid_receipt_is_not(status, passed):
    report = ValidationReport("fixture", "quick", {})
    report.coverage["quick_scan_complete"] = True
    report.add("publication.frozen_snapshot", status, "publication_readiness")
    assert report.quick_passed is passed
    assert report.to_dict()["quick_passed"] is passed
    assert bool(report.quick_blocking_checks) is not passed
    report.coverage["quick_scan_complete"] = False
    assert not report.quick_passed
    assert not report.release_passed


@pytest.mark.parametrize(
    ("mode", "status", "complete", "expected"),
    [
        ("quick", "pass", True, 0),
        ("full", "pass", True, 0),
        ("full", "fail", True, 1),
        ("quick", "pass", False, 2),
        ("full", "warning", True, 0),
        ("full", "needs-review", True, 2),
    ],
)
def test_merged_report_module_preserves_exit_policy(mode, status, complete, expected):
    report = ValidationReport("fixture", mode, {})
    report.coverage.update(
        quick_scan_complete=mode == "quick" and complete,
        scan_complete=mode == "full" and complete,
    )
    report.add("fixture.structure", status, "standard_conformance")
    assert report_exit_code([report.to_dict()]) == expected


@pytest.mark.parametrize(
    ("field", "value", "check", "status", "exit_code"),
    [
        ((), None, "identity", "pass", 0),
        (("snapshot_id",), "different-snapshot", "identity", "fail", 1),
        (("schema_sha256",), "b" * 64, "identity", "fail", 1),
        (("checks",), {}, "check_coverage", "fail", 1),
        (("checks",), [], "check_coverage", "fail", 1),
        (("source_identity",), {}, "provenance", "fail", 1),
        (("comparison_complete",), False, "full_comparison", "fail", 1),
        (("checks", "values", "status"), "fail", "values", "fail", 1),
        (("checks", "values", "status"), "blocked", "values", "blocked", 2),
        (("checks", "values", "status"), "warning", "values", "needs-review", 2),
        (("checks", "values", "evidence"), "", "values", "needs-review", 2),
        (("checks", "values"), None, "values", "not-run", 2),
    ],
)
def test_report_module_validates_conversion_receipts(field, value, check, status, exit_code):
    identity = {
        "data_revision": "revision-1",
        "snapshot_id": "snapshot-1",
        "product_id": "product-1",
        "product_version": "1.0",
        "config_id": "config-1",
        "template_version": "v1",
        "schema_sha256": "a" * 64,
    }
    report = ValidationReport("fixture", "full", identity)
    report.coverage["scan_complete"] = True
    receipt = {
        "dataset_id": report.dataset_id,
        **identity,
        "source_identity": {"id": "source-1", "snapshot_id": "source-snapshot-1"},
        "rules_version": "1.0",
        "mapping": {"source": "target"},
        "comparison_complete": True,
        "checks": {
            name: {"status": "pass", "evidence": "Explicit frozen mapping and full coverage"}
            for name in ("mapping", "coordinates", "channels", "statistics", "values", "coverage")
        },
    }
    if field:
        container = receipt
        for key in field[:-1]:
            container = container[key]
        container[field[-1]] = value
    attach_conversion_evidence(report, receipt)
    checks = {item.check_id: item.status for item in report.checks}
    assert checks[f"conversion_fidelity.{check}"] == status
    assert report.release_passed is (exit_code == 0)
    assert report_exit_code([report.to_dict()]) == exit_code


def test_json_csv_and_markdown_report_only_data_acceptance(published, tmp_path):
    path, spec = published
    report = validate_standard_dataset(path, spec, mode="full")
    assert report.release_passed
    document = report.to_dict()
    assert "consumer_compatibility" not in document
    assert "cross_dataset_compatibility" not in document
    assert all(
        check["category"] != "cross_dataset_compatibility"
        and check["check_id"] != "combination.scope"
        for check in document["checks"]
    )
    assert all(check["category"] != "consumer_compatibility" for check in document["checks"])
    assert document["conversion_fidelity"] == "not-applicable"
    assert document["identity"]["tool_version"] == "datakit-standard-validation/1.8"
    output = tmp_path / "reports"
    write_reports(output, [document])
    saved = json.loads(next(output.glob("*.json")).read_text(encoding="utf-8"))
    assert saved == document
    with (output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert "consumer_compatibility" not in rows[0]
    assert rows[0]["conversion_fidelity"] == "not-applicable"
    lines = (output / "summary.md").read_text(encoding="utf-8").splitlines()
    table = [line for line in lines if line.startswith("|")][:3]
    assert len(table) == 3
    assert all(len(line.split("|")) - 2 == 10 for line in table)
