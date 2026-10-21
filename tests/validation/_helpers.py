"""Shared receipt binding and report assertions for independent fixtures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from datakit.schema.standard_dataset import DatasetValidationSpec, EvidenceReference


def _receipt(path: Path, spec: DatasetValidationSpec) -> DatasetValidationSpec:
    receipt = path.parent / "production.json"
    receipt.write_text(
        json.dumps(
            {
                "dataset_id": spec.dataset_id,
                "data_revision": spec.data_revision,
                "snapshot_id": "frozen-fixture",
                "shape": list(spec.expected_shape),
                "logical_blocks_expected": spec.expected_shape[0],
                "complete": True,
                "allowed_omitted": [list(i) for i in spec.allowed_omitted],
                "stability": "writes-stopped",
            }
        ),
        encoding="utf-8",
    )
    return replace(
        spec,
        snapshot_id="frozen-fixture",
        production_evidence=EvidenceReference(
            receipt, hashlib.sha256(receipt.read_bytes()).hexdigest()
        ),
    )


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
