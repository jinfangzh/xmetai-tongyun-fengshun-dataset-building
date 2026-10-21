"""Public report serialization and completion boundaries."""

import json
from pathlib import Path

import numpy as np
import pytest

from datakit.validation import ValidationReport, to_jsonable


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
