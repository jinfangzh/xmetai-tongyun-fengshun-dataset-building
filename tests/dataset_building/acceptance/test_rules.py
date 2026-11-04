"""Direct coverage for the format/value/storage/conversion rule separation."""

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from datakit.schema.specifications import AuxiliaryDefinition, EvidenceReference
from pipelines.dataset.old.conversion_validation import (
    check_statistics as check_conversion_statistics,
)
from datakit.validation.comparison import compare_channel_values
from datakit.validation.structure import check_auxiliary_format, check_dataset_format
from datakit.validation.storage import block_storage_status, valid_chunk_key
from datakit.validation.values import check_statistics
from pipelines.dataset.reports import ValidationReport
from pipelines.dataset.old.conversion_validation import ConversionChannelRule, ConversionValidationSpec

from ._helpers import _checks, validate_fixture


def test_format_check_does_not_certify_statistics_values(published):
    path, spec = published
    with xr.open_zarr(str(path), consolidated=False, chunks=None, mask_and_scale=False) as obj:
        obj["std"] = xr.DataArray(np.zeros(3, dtype="f4"), dims=("channel",))
        report = ValidationReport(spec.dataset_id, "quick", {})
        valid, checks = check_dataset_format(obj, spec)
        report.checks.extend(checks)
        assert valid is True
        assert "statistics.finite_positive" not in _checks(report)
        valid, checks = check_statistics(obj["mean"].values, obj["std"].values)
        report.checks.extend(checks)
        assert not valid
        assert _checks(report)["statistics.finite_positive"] == "fail"


def test_invalid_auxiliary_header_does_not_read_values(published, monkeypatch):
    _, spec = published
    definition = AuxiliaryDefinition(
        "weight.nc",
        "netcdf",
        "weight",
        ("lat", "lon"),
        "float32",
        "weight",
        check_variable_name=True,
    )
    obj = xr.Dataset({"weight": (("lat", "lon"), np.zeros((2, 2), dtype="f4"))})
    report = ValidationReport(spec.dataset_id, "quick", {})

    def unexpected_values(self):
        raise AssertionError("Invalid header must be rejected before materialization")

    monkeypatch.setattr(xr.DataArray, "values", property(unexpected_values))
    values, checks = check_auxiliary_format(obj, definition, spec)
    report.checks.extend(checks)
    assert values is None
    assert _checks(report)["auxiliary.weight.nc.header_layout"] == "fail"
    assert _checks(report)["auxiliary.weight.nc.values"] == "blocked"


@pytest.mark.parametrize(
    ("present", "approved", "stable", "expected"),
    [
        (True, False, False, "physically_present"),
        (False, True, True, "allowed_omitted"),
        (False, True, False, "unapproved_missing"),
        (False, False, True, "unapproved_missing"),
    ],
)
def test_physical_omission_requires_approval_and_stability(present, approved, stable, expected):
    assert block_storage_status(present=present, approved=approved, stable=stable) == expected


def test_storage_key_rules_do_not_require_files_to_exist(tmp_path):
    metadata = {
        "path": str(tmp_path),
        "zarr_format": 3,
        "arrays": {"data": {"raw": {"chunk_key_encoding": {"name": "default"}}}},
    }
    assert valid_chunk_key(tmp_path / "data" / "c" / "1" / "0" / "0" / "0", metadata, 3)
    assert not valid_chunk_key(tmp_path / "data" / "c" / "3" / "0" / "0" / "0", metadata, 3)
    assert not valid_chunk_key(tmp_path / "data" / "c" / "1" / "1" / "0" / "0", metadata, 3)
    assert not valid_chunk_key(tmp_path / "data" / "garbage", metadata, 3)


@pytest.mark.parametrize("offset,passed", [(0.0, True), (0.01, True), (0.11, False)])
def test_conversion_tolerance_is_unchanged_and_inputs_are_not_mutated(offset, passed):
    source = np.array([0.0, 1.0, np.nan, np.inf, -np.inf], dtype="f4")
    actual = np.array([1.0 + offset, 3.0, np.nan, np.inf, -np.inf], dtype="f4")
    source_before, actual_before = source.copy(), actual.copy()
    rule = ConversionChannelRule(
        "target",
        "source",
        atol=0.1,
        rtol=0.0,
        value_factor=2.0,
        value_offset=1.0,
    )
    result = compare_channel_values(
        actual,
        source,
        value_factor=rule.value_factor,
        value_offset=rule.value_offset,
        atol=rule.atol,
        rtol=rule.rtol,
    )
    assert result["passed"] is passed
    assert result["missing_pattern_equal"]
    assert result["outside_tolerance"] == (0 if passed else 1)
    assert np.array_equal(source, source_before, equal_nan=True)
    assert np.array_equal(actual, actual_before, equal_nan=True)


def test_root_checks_are_retained_when_auxiliary_directory_is_missing(published, tmp_path):
    path, spec = published
    report = validate_fixture(
        path,
        replace(spec, directory_name=""),
        mode="quick",
        auxiliary_directory=tmp_path / "missing",
    )
    checks = _checks(report)
    assert checks["layout.declared_root_objects"] == "pass"
    assert checks["scan.exception"] == "fail"
    assert not report.quick_passed


@pytest.mark.parametrize(
    "case",
    ["valid", "identity", "provenance", "channels", "zero_std", "nonfinite", "shape"],
)
def test_conversion_statistics_receipt_rules_do_not_read_files(tmp_path, case):
    reference = EvidenceReference(tmp_path / "not-read.json", "0" * 64)
    conversion = ConversionValidationSpec(
        "source",
        "snapshot",
        "v1",
        "2024-01-01T00",
        "2024-01-01T00",
        6,
        ("a",),
        (ConversionChannelRule("a", "a", 0, 0),),
        source_statistics=reference,
    )
    statistics = {
        "source_identity": "source",
        "source_snapshot_id": "snapshot",
        "channels": ["a"],
        "provenance": "independent reference",
        "mean": [2.0],
        "std": [3.0],
    }
    if case == "identity":
        statistics["source_identity"] = "wrong"
    elif case == "provenance":
        statistics["provenance"] = ""
    elif case == "channels":
        statistics["channels"] = ["wrong"]
    elif case == "zero_std":
        statistics["std"] = [0.0]
    elif case == "nonfinite":
        statistics["mean"] = [float("inf")]
    elif case == "shape":
        statistics["mean"] = [2.0, 2.0]
    target = xr.Dataset({"mean": ("channel", [2.0]), "std": ("channel", [3.0])})
    if case in {"identity", "provenance", "channels"}:
        with pytest.raises(ValueError, match="identity/channels/provenance"):
            check_conversion_statistics(xr.Dataset(), target, conversion, [0], statistics)
    else:
        assert check_conversion_statistics(
            xr.Dataset(),
            target,
            conversion,
            [0],
            statistics,
        ) is (case == "valid")
    assert not reference.path.exists()
