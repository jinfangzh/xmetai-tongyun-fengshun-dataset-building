"""Build report persistence without dumping diagnostics to the terminal."""

import json

import numpy as np
import pytest

from pipelines.dataset.reports import (
    BuildReport,
    ValidationReport,
    print_build_summary,
    save_build_report,
)


def test_build_report_saves_complete_diagnostics_without_stdout(tmp_path, capsys):
    report = BuildReport(str(tmp_path / "product.zarr"), dry_run=True)
    report.creation_report = ValidationReport("test-product", "create", {})
    report.creation_report.quality["example"] = {"nan_fraction": np.float32(0.25),
                                                  "minimum": float("nan")}
    report.coverage["conformance_passed"] = True
    first = save_build_report(report, tmp_path / "reports")
    second = save_build_report(report, tmp_path / "reports")
    assert first != second and first.exists() and second.exists()
    saved = json.loads(first.read_text())
    assert saved["coverage"]["conformance_passed"]
    assert saved["validation_report"] is None
    assert saved["creation_report"]["quality"]["example"] == {"nan_fraction": 0.25, "minimum": "nan"}
    assert capsys.readouterr().out == ""


def test_build_report_cannot_be_saved_inside_zarr_destination(tmp_path):
    destination = tmp_path / "product.zarr"
    report = BuildReport(str(destination))
    with pytest.raises(ValueError, match="outside"):
        save_build_report(report, destination / "reports")
    assert not destination.exists()


def test_dry_run_summary_keeps_metrics_in_file(tmp_path, capsys):
    report = BuildReport(str(tmp_path / "product.zarr"), dry_run=True)
    report.coverage["conformance_passed"] = True
    report.creation_report = ValidationReport("era5.test", "create", {})
    for category in ("standard_conformance", "data_integrity", "data_quality"):
        report.creation_report.add(f"test.{category}", "pass", category)
    report.creation_report.quality["large_metrics"] = "DO_NOT_PRINT_CHANNEL_DIAGNOSTICS"
    print_build_summary(report, tmp_path / "report.json", processor="era5")
    output = capsys.readouterr().out
    assert "ERA5 pipeline：完成（dry-run）" in output
    assert "标准符合性：通过" in output
    assert "附件复制：未执行（dry-run）" in output
    assert "Quick校验：未执行" in output
    assert "完整发布验收：未完成" in output
    assert "DO_NOT_PRINT_CHANNEL_DIAGNOSTICS" not in output


def test_summary_separates_quick_failures_from_pending_release_checks(tmp_path, capsys):
    report = BuildReport(str(tmp_path / "product.zarr"))
    report.coverage.update(written=True, conformance_passed=True)
    report.stages["auxiliary_copy"] = "passed"
    validation = ValidationReport("oras5.test", "quick", {})
    validation.add("auxiliary.const.nc.exists", "fail", "standard_conformance")
    validation.add("publication.frozen_snapshot", "not-run", "publication_readiness")
    validation.add("scan.complete", "not-run", "full_scan")
    report.validation_report = validation
    report.fail_checks("validate", validation)
    print_build_summary(report, tmp_path / "report.json", processor="oras5")
    output = capsys.readouterr().out
    assert "ORAS5 pipeline：失败" in output
    assert "写入：成功" in output
    assert "Quick校验：未通过" in output
    assert "失败：auxiliary.const.nc.exists" in output
    assert "待完成项：publication.frozen_snapshot、scan.complete" in output
    assert "失败：scan.complete" not in output
    assert "失败阶段：validate" not in output


def test_summary_shows_exception_when_processing_fails(tmp_path, capsys):
    report = BuildReport(str(tmp_path / "product.zarr"), dry_run=True)
    report.fail("pipeline", "missing hourly source data", ValueError())
    print_build_summary(report, tmp_path / "report.json", processor="era5")
    output = capsys.readouterr().out
    assert "pipeline：失败（dry-run）" in output
    assert "标准符合性：未执行" in output
    assert "失败阶段：pipeline；原因：missing hourly source data" in output
