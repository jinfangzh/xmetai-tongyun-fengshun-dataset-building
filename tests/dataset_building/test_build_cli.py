"""Command-line dispatch and report behavior, without real dataset writes."""

import json

import pytest

from examples.dataset_building import example_build_era5 as era5
from examples.dataset_building import example_build_oras5 as oras5
from pipelines import dataset as dataset_pipeline
from pipelines.dataset.reports import BuildReport, ValidationReport


def _arguments(tmp_path):
    return ["--source-config", str(tmp_path / "source.yaml"),
            "--target-spec", str(tmp_path / "target.yaml"),
            "--source", str(tmp_path / "input"),
            "--destination", str(tmp_path / "output" / "product.zarr"),
            "--mask-nc", str(tmp_path / "mask.nc"),
            "--report-dir", str(tmp_path / "output" / "reports")]


@pytest.mark.parametrize("module,stage", [(oras5, None), (era5, "raw"), (era5, "packed")])
@pytest.mark.parametrize("write", [False, True])
def test_cli_dispatch_and_concise_summary(tmp_path, monkeypatch, capsys, module, stage, write):
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        report = BuildReport(str(kwargs["destination"]), dry_run=kwargs["dry_run"])
        report.coverage.update(conformance_passed=True, written=not kwargs["dry_run"])
        if not kwargs["dry_run"]:
            report.stages["validate"] = "passed"
            report.validation_report = ValidationReport("test", "quick", {})
            report.validation_report.coverage["quick_scan_complete"] = True
        return report

    monkeypatch.setattr(dataset_pipeline, "build_zarr", fake_build)
    args = _arguments(tmp_path)
    if stage:
        args += ["--source-stage", stage, "--accumulation-hours", "6"]
    if write:
        args += ["--write"]
    args += ["--mean-nc", str(tmp_path / "mean.nc"), "--std-nc", str(tmp_path / "std.nc")]
    assert module.main(args) == 0
    call = calls[0]
    assert call["dry_run"] == (not write)
    assert call["process"].keywords["mean_path"] == tmp_path / "mean.nc"
    assert call["process"].keywords["mask_path"] == tmp_path / "mask.nc"
    if stage:
        assert (call.get("read") is era5.read_source) == (stage == "raw")
        assert call["process"].keywords["source_stage"] == stage
    files = list((tmp_path / "output" / "reports").glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["coverage"]["conformance_passed"]
    output = capsys.readouterr().out
    assert "pipeline：完成" in output and "完整报告：" in output
    assert "写入：成功" in output if write else "写入：未执行（dry-run）" in output
    assert "Quick校验：通过" in output if write else "Quick校验：未执行" in output
    assert "creation_report" not in output
    assert not (tmp_path / "output" / "product.zarr").exists()


def test_cli_exception_is_saved_to_custom_report_directory(tmp_path, monkeypatch):
    def fail_build(**kwargs):
        raise ValueError("incomplete source timestamps")

    monkeypatch.setattr(dataset_pipeline, "build_zarr", fail_build)
    reports = tmp_path / "diagnostics"
    assert oras5.main(_arguments(tmp_path) + ["--report-dir", str(reports)]) == 1
    saved = json.loads(next(reports.glob("*.json")).read_text())
    assert saved["failures"][0]["message"] == "incomplete source timestamps"
    assert not saved["coverage"]["written"]


@pytest.mark.parametrize("hours", ["0", "-1", "nan", "inf", "1.5"])
def test_cli_rejects_invalid_raw_accumulation_hours(tmp_path, hours):
    with pytest.raises(SystemExit) as caught:
        era5.main(_arguments(tmp_path) + ["--accumulation-hours", hours])
    assert caught.value.code == 2
    assert not (tmp_path / "output").exists()


def test_cli_rejects_conflicting_write_modes(tmp_path):
    with pytest.raises(SystemExit) as caught:
        oras5.main(_arguments(tmp_path) + ["--write", "--dry-run"])
    assert caught.value.code == 2


@pytest.mark.parametrize("module", [era5, oras5])
def test_cli_requires_report_directory(tmp_path, module, capsys):
    args = _arguments(tmp_path)
    del args[-2:]
    with pytest.raises(SystemExit) as caught:
        module.main(args)
    assert caught.value.code == 2
    assert "--report-dir" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("module", [era5, oras5])
def test_cli_passes_task_identity_and_coverage(tmp_path, monkeypatch, module):
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        report = BuildReport(str(kwargs["destination"]), dry_run=True)
        report.coverage["conformance_passed"] = True
        return report

    monkeypatch.setattr(dataset_pipeline, "build_zarr", fake_build)
    assert module.main(_arguments(tmp_path) + [
        "--start-time", "2025-01-01T00", "--end-time", "2025-01-31T00",
        "--dataset-id", "product.zarr", "--content-version", "v2", "--data-revision", "20261104",
    ]) == 0
    assert calls[0]["build_overrides"] == {
        "first_time": "2025-01-01T00", "last_time": "2025-01-31T00",
        "dataset_id": "product.zarr", "content_version": "v2", "data_revision": "20261104",
    }


@pytest.mark.parametrize("module", [era5, oras5])
@pytest.mark.parametrize("extra", [
    ["--start-time", "2025-01-01T00"],
    ["--end-time", "2025-01-31T00"],
    ["--start-time", "2025-01-01T00", "--end-time", "2025-01-31T00"],
])
def test_cli_rejects_incomplete_task_definition(tmp_path, module, extra):
    with pytest.raises(SystemExit) as caught:
        module.main(_arguments(tmp_path) + extra)
    assert caught.value.code == 2


@pytest.mark.parametrize("module,weight_option,weight_name", [
    (oras5, "--weight-nc", "weight.nc"), (era5, "--weight-npy", "weight.npy"),
])
def test_cli_passes_attachment_paths(tmp_path, monkeypatch, module, weight_option, weight_name):
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        report = BuildReport(str(kwargs["destination"]), dry_run=True)
        report.coverage["conformance_passed"] = True
        return report

    monkeypatch.setattr(dataset_pipeline, "build_zarr", fake_build)
    assert module.main(_arguments(tmp_path) + ["--const-nc", str(tmp_path / "static.nc"),
                                              weight_option, str(tmp_path / "weights"),
                                              "--auxiliary", f"extra.nc={tmp_path / 'extra.nc'}"]) == 0
    assert calls[0]["auxiliaries"] == {"const.nc": tmp_path / "static.nc", weight_name: tmp_path / "weights",
                                       "extra.nc": tmp_path / "extra.nc"}
    with pytest.raises(SystemExit) as caught:
        module.main(_arguments(tmp_path) + ["--const-nc", str(tmp_path / "static.nc"),
                                            "--auxiliary", "const.nc=duplicate.nc"])
    assert caught.value.code == 2
