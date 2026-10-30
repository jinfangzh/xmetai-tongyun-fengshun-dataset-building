"""Sample-driven single-dataset validation and conversion-evidence workflows."""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from datakit.validation.zarr_inspection import read_zarr_metadata, zarr_chunk_path
from pipelines.dataset import validate_zarr
from pipelines.dataset.configuration import load_validation_run
from pipelines.dataset.old.conversion_validation import run_conversion_validation

from .samples.standard import (
    evidence_reference,
    snapshot_files,
    write_conversion_reference,
    write_validation_run,
)


def _validate_runs(runs, output, **options):
    # Batch orchestration belongs to the caller; each API call handles one dataset.
    reports = []
    for number, run in enumerate(runs, start=1):
        spec, locations = load_validation_run(run)
        report = validate_zarr(
            locations["dataset_path"],
            spec,
            output=output / f"{number}.json",
            auxiliary_directory=locations["auxiliary_directory"],
            **options,
        )
        reports.append(report.to_dict())
    return reports


def _assert_saved_reports(output, reports):
    assert len(list(output.iterdir())) == len(reports)
    for number, report in enumerate(reports, start=1):
        saved = output / f"{number}.json"
        assert json.loads(saved.read_text(encoding="utf-8")) == report
        coverage = report["coverage"]
        assert coverage["value_checked"] + coverage["read_failed"] + coverage["not_scanned"] == 12
        assert (
            sum(
                coverage[key]
                for key in (
                    "physically_present",
                    "allowed_omitted",
                    "unapproved_missing",
                    "storage_unknown",
                )
            )
            == 12
        )
    assert not (output / "batch.json").exists()
    assert not list(output.glob("*.jsonl"))


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_independent_sample_batch_matches_expected_results(validation_samples, tmp_path, mode):
    runs = [write_validation_run(schema, spec) for _, spec, schema, _ in validation_samples]
    frozen_inputs = {
        path: path.read_bytes() for path in [*runs, *(sample[2] for sample in validation_samples)]
    }
    frozen_data = [snapshot_files(sample[0]) for sample in validation_samples]
    for _, spec, _, _ in validation_samples:
        assert spec.production_evidence is not None
        receipt = spec.production_evidence.path
        frozen_inputs[receipt] = receipt.read_bytes()
    first, later = [sample[1] for sample in validation_samples]
    assert first.product.channel_names == ("t2m", "q1000", "tp")
    assert later.product.channel_names == ("q1000", "t2m")
    assert first.last_time < later.first_time
    output = tmp_path / "reports"

    reports = _validate_runs(runs, output, mode=mode)

    assert len(reports) == 2
    assert [report["dataset_id"] for report in reports] == ["fixture", "later"]
    for report, (path, spec, schema, _) in zip(reports, validation_samples, strict=True):
        assert report["quick_passed"] is (mode == "quick")
        assert report["release_passed"] is (mode == "full")
        assert report["coverage"]["scan_complete"] is (mode == "full")
        assert report["coverage"]["value_checked"] == (3 if mode == "quick" else 12)
        assert report["coverage"]["read_failed"] == 0
        assert report["identity"]["schema_sha256"] == evidence_reference(schema)["sha256"]
        assert report["identity"]["config_id"] == spec.config_id
        assert report["identity"]["path"] == str(path)
        assert "cross_dataset_compatibility" not in report
        assert all(check["check_id"] != "combination.scope" for check in report["checks"])
    _assert_saved_reports(output, reports)
    assert [snapshot_files(sample[0]) for sample in validation_samples] == frozen_data
    assert all(path.read_bytes() == raw for path, raw in frozen_inputs.items())


@pytest.mark.parametrize(
    ("fault", "check_id"),
    [
        ("channel-order", "channels.exact_order"),
        ("nan", "quality.t2m"),
        ("out-of-bounds", "quality.q1000"),
        ("statistics", "statistics.finite_positive"),
        ("missing-block", "storage.no_unapproved_missing"),
        ("corrupt-block", "scan.no_read_failures"),
    ],
)
def test_bad_sample_is_reported_without_blocking_other_datasets(
    validation_samples,
    tmp_path,
    fault,
    check_id,
):
    path = validation_samples[0][0]
    group = zarr.open_group(str(path), mode="r+")
    if fault == "channel-order":
        group["channel"][:] = ["tp", "q1000", "t2m"]
    elif fault == "nan":
        group["data"][1, 0, 0, 0] = np.nan
    elif fault == "out-of-bounds":
        group["data"][1, 1, 0, 0] = 11
    elif fault == "statistics":
        group["std"][0] = 0
    else:
        chunk = zarr_chunk_path(read_zarr_metadata(path), "data", (1, 0, 0, 0))
        if fault == "missing-block":
            chunk.unlink()
        else:
            chunk.write_bytes(b"deliberately invalid compressed sample")
    runs = [write_validation_run(schema, spec) for _, spec, schema, _ in validation_samples]
    before = [snapshot_files(sample[0]) for sample in validation_samples]
    output = tmp_path / "reports"

    reports = _validate_runs(runs, output, mode="full")

    assert len(reports) == 2
    bad, good = reports
    assert bad["dataset_id"] == "fixture" and good["dataset_id"] == "later"
    assert bad["release_passed"] is False
    assert {check["check_id"]: check["status"] for check in bad["checks"]}[check_id] == "fail"
    assert good["release_passed"] is True
    assert good["coverage"]["value_checked"] == 12
    assert good["coverage"]["unapproved_missing"] == good["coverage"]["read_failed"] == 0
    if fault == "missing-block":
        assert bad["coverage"]["unapproved_missing"] == 1
    if fault == "corrupt-block":
        assert bad["coverage"]["read_failed"] == 1
    _assert_saved_reports(output, reports)
    assert [snapshot_files(sample[0]) for sample in validation_samples] == before


def test_quick_pass_is_not_full_acceptance_of_an_unselected_bad_frame(published_schema, tmp_path):
    path, spec, schema, _ = published_schema
    zarr.open_array(str(path / "data"), mode="r+")[1, 0, 0, 0] = np.nan
    run = write_validation_run(schema, spec)
    before = snapshot_files(path)

    quick = _validate_runs([run], tmp_path / "quick", mode="quick")[0]
    full = _validate_runs([run], tmp_path / "full", mode="full")[0]

    assert quick["coverage"]["selection"] == [0, 5, 11]
    assert quick["coverage"]["value_checked"] == 3
    assert quick["quick_passed"] is True
    assert quick["release_passed"] is False
    assert full["coverage"]["value_checked"] == 12
    assert full["release_passed"] is False
    assert {check["check_id"]: check["status"] for check in full["checks"]}["quality.t2m"] == "fail"
    assert snapshot_files(path) == before


@pytest.mark.parametrize("mismatch", [False, True], ids=["matching", "wrong-value"])
def test_mapped_conversion_receipt_controls_later_acceptance(published_schema, tmp_path, mismatch):
    target, spec, schema, _ = published_schema
    source, config = write_conversion_reference(schema, spec)
    if mismatch:
        zarr.open_array(str(source / "data"), mode="r+")[1, 0, 0, 0] += 1
    frozen_yaml = schema.read_bytes()
    source_before, target_before = snapshot_files(source), snapshot_files(target)
    receipt_path = tmp_path / "receipt.json"

    receipt = run_conversion_validation(config, receipt_path, mode="full")

    assert receipt["comparison_complete"] is True
    assert set(receipt["checks"]) == {
        "mapping",
        "coordinates",
        "channels",
        "statistics",
        "values",
        "coverage",
    }
    assert receipt["checks"]["values"]["status"] == ("fail" if mismatch else "pass")
    assert receipt["checks"]["coordinates"]["status"] == "pass"
    assert receipt["checks"]["channels"]["status"] == "pass"
    assert receipt["checks"]["statistics"]["status"] == "pass"
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert (
        len(receipt_path.with_suffix(".blocks.jsonl").read_text(encoding="utf-8").splitlines())
        == 12
    )
    run = write_validation_run(schema, spec)
    document = json.loads(run.read_text(encoding="utf-8"))
    document["execution"].update(
        conversion_required=True,
        conversion_evidence=evidence_reference(receipt_path),
    )
    run.write_text(json.dumps(document), encoding="utf-8")
    reports = _validate_runs([run], tmp_path / "reports", mode="full")
    report = reports[0]
    assert report["release_passed"] is (not mismatch)
    assert {check["check_id"]: check["status"] for check in report["checks"]}[
        "conversion_fidelity.full_comparison"
    ] == "pass"
    assert {check["check_id"]: check["status"] for check in report["checks"]}[
        "conversion_fidelity.values"
    ] == ("fail" if mismatch else "pass")
    assert report["identity"]["schema_sha256"] == receipt["schema_sha256"]
    assert snapshot_files(source) == source_before
    assert snapshot_files(target) == target_before
    assert schema.read_bytes() == frozen_yaml
