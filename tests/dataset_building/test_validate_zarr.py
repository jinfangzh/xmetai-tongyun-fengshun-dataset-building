"""Single-dataset validation returns a report and saves YAML only on request."""

import hashlib
from dataclasses import replace

import yaml

import pytest

from pipelines.dataset import validate_zarr
from datakit.schema.specifications import EvidenceReference

from .samples.standard import snapshot_files


@pytest.mark.parametrize("filename", ["report.json", "report.yml", "report", "report.YAML", "report.txt"])
def test_report_suffix_is_rejected_before_reading_inputs(tmp_path, monkeypatch, filename):
    import pipelines.dataset.validate as validation

    def unexpected(*args, **kwargs):
        pytest.fail("must reject the output suffix before scanning data")
    monkeypatch.setattr(validation, "_validate_dataset", unexpected)
    output = tmp_path / "reports" / filename
    with pytest.raises(ValueError, match=".yaml suffix"):
        validate_zarr(tmp_path / "missing.zarr", tmp_path / "missing-spec.yaml", output)
    assert not output.parent.exists()


def test_default_full_validation_does_not_write_files(published, tmp_path):
    target, spec = published
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    original = snapshot_files(target)
    report = validate_zarr(target, spec)
    assert report.mode == "full"
    assert report.identity["validation_basis"] == "target_contract_and_declared_rules"
    assert report.identity["source_target_comparison"] is False
    assert report.coverage["value_checked"] == 12
    assert report.release_passed
    assert "report_path" not in report.identity
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == before
    assert snapshot_files(target) == original


def test_explicit_report_and_callbacks_do_not_generate_sidecars(published, tmp_path):
    target, spec = published
    output = tmp_path / "reports" / "single.yaml"
    blocks = []
    report = validate_zarr(target, spec, output, on_block=blocks.append)
    assert len(blocks) == report.coverage["value_checked"] == 12
    assert list(output.parent.iterdir()) == [output]
    assert yaml.safe_load(output.read_text()) == report.to_dict()
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        validate_zarr(target, spec, output)
    assert output.read_bytes() == before


def test_unsaved_validation_can_be_repeated(published, tmp_path):
    target, spec = published
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    first = validate_zarr(target, spec, output=None)
    second = validate_zarr(target, spec, output=None)
    assert first.release_passed and second.release_passed
    assert "report_path" not in first.identity
    assert "report_path" not in second.identity
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == before


def test_report_cannot_modify_dataset_or_schema(published_schema):
    target, spec, schema, _ = published_schema
    before = snapshot_files(target)
    raw_schema = schema.read_bytes()
    for output in (target / "report.yaml", schema):
        with pytest.raises(ValueError):
            validate_zarr(target, spec, output)
    assert snapshot_files(target) == before
    assert schema.read_bytes() == raw_schema


@pytest.mark.parametrize("options", [
    {"mode": "unknown"}, {"sample_blocks": True}, {"sample_blocks": 1.5},
    {"sample_blocks": 0}, {"max_blocks": False}, {"max_blocks": 2.5},
    {"memory_budget_mib": float("nan")}, {"memory_budget_mib": float("inf")},
    {"memory_budget_mib": True}, {"memory_budget_mib": 0},
    {"stop_requested": True}, {"on_block": []},
])
def test_invalid_scan_controls_fail_before_reading_contract(tmp_path, options):
    # A missing contract must not mask an invalid scan request or produce files.
    output = tmp_path / "reports" / "result.yaml"
    with pytest.raises((TypeError, ValueError)):
        validate_zarr(tmp_path / "missing.zarr", tmp_path / "missing.yaml", output, **options)
    assert not output.parent.exists()


@pytest.mark.parametrize("field,check_id", [
    ("production_evidence", "publication.frozen_snapshot"),
    ("conversion_evidence", "conversion_fidelity.external_evidence"),
])
@pytest.mark.parametrize("raw", [b"[]", b"null", b'"text"', b'{"broken":'])
def test_malformed_evidence_is_reported_without_aborting_target_checks(published, tmp_path, field, check_id, raw):
    target, spec = published
    receipt = tmp_path / "invalid-evidence.json"
    receipt.write_bytes(raw)
    spec = replace(spec, **{field: EvidenceReference(receipt, hashlib.sha256(raw).hexdigest())})
    original = snapshot_files(target)
    report = validate_zarr(target, spec, output=tmp_path / "report.yaml")
    assert not report.release_passed
    assert report.coverage["value_checked"] == spec.expected_shape[0]
    assert any(check.check_id == check_id and check.status == "fail" for check in report.checks)
    assert snapshot_files(target) == original


def test_declared_conversion_evidence_requirement_is_not_silently_dropped(published):
    target, spec = published
    report = validate_zarr(target, replace(spec, conversion_required=True))
    assert not report.release_passed
    assert report.coverage["value_checked"] == spec.expected_shape[0]
    assert report.identity["source_target_comparison"] is False
    assert any(check.check_id == "conversion_fidelity.external_evidence"
               and check.status == "not-run" for check in report.checks)
