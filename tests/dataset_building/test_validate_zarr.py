"""Single-dataset validation returns a report and saves JSON only on request."""

import json

import pytest

from pipelines.dataset import validate_zarr

from .samples.standard import snapshot_files


def test_default_full_validation_does_not_write_files(published, tmp_path):
    target, spec = published
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    original = snapshot_files(target)
    report = validate_zarr(target, spec)
    assert report.mode == "full"
    assert report.coverage["value_checked"] == 12
    assert report.release_passed
    assert "report_path" not in report.identity
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == before
    assert snapshot_files(target) == original


def test_explicit_report_and_callbacks_do_not_generate_sidecars(published, tmp_path):
    target, spec = published
    output = tmp_path / "reports" / "single.json"
    blocks = []
    report = validate_zarr(target, spec, output, on_block=blocks.append)
    assert len(blocks) == report.coverage["value_checked"] == 12
    assert list(output.parent.iterdir()) == [output]
    assert json.loads(output.read_text()) == report.to_dict()
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
    for output in (target / "report.json", schema):
        with pytest.raises(ValueError):
            validate_zarr(target, spec, output)
    assert snapshot_files(target) == before
    assert schema.read_bytes() == raw_schema
