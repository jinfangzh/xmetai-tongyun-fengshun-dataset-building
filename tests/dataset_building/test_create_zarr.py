"""Creation enforces a spec without adapting or mutating the caller's Dataset."""

import json
from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from pipelines.dataset import create_zarr, validate_zarr


@pytest.mark.parametrize("version", [2, 3])
def test_create_checks_and_preserves_input(published, tmp_path, version):
    path, spec = published
    spec = replace(spec, zarr_format=version)
    target = tmp_path / "created.zarr"
    output = tmp_path / "report.json"
    with xr.open_zarr(path, chunks=None, mask_and_scale=False) as dataset:
        dataset["data"].encoding.update(dtype="int16", scale_factor=2.0, add_offset=10.0)
        before = dataset.copy(deep=True).load()
        encodings = {name: dict(dataset[name].encoding) for name in dataset.variables}
        closed = []
        dataset.set_close(lambda: closed.append(True))
        report = create_zarr(dataset, target, spec, output=output)
        assert report.coverage["conformance_passed"], report.to_dict()
        assert report.coverage["written"], report.to_dict()
        assert report.coverage["frames_checked"] == 12
        assert not report.release_passed
        assert not closed
        xr.testing.assert_identical(dataset, before)
        assert encodings == {name: dataset[name].encoding for name in dataset.variables}
        with xr.open_zarr(target, chunks=None, mask_and_scale=False) as actual:
            xr.testing.assert_identical(actual, before)
            assert actual.data.encoding["chunks"] == spec.expected_chunks
        assert json.loads(output.read_text()) == report.to_dict()
        if version == 3:
            accepted = validate_zarr(target, spec)
            assert accepted.release_passed, accepted.to_dict()


@pytest.mark.parametrize("fault", ["dtype", "channels", "attribute", "mask", "statistics", "quality", "missing"])
def test_invalid_input_returns_report_without_creating(published, tmp_path, fault):
    path, spec = published
    with xr.open_zarr(path, chunks=None, mask_and_scale=False) as opened:
        dataset = opened.load()
    if fault == "dtype":
        dataset["data"] = dataset.data.astype("f4")
    elif fault == "channels":
        dataset = dataset.isel(channel=[1, 0, 2])
    elif fault == "attribute":
        del dataset.attrs["dataset_id"]
    elif fault == "mask":
        dataset["mask"][:] = 2
    elif fault == "statistics":
        dataset["std"][:] = 0
    elif fault == "quality":
        dataset["data"][0, 0, 0, 0] = np.inf
    else:
        dataset = dataset.drop_vars("mean")
    before = dataset.copy(deep=True)
    target = tmp_path / "invalid.zarr"
    output = tmp_path / "failure.json"
    report = create_zarr(dataset, target, spec, output=output)
    assert not report.coverage["conformance_passed"]
    assert not report.coverage["written"]
    assert not target.exists()
    assert any(c.status == "fail" for c in report.checks)
    xr.testing.assert_identical(dataset, before)
    assert json.loads(output.read_text()) == report.to_dict()


def test_dry_run_accepts_yaml_and_writes_only_requested_report(published_schema, tmp_path):
    path, _, schema, _ = published_schema
    target = tmp_path / "preview" / path.name
    with xr.open_zarr(path, chunks=None, mask_and_scale=False) as dataset:
        before = {p for p in tmp_path.rglob("*")}
        report = create_zarr(dataset, target, schema, dry_run=True)
        assert report.coverage["conformance_passed"], [(c.check_id, c.status, c.evidence) for c in report.checks if c.status not in ("pass", "not-applicable", "warning")]
        assert not report.coverage["written"]
        assert {p for p in tmp_path.rglob("*")} == before


def test_existing_target_and_reports_are_protected(published_schema, tmp_path):
    path, spec, schema, _ = published_schema
    with xr.open_zarr(path, chunks=None, mask_and_scale=False) as dataset:
        before = {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
        report = create_zarr(dataset, path, spec)
        assert not report.coverage["written"]
        assert any(c.check_id == "create.destination_available" and c.status == "fail" for c in report.checks)
        assert before == {p: p.read_bytes() for p in path.rglob("*") if p.is_file()}
        for output in (path / "report.json", schema, tmp_path / "new.zarr" / "report.json"):
            with pytest.raises(ValueError):
                create_zarr(dataset, tmp_path / "new.zarr", spec, output=output)
        existing = tmp_path / "existing.json"
        existing.write_text("preserve")
        with pytest.raises(FileExistsError):
            create_zarr(dataset, tmp_path / "new.zarr", spec, output=existing)
        assert existing.read_text() == "preserve"


@pytest.mark.parametrize("options", [{"encoding": {}}, {"chunks": {"time": 2}}, {"mode": "w"}, {"compute": False}, {"zarr_format": 2}])
def test_storage_options_cannot_override_spec(published, tmp_path, options):
    path, spec = published
    with xr.open_zarr(path, chunks=None, mask_and_scale=False) as dataset:
        with pytest.raises(ValueError, match="cannot override"):
            create_zarr(dataset, tmp_path / "target.zarr", spec, write_options=options)
    assert not (tmp_path / "target.zarr").exists()


def test_create_requires_dataset_and_spec(published, tmp_path):
    path, spec = published
    with pytest.raises(TypeError, match="xr.Dataset"):
        create_zarr(path, tmp_path / "target.zarr", spec)
    with pytest.raises(TypeError):
        create_zarr(xr.Dataset(), tmp_path / "target.zarr")
