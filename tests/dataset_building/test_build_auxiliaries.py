"""Attachment copies and build ordering, using plain files rather than Zarr writes."""

import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from datakit.schema import load_schema
from pipelines.dataset import build
from pipelines.dataset.auxiliaries import copy_auxiliaries, prepare_auxiliaries
from pipelines.dataset.reports import ValidationReport

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def attachments(tmp_path):
    spec, _ = load_schema(ROOT / "config/dataset_specs/oras5_c88_p25_h24.yaml")
    spec = replace(spec, first_time="2025-01-01T00", last_time="2025-01-01T00")
    inputs = {}
    for name in ("const.nc", "weight.nc"):
        path = tmp_path / name
        path.write_bytes(f"attachment {name}".encode())
        inputs[name] = path
    return spec, inputs, tmp_path / "output"


def test_attachment_copy_preserves_inputs_and_records_hashes(attachments):
    spec, inputs, destination = attachments
    assets = prepare_auxiliaries(spec, inputs, destination)
    assert not destination.exists()
    destination.mkdir()
    copy_auxiliaries(assets, destination)
    for asset in assets:
        assert len(asset.sha256) == 64
        assert asset.source.read_bytes() == (destination / asset.filename).read_bytes()
    with pytest.raises(FileExistsError):
        copy_auxiliaries(assets, destination)


def test_attachment_source_change_is_rejected(attachments):
    spec, inputs, destination = attachments
    assets = prepare_auxiliaries(spec, inputs, destination)
    inputs["const.nc"].write_bytes(b"changed")
    destination.mkdir()
    with pytest.raises(ValueError, match="changed"):
        copy_auxiliaries(assets, destination)
    assert not (destination / "const.nc").exists()


def test_attachment_inputs_require_declared_existing_files(attachments):
    spec, inputs, destination = attachments
    with pytest.raises(ValueError, match="missing"):
        prepare_auxiliaries(spec, {}, destination)
    assert prepare_auxiliaries(spec, {}, destination, require_all=False) == []
    with pytest.raises(ValueError, match="not declared"):
        prepare_auxiliaries(spec, {**inputs, "unknown.nc": inputs["const.nc"]}, destination)
    inputs["const.nc"].unlink()
    with pytest.raises(FileNotFoundError):
        prepare_auxiliaries(spec, inputs, destination)


@pytest.mark.parametrize("mode", ["write", "dry-run", "missing", "copy-error"])
def test_build_copies_after_write_before_validation(attachments, monkeypatch, mode):
    spec, inputs, destination = attachments
    events = []
    monkeypatch.setattr(build, "load_source_contract", lambda _: {})
    monkeypatch.setattr(build, "validate_source", lambda *_: {})
    monkeypatch.setattr(build, "load_target_contract", lambda _: spec)
    monkeypatch.setattr(build, "read_source", lambda *_: xr.Dataset())
    monkeypatch.setattr(build, "assemble_standard_dataset", lambda product, _: product)

    def fake_create(product, target, spec, *, dry_run):
        events.append("create")
        if not dry_run:
            target.mkdir()
        report = ValidationReport(spec.dataset_id, "create", {},
                                  coverage={"conformance_passed": True, "written": not dry_run})
        report.add("create.input", "pass", "standard_conformance")
        return report

    def fake_validate(target, spec, *, mode):
        events.append("validate")
        for name, source in inputs.items():
            assert (target / name).read_bytes() == source.read_bytes()
        return ValidationReport(spec.dataset_id, "quick", {}, coverage={"quick_scan_complete": True})

    monkeypatch.setattr(build, "create_zarr", fake_create)
    monkeypatch.setattr(build, "validate_zarr", fake_validate)
    if mode == "copy-error":
        def fail_copy(*_):
            raise OSError("simulated attachment copy error")

        monkeypatch.setattr(build, "copy_auxiliaries", fail_copy)
    report = build.build_zarr({}, spec, destination.parent / "source", destination,
                              process=lambda *_: xr.Dataset(), dry_run=mode == "dry-run",
                              auxiliaries={} if mode == "missing" else inputs)
    if mode == "missing":
        assert report.stages["auxiliary_inputs"] == "failed"
        assert events == [] and not destination.exists()
    elif mode == "copy-error":
        assert report.coverage["written"]
        assert report.stages["auxiliary_copy"] == "failed"
        assert events == ["create"]
    elif mode == "dry-run":
        assert report.coverage["conformance_passed"]
        assert report.stages["auxiliary_copy"] == "skipped"
        assert not destination.exists() and events == ["create"]
    else:
        assert report.quick_passed and report.coverage["written"]
        assert report.stages["auxiliary_copy"] == "passed"
        assert report.coverage["auxiliary_inputs"]["const.nc"]["sha256"]
        assert events == ["create", "validate"]


def test_build_releases_source_and_products_before_readback(attachments, monkeypatch):
    spec, inputs, destination = attachments
    references, closed = [], []
    monkeypatch.setattr(build, "load_source_contract", lambda _: {})
    monkeypatch.setattr(build, "validate_source", lambda *_: {})
    monkeypatch.setattr(build, "load_target_contract", lambda _: spec)

    def make_dataset(name):
        dataset = xr.Dataset({"data": ("x", np.arange(1024, dtype="f4"))})
        # Exercise cyclic references, which del/refcounting alone cannot release.
        dataset.attrs["cycle"] = [dataset]
        dataset.set_close(lambda: closed.append(name))
        references.append(weakref.ref(dataset))
        return dataset

    monkeypatch.setattr(build, "read_source", lambda *_: make_dataset("raw"))
    monkeypatch.setattr(build, "assemble_standard_dataset", lambda *_: make_dataset("assembled"))

    def fake_create(product, target, spec, *, dry_run):
        assert closed == []
        target.mkdir()
        report = ValidationReport(spec.dataset_id, "create", {},
                                  coverage={"conformance_passed": True, "written": True})
        report.add("create.input", "pass", "standard_conformance")
        return report

    def fake_validate(target, spec, *, mode):
        assert sorted(closed) == ["assembled", "processed", "raw"]
        assert all(reference() is None for reference in references)
        return ValidationReport(spec.dataset_id, "quick", {}, coverage={"quick_scan_complete": True})

    monkeypatch.setattr(build, "create_zarr", fake_create)
    monkeypatch.setattr(build, "validate_zarr", fake_validate)
    report = build.build_zarr({}, spec, destination.parent / "source", destination,
                              process=lambda *_: make_dataset("processed"), auxiliaries=inputs)
    assert report.quick_passed
