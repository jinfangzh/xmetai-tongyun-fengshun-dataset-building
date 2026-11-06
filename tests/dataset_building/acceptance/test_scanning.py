"""Regression checks for standard dataset scanning."""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pytest
import zarr

from datakit.runtime import memory
from datakit.validation.zarr_inspection import read_zarr_metadata, zarr_chunk_path
from pipelines.dataset import validate as dataset_validation

from ._helpers import _assert_counts, _checks, _receipt, validate_fixture


def test_full_publication_and_quick_scope(published):
    path, spec = published
    full = validate_fixture(path, spec, mode="full")
    assert full.release_passed, json.dumps(full.to_dict(), indent=2)
    assert full.coverage["completed_ranges"] == [[0, 11]]
    quick = validate_fixture(path, spec, mode="quick")
    assert quick.to_dict()["quick_passed"]
    assert quick.coverage["value_checked"] == 3
    assert not quick.coverage["scan_complete"] and not quick.release_passed
    _assert_counts(full)
    _assert_counts(quick)


def test_unselected_numeric_corruption_is_full_failure(published):
    path, spec = published
    array = zarr.open_array(str(path / "data"), mode="r+")
    array[1, 0, 0, 0] = np.nan
    assert validate_fixture(path, spec, mode="quick").to_dict()["quick_passed"]
    full = validate_fixture(path, spec, mode="full")
    assert full.coverage["scan_complete"] and not full.release_passed
    assert _checks(full)["quality.t2m"] == "fail"


def test_missing_key_fill_is_not_production_completion(published):
    path, spec = published
    zarr_chunk_path(read_zarr_metadata(path), "data", (1, 0, 0, 0)).unlink()
    full = validate_fixture(path, spec, mode="full")
    assert full.coverage["value_checked"] == 12
    assert full.coverage["unapproved_missing"] == 1
    assert not full.coverage["scan_complete"] and not full.release_passed
    _assert_counts(full)


def test_approved_sparse_requires_frozen_production_receipt(published):
    path, spec = published
    index = (1, 0, 0, 0)
    zarr_chunk_path(read_zarr_metadata(path), "data", index).unlink()
    spec = replace(
        spec,
        allowed_omitted=(index,),
        quality=tuple(replace(rule, allow_constant=True) for rule in spec.quality),
    )
    assert not validate_fixture(path, spec, mode="full").release_passed
    spec = _receipt(path, spec)
    full = validate_fixture(path, spec, mode="full")
    assert full.release_passed
    assert full.coverage["allowed_omitted"] == 1
    _assert_counts(full)


def test_corrupt_chunk_and_partial_full_keep_complete_counters(published):
    path, spec = published
    zarr_chunk_path(read_zarr_metadata(path), "data", (1, 0, 0, 0)).write_bytes(
        b"broken compressed bytes"
    )
    full = validate_fixture(path, spec, mode="full")
    assert full.coverage["read_failed"] == 1 and not full.release_passed
    assert full.coverage["physically_present"] == 12
    partial = validate_fixture(path, spec, mode="full", max_blocks=2)
    assert partial.coverage["interrupted"] and partial.coverage["not_scanned"] == 10
    assert not partial.coverage["scan_complete"]
    _assert_counts(full)
    _assert_counts(partial)


def test_consolidated_staleness_and_concurrent_metadata_change(published):
    path, spec = published
    file = path / "data" / "zarr.json"
    original = json.loads(file.read_text())
    changed = {**original, "attributes": {**original["attributes"], "note": "changed"}}
    file.write_text(json.dumps(changed))
    full = validate_fixture(path, spec, mode="full")
    assert _checks(full)["metadata.consolidated_matches_raw"] == "fail"
    assert not full.coverage["scan_complete"]
    file.write_text(json.dumps(original))

    def change(_):
        file.write_text(json.dumps(changed))

    full = validate_fixture(path, spec, mode="full", on_block=change)
    assert _checks(full)["publication.metadata_unchanged"] == "fail"
    assert not full.coverage["scan_complete"]


def test_unaudited_sharding_and_unconfirmed_policy_never_pass(published):
    path, spec = published
    unconfirmed = replace(spec, production_evidence=None, quality=())
    full = validate_fixture(path, unconfirmed, mode="full")
    assert not full.release_passed and not full.coverage["scan_complete"]
    assert _checks(full)["quality.tp"] == "needs-review"
    # Actual valid sharded v3 storage, not just a fake metadata codec name.
    group = zarr.open_group(str(path), mode="r+")
    values = group["data"][:]
    attrs = dict(group["data"].attrs)
    del group["data"]
    group.create_array(
        "data",
        data=values,
        chunks=(1, 3, 3, 4),
        shards=(2, 3, 3, 4),
        dimension_names=("time", "channel", "lat", "lon"),
        attributes=attrs,
    )
    zarr.consolidate_metadata(str(path))
    full = validate_fixture(path, spec, mode="full")
    assert not full.coverage["scan_complete"] and not full.release_passed
    assert _checks(full)["storage.physical_support"] == "blocked"


def test_interrupt_callback_memory_budget_and_frozen_hash(published):
    path, spec = published
    entries = []
    report = validate_fixture(
        path, spec, mode="full", on_block=entries.append, stop_requested=lambda: len(entries) >= 2
    )
    assert report.coverage["value_checked"] == 2 and report.coverage["not_scanned"] == 10
    assert report.coverage["interrupted"] and not report.release_passed
    _assert_counts(report)
    report = validate_fixture(path, spec, mode="full", memory_budget_mib=1)
    assert _checks(report)["scan.memory_budget"] == "fail" and report.coverage["value_checked"] == 0
    spec.production_evidence.path.write_text("{}")
    report = validate_fixture(path, spec, mode="full")
    assert _checks(report)["publication.frozen_snapshot"] == "fail" and not report.release_passed


def test_v2_nonsharded_contract(published):
    path, spec = published
    source = zarr.open_group(str(path), mode="r")
    destination = path.parent / "v2.zarr"
    target = zarr.open_group(str(destination), mode="w", zarr_format=2)
    target.attrs.update(dict(source.attrs))
    for name, array in source.arrays():
        attrs = {**dict(array.attrs), "_ARRAY_DIMENSIONS": list(array.metadata.dimension_names)}
        target.create_array(
            name,
            data=array[:],
            chunks=array.chunks,
            attributes=attrs,
            config={"write_empty_chunks": True},
        )
    zarr.consolidate_metadata(str(destination))
    report = validate_fixture(destination, replace(spec, zarr_format=2), mode="full")
    assert report.release_passed, report.to_dict()


def test_blocked_main_scan_does_not_report_main_quality_pass(published):
    path, spec = published
    group = zarr.open_group(str(path), mode="r+")
    group["channel"][:] = ["tp", "q1000", "t2m"]
    full = validate_fixture(path, spec, mode="full")
    assert full.coverage["value_checked"] == 0
    assert full.to_dict()["data_quality"] == "incomplete"
    assert _checks(full)["quality.main_data_scan"] == "not-run"


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_previous_process_peak_does_not_block_new_scan(published, monkeypatch, mode):
    """Earlier evaluation work must not consume a later scan's memory budget."""
    path, spec = published
    monkeypatch.setattr(memory, "peak_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(memory, "resident_memory_bytes", lambda: 128 * 1024**2)

    report = validate_fixture(path, spec, mode=mode)

    assert report.release_passed if mode == "full" else report.quick_passed, report.to_dict()
    assert report.coverage["value_checked"] == (12 if mode == "full" else 3)
    assert report.coverage["peak_process_memory_bytes"] == 8 * 1024**3
    assert report.coverage["process_resident_before_bytes"] == 128 * 1024**2
    assert report.coverage["sampled_peak_resident_bytes"] == 128 * 1024**2
    _assert_counts(report)


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_current_memory_above_budget_blocks_scan(published, monkeypatch, mode):
    path, spec = published
    monkeypatch.setattr(memory, "resident_memory_bytes", lambda: 5 * 1024**3)

    report = validate_fixture(path, spec, mode=mode)

    assert not (report.release_passed if mode == "full" else report.quick_passed)
    assert _checks(report)["scan.memory_budget"] == "fail"
    assert report.coverage["value_checked"] == 0
    assert report.coverage["not_scanned"] == 12
    _assert_counts(report)


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_current_memory_growth_interrupts_scan(published, monkeypatch, mode):
    path, spec = published
    samples = iter([128 * 1024**2, 128 * 1024**2, 5 * 1024**3])
    monkeypatch.setattr(memory, "resident_memory_bytes", lambda: next(samples))

    report = validate_fixture(path, spec, mode=mode)

    assert not (report.release_passed if mode == "full" else report.quick_passed)
    assert _checks(report)["scan.peak_budget"] == "fail"
    assert report.coverage["interrupted"]
    assert report.coverage["value_checked"] == 2
    assert report.coverage["not_scanned"] == 10
    assert report.coverage["sampled_peak_resident_bytes"] == 5 * 1024**3
    _assert_counts(report)


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_unavailable_resident_measurement_keeps_estimated_budget_guard(
    published, monkeypatch, mode
):
    path, spec = published
    monkeypatch.setattr(memory, "resident_memory_bytes", lambda: None)

    accepted = validate_fixture(path, spec, mode=mode)
    rejected = validate_fixture(path, spec, mode=mode, memory_budget_mib=1)

    assert accepted.release_passed if mode == "full" else accepted.quick_passed, accepted.to_dict()
    assert accepted.coverage["sampled_peak_resident_bytes"] is None
    assert not rejected.release_passed
    assert _checks(rejected)["scan.memory_budget"] == "fail"
    assert rejected.coverage["value_checked"] == 0


@pytest.mark.parametrize("mode", ["quick", "full"])
def test_chunk_change_during_read_blocks_both_modes(published, monkeypatch, mode):
    path, spec = published
    key = zarr_chunk_path(read_zarr_metadata(path), "data", (0, 0, 0, 0))
    original_read = dataset_validation.read_zarr_block

    def read_and_change_chunk(array, selection):
        block = original_read(array, selection)
        if selection[0].start == 0:
            stat = key.stat()
            os.utime(key, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        return block

    monkeypatch.setattr(dataset_validation, "read_zarr_block", read_and_change_chunk)
    report = validate_fixture(path, spec, mode=mode)

    assert _checks(report)["publication.block_changed"] == "fail"
    assert not report.coverage["scan_complete"]
    assert not (report.release_passed if mode == "full" else report.quick_passed)
    _assert_counts(report)
