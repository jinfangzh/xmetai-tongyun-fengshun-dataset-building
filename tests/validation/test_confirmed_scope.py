"""Regression checks for standard dataset confirmed scope."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import zarr

from datakit.io._zarr_inspection import read_zarr_metadata, zarr_chunk_path
from pipelines.dataset_building import validate_standard_dataset

from ._helpers import _checks, _receipt


def test_confirmed_format_quick_scope_and_incomplete_scan(published):
    path, original = published
    spec = replace(
        original,
        validation_scope="confirmed-format",
        quality=(),
        production_evidence=None,
        snapshot_id=None,
    )
    quick = validate_standard_dataset(path, spec, mode="quick")
    assert quick.to_dict()["quick_passed"]
    assert _checks(quick)["publication.frozen_snapshot"] == "not-run"
    assert quick.to_dict()["data_integrity"] == "pass"
    assert not quick.release_passed
    partial = validate_standard_dataset(path, spec, mode="quick", max_blocks=1)
    assert not partial.to_dict()["quick_passed"]
    assert not partial.coverage["quick_scan_complete"]
    full = validate_standard_dataset(path, spec, mode="full")
    assert not full.release_passed and not full.coverage["scan_complete"]
    assert validate_standard_dataset(path, _receipt(path, spec), mode="full").release_passed


@pytest.mark.parametrize("fault", ["inf", "negative_tp", "channel_order", "missing_block"])
def test_confirmed_format_preserves_confirmed_failure_gates(published, fault):
    path, original = published
    spec = replace(original, validation_scope="confirmed-format", quality=())
    group = zarr.open_group(str(path), mode="r+")
    if fault == "inf":
        group["data"][0, 0, 0, 0] = np.inf
    elif fault == "negative_tp":
        group["data"][0, 2, 0, 0] = -1
    elif fault == "channel_order":
        group["channel"][:] = ["tp", "q1000", "t2m"]
    else:
        zarr_chunk_path(read_zarr_metadata(path), "data", (0, 0, 0, 0)).unlink()
    quick = validate_standard_dataset(path, spec, mode="quick")
    assert not quick.to_dict()["quick_passed"]
    assert any(check.status == "fail" for check in quick.checks)


def test_confirmed_format_does_not_invent_nan_or_constant_limits(published):
    path, original = published
    spec = replace(original, validation_scope="confirmed-format", quality=())
    group = zarr.open_group(str(path), mode="r+")
    group["data"][0, 0, :, :] = np.nan
    group["data"][0, 1, :, :] = 100
    quick = validate_standard_dataset(path, spec, mode="quick")
    assert quick.to_dict()["quick_passed"]
    assert quick.quality["t2m"]["nan"] == 12
    assert quick.quality["q1000"]["constant_blocks"] == 1


def test_confirmed_format_skips_only_explicit_unconfirmed_semantics(published):
    path, original = published
    channels = tuple(
        replace(c, unconfirmed_fields=("units", "source_units")) if c.name == "t2m" else c
        for c in original.product.channels
    )
    product = replace(original.product, channels=channels, semantic_review=("pending units",))
    spec = replace(original, product=product, validation_scope="confirmed-format", quality=())
    group = zarr.open_group(str(path), mode="r+")
    info = group["channel"].attrs["channel_info"]
    info["t2m"]["units"] = "different"
    group["channel"].attrs["channel_info"] = info
    zarr.consolidate_metadata(str(path))
    assert validate_standard_dataset(path, spec).to_dict()["quick_passed"]
    assert not validate_standard_dataset(
        path, replace(spec, validation_scope="strict-standard", quality=original.quality)
    ).to_dict()["quick_passed"]
    info["t2m"]["level"] = 10
    group["channel"].attrs["channel_info"] = info
    zarr.consolidate_metadata(str(path))
    assert not validate_standard_dataset(path, spec).to_dict()["quick_passed"]
