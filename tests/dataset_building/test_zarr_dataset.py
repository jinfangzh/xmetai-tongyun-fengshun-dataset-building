"""Exercise the public Zarr operations using independent small datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pipelines.dataset import (
    compare_zarr,
    validate_zarr,
)


def _sample(start=0, count=2):
    return xr.Dataset(
        {
            "data": (
                ("time", "channel"),
                np.arange(start * 2, (start + count) * 2, dtype="f4").reshape(count, 2),
            ),
            "mean": ("channel", np.array([1, 2], dtype="f4")),
            "std": ("channel", np.array([2, 3], dtype="f4")),
        },
        coords={
            "time": np.datetime64("2025-01-01", "ns")
            + np.arange(start, start + count) * np.timedelta64(1, "D"),
            "channel": ["a", "b"],
        },
        attrs={"data_representation": "normalized", "dataset_id": "preserve-me"},
    )


def _snapshot(path):
    return {p.relative_to(path): p.read_bytes() for p in Path(path).rglob("*") if p.is_file()}


def _write_sample(dataset, destination, *, write_options=None):
    # Mutation/comparison fixtures deliberately do not claim schema conformance.
    options = dict(write_options or {})
    chunks = options.pop("chunks", None)
    if chunks:
        options["encoding"] = {
            name: {"chunks": tuple(chunks.get(dim, dataset.sizes[dim]) for dim in value.dims)}
            for name, value in dataset.variables.items()
        }
    dataset.to_zarr(destination, mode="w-", **options)
    return destination


def test_compare_uses_optional_mapping_and_exact_label_sets(tmp_path):
    target = _write_sample(_sample(), tmp_path / "target.zarr")
    left = (
        _sample()
        .isel(channel=[1, 0], time=[1, 0])
        .rename({"channel": "level", "data": "prediction"})
    )
    left = left.assign_coords(level=["old_b", "old_a"])
    source = _write_sample(left, tmp_path / "source.zarr")
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(
        "dimensions: {level: channel}\nvariables: {prediction: data}\nchannels: {old_a: a, old_b: b}\n"
    )
    assert not compare_zarr(source, target)["equal"]
    report = compare_zarr(source, target, mapping=mapping, memory_budget_mib=0.0001)
    assert report["complete"] and report["equal"]
    assert report["variables"]["data"]["values_compared"] == 4
    missing = _write_sample(_sample().isel(channel=[0]), tmp_path / "missing.zarr")
    report = compare_zarr(missing, target)
    assert not report["equal"] and not report["complete"]
    assert report["differences"]


def test_compare_values_tolerances_nan_and_large_integers(tmp_path):
    left = xr.Dataset(
        {"x": ("row", [1.0, np.nan, np.inf]), "i": ("row", np.array([2**60, 1, 2], dtype="i8"))}
    )
    right = left.copy(deep=True)
    right["x"][0] = 1.001
    a = _write_sample(left, tmp_path / "a.zarr")
    b = _write_sample(right, tmp_path / "b.zarr")
    assert not compare_zarr(a, b)["equal"]
    assert compare_zarr(a, b, atol=0.002)["equal"]
    right["i"][0] += 1
    c = _write_sample(right, tmp_path / "c.zarr")
    report = compare_zarr(a, c)
    assert report["variables"]["i"]["mismatches"] == 1


def test_validate_entry_accepts_schema_object_and_yaml(published_schema):
    path, spec, schema, _ = published_schema
    expected = validate_zarr(path, spec, output=path.parent / "expected.yaml")
    result = validate_zarr(path, spec)
    assert result.quick_passed == expected.quick_passed
    assert result.dataset_id == expected.dataset_id
    loaded = validate_zarr(path, schema, output=path.parent / "loaded.yaml")
    assert loaded.quick_passed == expected.quick_passed


def test_exact_compare_does_not_round_mixed_large_integer_and_float():
    source = xr.Dataset({"x": ("row", np.array([2**60 + 1], dtype="i8"))})
    target = xr.Dataset({"x": ("row", np.array([2**60], dtype="f8"))})
    result = compare_zarr(source, target)
    assert not result["equal"]
    assert result["variables"]["x"]["mismatches"] == 1
    assert result["variables"]["x"]["max_absolute_error"] == 1


@pytest.mark.parametrize("options", [{"atol": -1}, {"rtol": np.inf}, {"memory_budget_mib": 0}])
def test_comparison_options_are_validated_before_opening_inputs(tmp_path, options):
    with pytest.raises(ValueError, match="Tolerances"):
        compare_zarr(tmp_path / "missing-source.zarr", tmp_path / "missing-target.zarr", **options)
