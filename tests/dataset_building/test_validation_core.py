"""Foundational checks work directly, without a pipeline report or runtime spec."""

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from datakit.schema import DatasetSchema
from datakit.validation import CheckResult, compare_datasets
from datakit.validation.storage import check_raw_block
from datakit.validation.structure import check_dataset_format
from datakit.validation.values import (
    StandardQualityAccumulator,
    check_mask_values,
    check_statistics,
)


def test_foundational_imports_work_when_pipeline_imports_are_forbidden():
    script = """
import importlib.abc
import sys
class RejectPipelineImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pipelines', 'examples', 'torch', 'xmetai'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectPipelineImports())
from datakit.validation import CheckResult, compare_datasets
from datakit.validation import structure, values, storage
from datakit.schema import load_schema
schema, _ = load_schema('config/dataset_specs/oras5_c88_p25_h24.yaml')
assert schema.expected_shape == (11688, 88, 721, 1440)
assert not any(n == 'pipelines' or n.startswith('pipelines.') for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_structure_values_and_quality_accept_only_base_schema(published, monkeypatch):
    path, runtime = published
    schema = DatasetSchema(
        **{field.name: getattr(runtime, field.name) for field in fields(DatasetSchema)}
    )
    assert not hasattr(schema, "snapshot_id")
    with xr.open_zarr(path, chunks=None, mask_and_scale=False) as dataset:
        original = dataset.copy(deep=True).load()
        read_values = xr.DataArray.values.fget

        def values_without_main_data(array):
            if array.name == "data":
                raise AssertionError("Structure checks must not materialize main data")
            return read_values(array)

        with monkeypatch.context() as patch:
            patch.setattr(xr.DataArray, "values", property(values_without_main_data))
            valid, checks = check_dataset_format(dataset, schema)
        assert valid is True
        assert all(isinstance(check, CheckResult) for check in checks)
        assert all(check.status == "pass" for check in checks)
        valid, checks = check_statistics(dataset["mean"].values, dataset["std"].values)
        assert valid and checks[0].check_id == "statistics.finite_positive"
        valid, checks = check_mask_values(dataset["mask"].values, schema)
        assert valid
        accumulator = StandardQualityAccumulator(
            schema,
            dataset["mean"].values,
            dataset["std"].values,
            dataset["mask"].values,
        )
        accumulator.update(dataset.data.isel(time=slice(0, 1)).values)
        checks, metrics = accumulator.finish()
        assert all(check.status == "pass" for check in checks)
        assert set(metrics) == set(schema.product.channel_names)
        assert all(metric["count"] == 12 for metric in metrics.values())
        xr.testing.assert_identical(dataset, original)


def test_plain_dataset_comparison_aligns_labels_and_preserves_large_integers():
    left = xr.Dataset(
        {"x": ("channel", np.array([2**60 + 1, 7], dtype="i8"))},
        coords={"channel": ["a", "b"]},
    )
    assert compare_datasets(left, left.isel(channel=[1, 0]))["equal"]
    rounded = left.assign(x=left.x.astype("f8"))
    report = compare_datasets(left, rounded)
    assert not report["equal"] and report["complete"]
    assert report["variables"]["x"]["mismatches"] == 1


def test_statistics_failure_is_reported_without_mutating_input():
    mean = np.array([1, np.nan], dtype="f4")
    std = np.array([1, 0], dtype="f4")
    before = mean.copy(), std.copy()
    valid, checks = check_statistics(mean, std)
    assert not valid and checks[0].status == "fail"
    np.testing.assert_array_equal(mean, before[0])
    np.testing.assert_array_equal(std, before[1])


def test_raw_block_validation_retains_shape_and_dtype_requirements():
    with pytest.raises(ValueError):
        check_raw_block(np.zeros((2, 2), dtype="f4"), (1, 2), "float32")
    with pytest.raises(ValueError):
        check_raw_block(np.zeros((1, 2), dtype="f8"), (1, 2), "float32")


def test_check_result_rejects_unknown_status():
    with pytest.raises(ValueError, match="Unknown check status"):
        CheckResult("invalid", "unknown", "standard_conformance")
