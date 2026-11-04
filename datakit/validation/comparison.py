"""Comparison helpers for weather arrays from different storage formats."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .structure import check_same_layout
from .structure import collect_structure as collect_structure


def normalize_channel_dim(array: xr.DataArray) -> xr.DataArray:
    """Rename ``level`` to ``channel`` when needed for consistent comparisons."""

    if "level" in array.dims and "channel" not in array.dims:
        return array.rename({"level": "channel"})
    return array


def select_data_array(
    obj: xr.Dataset | xr.DataArray,
    variable: str,
    *,
    source_name: str,
) -> xr.DataArray:
    """Select one variable as ``DataArray`` and normalize channel dimension name."""

    if isinstance(obj, xr.DataArray):
        return normalize_channel_dim(obj)
    if variable in obj.data_vars:
        return normalize_channel_dim(obj[variable])
    if len(obj.data_vars) == 1:
        return normalize_channel_dim(obj[next(iter(obj.data_vars))])
    raise KeyError(
        f"Variable {variable!r} not found in {source_name}; "
        f"available: {list(obj.data_vars)}"
    )


def normalize_core_dims(array: xr.DataArray) -> xr.DataArray:
    """Ensure channel naming is normalized and singleton time coordinate becomes a time dim."""

    result = normalize_channel_dim(array)
    if "time" not in result.dims and "time" in result.coords:
        result = result.expand_dims(time=[pd.Timestamp(result.time.values)])
    return result


def compare_coordinates(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    atol: float = 0.0,
) -> dict[str, bool]:
    """Compare time/lat/lon coordinate values and order."""

    result: dict[str, bool] = {}

    left_time = pd.to_datetime(left.time.values)
    right_time = pd.to_datetime(right.time.values)
    result["time"] = np.array_equal(left_time.values, right_time.values)

    left_lat = np.asarray(left.lat.values, dtype=np.float64)
    right_lat = np.asarray(right.lat.values, dtype=np.float64)
    result["lat"] = np.allclose(left_lat, right_lat, atol=atol, rtol=0.0)

    left_lon = np.asarray(left.lon.values, dtype=np.float64)
    right_lon = np.asarray(right.lon.values, dtype=np.float64)
    result["lon"] = np.allclose(left_lon, right_lon, atol=atol, rtol=0.0)

    return result


def compare_channels(left: xr.DataArray, right: xr.DataArray) -> dict[str, Any]:
    """Compare channel names, order and size."""

    left_channels = [str(item) for item in left.channel.values.tolist()]
    right_channels = [str(item) for item in right.channel.values.tolist()]

    left_set = set(left_channels)
    right_set = set(right_channels)

    return {
        "count_equal": len(left_channels) == len(right_channels),
        "order_equal": left_channels == right_channels,
        "missing_in_right": sorted(left_set - right_set),
        "missing_in_left": sorted(right_set - left_set),
        "left_count": len(left_channels),
        "right_count": len(right_channels),
    }


def compute_bias_mean(left: xr.DataArray, right: xr.DataArray) -> float:
    """Compute elementwise bias mean over finite pairs only."""

    failures = [
        check for check in check_same_layout(left, right) if check.status == "fail"
    ]
    if failures:
        raise ValueError(failures[0].evidence)
    finite_left = xr.apply_ufunc(np.isfinite, left, dask="allowed")
    finite_right = xr.apply_ufunc(np.isfinite, right, dask="allowed")
    finite_mask = finite_left & finite_right

    valid_count = int(finite_mask.sum().compute().item())
    if valid_count == 0:
        raise ValueError("No finite overlapping values available for bias calculation")

    delta = (left - right).where(finite_mask)
    delta_sum = float(delta.sum(skipna=True).compute().item())
    return delta_sum / float(valid_count)


def compute_bias_mean_by_channel(
    left: xr.DataArray,
    right: xr.DataArray,
) -> dict[str, float]:
    """Compute per-channel bias mean using finite-value filtering."""

    if "channel" not in left.dims or "channel" not in right.dims:
        raise ValueError("Both arrays must contain 'channel' dimension")

    result: dict[str, float] = {}
    for channel in left.channel.values.tolist():
        result[str(channel)] = float(
            compute_bias_mean(left.sel(channel=channel), right.sel(channel=channel))
        )
    return result


def summarize_statistics(array: xr.DataArray) -> dict[str, float]:
    """Summarize basic value statistics for one array."""

    finite_mask = xr.apply_ufunc(np.isfinite, array, dask="allowed")
    valid_count = int(finite_mask.sum().compute().item())
    if valid_count == 0:
        raise ValueError("Array has no finite values")

    finite_values = array.where(finite_mask)
    minimum = float(finite_values.min(skipna=True).compute().item())
    maximum = float(finite_values.max(skipna=True).compute().item())
    mean = float(finite_values.mean(skipna=True).compute().item())
    std = float(finite_values.std(skipna=True).compute().item())

    nan_count = int(array.isnull().sum().compute().item())
    inf_count = int(
        xr.apply_ufunc(np.isinf, array, dask="allowed").sum().compute().item()
    )

    return {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "nan_count": float(nan_count),
        "inf_count": float(inf_count),
    }


def build_statistics_diff(
    left: dict[str, float],
    right: dict[str, float],
) -> dict[str, float]:
    """Build a serializable statistics difference dictionary: left - right."""

    return {
        "min": float(left["min"] - right["min"]),
        "max": float(left["max"] - right["max"]),
        "mean": float(left["mean"] - right["mean"]),
        "std": float(left["std"] - right["std"]),
        "nan_count": float(left["nan_count"] - right["nan_count"]),
        "inf_count": float(left["inf_count"] - right["inf_count"]),
    }


def _blocks(shape, max_elements):
    block = [1] * len(shape)
    for axis in reversed(range(len(shape))):
        block[axis] = max(1, min(shape[axis], max(1, max_elements)))
        max_elements //= max(1, block[axis])
    for start in product(*(range(0, size, step) for size, step in zip(shape, block))):
        yield tuple(
            slice(pos, min(pos + step, size))
            for pos, step, size in zip(start, block, shape)
        )


def validate_comparison_options(atol, rtol, memory_budget_mib):
    """Validate options before a caller opens any input files."""
    if (
        not np.isfinite([atol, rtol, memory_budget_mib]).all()
        or min(atol, rtol) < 0
        or memory_budget_mib <= 0
    ):
        raise ValueError(
            "Tolerances must be finite and non-negative; memory budget must be positive"
        )


def compare_datasets(
    left: xr.Dataset, right: xr.Dataset, *, atol=0.0, rtol=0.0, memory_budget_mib=64
):
    """Compare every array after exact label alignment, in bounded logical blocks.

    This does not read paths, normalize values, or compare arbitrary attributes
    and storage encodings. Logical block limits do not bound decoder buffers.
    """
    validate_comparison_options(atol, rtol, memory_budget_mib)
    report = {"equal": False, "complete": False, "differences": [], "variables": {}}
    if set(left.variables) != set(right.variables):
        report["differences"].append("Variable/coordinate names differ")
    if dict(left.sizes) != dict(right.sizes):
        report["differences"].append("Dimension names or sizes differ")
    if report["differences"]:
        return report
    indexers = {}
    for dim in right.dims:
        if (dim in left.indexes) != (dim in right.indexes):
            report["differences"].append(f"Coordinate labels missing for {dim}")
        elif dim in right.indexes:
            a, b = left.indexes[dim], right.indexes[dim]
            if not a.is_unique or not b.is_unique:
                report["differences"].append(f"Duplicate coordinate labels for {dim}")
            else:
                positions = a.get_indexer(b)
                if (positions < 0).any():
                    report["differences"].append(
                        f"Coordinate label sets differ for {dim}"
                    )
                elif not a.equals(b):
                    indexers[dim] = positions
    if report["differences"]:
        return report
    left = left.isel(indexers)
    for name in right.variables:
        a, b = left[name], right[name]
        if set(a.dims) != set(b.dims):
            report["differences"].append(f"Variable dimensions differ for {name}")
            continue
        a = a.transpose(*b.dims)
        numeric = a.dtype.kind in "biufc" and b.dtype.kind in "biufc"
        max_elements = max(
            1,
            int(memory_budget_mib * 1024**2)
            // (3 * max(a.dtype.itemsize, b.dtype.itemsize, 8)),
        )
        mismatches, count, max_error = 0, 0, 0.0
        for slices in _blocks(b.shape, max_elements):
            selectors = dict(zip(b.dims, slices))
            x, y = (
                np.asarray(a.isel(selectors).values),
                np.asarray(b.isel(selectors).values),
            )
            count += x.size
            if numeric:
                if atol == rtol == 0:
                    # NumPy promotes mixed int/float or signed/unsigned
                    # arrays, which can erase differences above 2**53.
                    if x.dtype != y.dtype and (
                        x.dtype.kind in "biu" or y.dtype.kind in "biu"
                    ):
                        equal = x.astype(object) == y.astype(object)
                    else:
                        equal = x == y
                    equal = equal | (np.isnan(x) & np.isnan(y))
                else:
                    equal = np.isclose(x, y, atol=atol, rtol=rtol, equal_nan=True)
                finite = np.isfinite(x) & np.isfinite(y)
                if finite.any():
                    dtype = (
                        np.complex128
                        if x.dtype.kind == "c" or y.dtype.kind == "c"
                        else np.longdouble
                    )
                    max_error = max(
                        max_error,
                        float(
                            np.max(
                                np.abs(
                                    x[finite].astype(dtype) - y[finite].astype(dtype)
                                )
                            )
                        ),
                    )
            else:
                equal = x == y
                if x.dtype.kind in "mM" and y.dtype.kind in "mM":
                    equal |= np.isnat(x) & np.isnat(y)
            mismatches += int(np.count_nonzero(~equal))
        report["variables"][name] = {
            "equal": mismatches == 0,
            "values_compared": count,
            "mismatches": mismatches,
            "max_absolute_error": max_error if numeric else None,
        }
        if mismatches:
            report["differences"].append(f"Values differ for {name}: {mismatches}")
    report["complete"] = len(report["variables"]) == len(right.variables)
    report["equal"] = report["complete"] and not report["differences"]
    return report


def compare_channel_values(
    actual: np.ndarray,
    source: np.ndarray,
    *,
    value_factor=1.0,
    value_offset=0.0,
    atol=0.0,
    rtol=0.0,
) -> dict:
    """Compare one channel without clipping, filling or broadcasting."""
    if actual.shape != source.shape:
        return {
            "passed": False,
            "reason": "different raw shapes",
            "actual_shape": actual.shape,
            "source_shape": source.shape,
        }
    expected = source.astype("f4")
    expected *= value_factor
    expected += value_offset
    same_missing = (
        np.array_equal(np.isnan(actual), np.isnan(expected))
        and np.array_equal(np.isposinf(actual), np.isposinf(expected))
        and np.array_equal(np.isneginf(actual), np.isneginf(expected))
    )
    finite = np.isfinite(actual) & np.isfinite(expected)
    difference = np.abs(actual[finite].astype("f4") - expected[finite])
    outside = int((difference > atol + rtol * np.abs(expected[finite])).sum())
    return {
        "passed": same_missing and not outside,
        "missing_pattern_equal": same_missing,
        "outside_tolerance": outside,
        "max_absolute_error": float(difference.max()) if difference.size else None,
        "finite_compared": int(finite.sum()),
        "atol": atol,
        "rtol": rtol,
    }
