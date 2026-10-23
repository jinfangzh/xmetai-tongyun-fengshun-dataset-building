"""Comparison helpers for weather arrays from different storage formats."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from datakit.transforms import denormalize_xarray


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


def collect_structure(array: xr.DataArray) -> dict[str, object]:
    """Return a serializable structure summary."""

    return {
        "dims": list(array.dims),
        "sizes": {name: int(size) for name, size in array.sizes.items()},
    }


def load_channel_statistic_from_zarr(
    path: str | Path,
    name: str,
    channels: Sequence[Any],
) -> xr.DataArray:
    """Load one channel statistic vector from Zarr storage.

    Statistics are resolved from top-level arrays (``mean``/``std``).
    """

    import zarr

    candidate = Path(path).expanduser() / name
    if not candidate.exists():
        raise FileNotFoundError(f"Cannot find zarr statistic {name!r} in {path}")
    node = zarr.open(str(candidate), mode="r")
    if isinstance(node, zarr.Group):
        keys = list(node.array_keys())
        if not keys:
            raise FileNotFoundError(f"Cannot find zarr statistic {name!r} in {path}")
        node = node[keys[0]]
    values = np.asarray(node[...]).reshape(-1)
    if len(values) != len(channels):
        raise ValueError(
            f"{name} length {len(values)} does not match channel count {len(channels)}"
        )
    return xr.DataArray(values, dims=("channel",), coords={"channel": list(channels)})


def denormalize_zarr_array(
    array: xr.DataArray,
    zarr_path: str | Path,
    *,
    log_channels: Sequence[str] = (),
) -> xr.DataArray:
    """Denormalize a Zarr-backed channel array using Zarr mean/std statistics."""

    channels = array.channel.values.tolist()
    mean = load_channel_statistic_from_zarr(zarr_path, "mean", channels)
    std = load_channel_statistic_from_zarr(zarr_path, "std", channels)
    return denormalize_xarray(
        array.astype("float32"),
        mean,
        std,
        channel_dim="channel",
        log_channels=log_channels,
    )


def validate_structure(
    array: xr.DataArray,
    *,
    required_dims: Sequence[str] = ("time", "channel", "lat", "lon"),
) -> None:
    """Validate required dimensions exist and are non-empty."""

    missing = [dim for dim in required_dims if dim not in array.dims]
    if missing:
        raise ValueError(f"Missing required dimensions: {missing}")

    for dim in required_dims:
        if array.sizes.get(dim, 0) <= 0:
            raise ValueError(f"Dimension {dim!r} is empty")


def validate_coordinate_values(array: xr.DataArray) -> None:
    """Validate core coordinates exist, are non-empty and unique."""

    for coord in ("time", "lat", "lon"):
        if coord not in array.coords:
            raise ValueError(f"Missing coordinate: {coord!r}")
        values = np.asarray(array[coord].values)
        if values.size == 0:
            raise ValueError(f"Coordinate {coord!r} is empty")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Coordinate {coord!r} contains duplicates")

    times = pd.DatetimeIndex(pd.to_datetime(array.time.values))
    if not times.is_monotonic_increasing:
        raise ValueError("Time coordinate is not monotonically increasing")


def validate_channel_values(array: xr.DataArray) -> None:
    """Validate channel coordinates are present, non-empty and unique."""

    if "channel" not in array.coords:
        raise ValueError("Missing coordinate: 'channel'")
    values = array.channel.values.tolist()
    if not values:
        raise ValueError("Channel coordinate is empty")
    as_text = [str(item) for item in values]
    if len(set(as_text)) != len(as_text):
        raise ValueError("Channel coordinate contains duplicates")


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


def validate_same_layout(left: xr.DataArray, right: xr.DataArray) -> None:
    """Ensure arrays have identical shape and dimension order before elementwise compare."""

    if left.dims != right.dims:
        raise ValueError(
            f"Dimension order mismatch: left={left.dims}, right={right.dims}"
        )
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: left={left.shape}, right={right.shape}")


def compute_bias_mean(left: xr.DataArray, right: xr.DataArray) -> float:
    """Compute elementwise bias mean over finite pairs only."""

    validate_same_layout(left, right)
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


def save_report_json(path: str | Path, payload: dict[str, object]) -> None:
    """Save a validation report as UTF-8 JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
