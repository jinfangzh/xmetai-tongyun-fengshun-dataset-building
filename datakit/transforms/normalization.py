from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from numbers import Real

import numpy as np
import xarray as xr


def compute_mean_std(
    data: xr.DataArray,
    dim: Sequence[str],
    *,
    skipna: bool = True,
    ddof: int = 0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute mean/std along explicit dimensions using float64 reductions.

    Does not load files, inspect contracts, select coverage, cast storage types
    or reject zero/nonfinite results. Callers decide how to use the statistics.
    """
    values = data.astype("float64")
    return (values.mean(dim=dim, skipna=skipna),
            values.std(dim=dim, skipna=skipna, ddof=ddof))


def apply_zscore(
    data: xr.DataArray,
    mean: xr.DataArray | Real,
    std: xr.DataArray | Real,
) -> xr.DataArray:
    """Apply only (data - mean) / std, preserving data labels and attributes.

    Statistics must be finite real scalars or labeled DataArrays over a subset
    of data dimensions. Labels must be unique and identical in membership;
    reordered statistics are selected into data order. No implicit intersection,
    extra dimensions, log transform or clipping. Only statistics are materialized
    for validation; main-data arithmetic remains lazy. No dtype conversion.
    """
    if not isinstance(data, xr.DataArray) or data.dtype.kind not in "iuf":
        raise TypeError("data must be a real numeric DataArray")

    def align_statistic(statistic, name):
        if isinstance(statistic, xr.DataArray):
            if statistic.dtype.kind not in "iuf":
                raise TypeError(f"{name} must contain real numeric values")
            if not set(statistic.dims).issubset(data.dims):
                raise ValueError(f"{name} introduces dimensions absent from data")
            for dim in statistic.dims:
                if dim not in statistic.indexes or dim not in data.indexes:
                    raise ValueError(f"{name}: dimension {dim} requires labeled coordinates")
                source, target = statistic.indexes[dim], data.indexes[dim]
                if (not source.is_unique or not target.is_unique or len(source) != len(target)
                        or not target.isin(source).all()):
                    raise ValueError(f"{name}: labels differ from data on {dim}")
            statistic = statistic.sel({dim: data[dim] for dim in statistic.dims})
            values = np.asarray(statistic.values)
        elif isinstance(statistic, Real) and not isinstance(statistic, (bool, np.bool_)):
            values = np.asarray(statistic)
        else:
            raise TypeError(f"{name} must be a real scalar or labeled DataArray")
        if not np.isfinite(values).all() or (name == "std" and np.any(values <= 0)):
            raise ValueError(f"{name} must be finite; std must also be positive")
        return statistic

    mean = align_statistic(mean, "mean")
    std = align_statistic(std, "std")
    result = ((data - mean) / std).transpose(*data.dims)
    result.name, result.attrs = data.name, dict(data.attrs)
    return result


def _channel_dim(obj: xr.Dataset | xr.DataArray, requested: str | None) -> str:
    if requested is not None:
        if requested not in obj.dims:
            raise ValueError(f"Channel dimension '{requested}' not found")
        return requested
    for candidate in ("channel", "level"):
        if candidate in obj.dims:
            return candidate
    raise ValueError("Expected a 'channel' or 'level' dimension")


def _present_channels(
    obj: xr.Dataset | xr.DataArray,
    dim: str,
    names: Sequence[str],
) -> list[str]:
    available = set(obj[dim].values.tolist())
    return [name for name in names if name in available]


def _validate_std(std: Any) -> None:
    values = np.asarray(std)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Standard deviation must contain finite positive values")


def normalize_xarray(
    obj: xr.Dataset | xr.DataArray,
    mean: Any,
    std: Any,
    *,
    channel_dim: str | None = None,
    log_channels: Sequence[str] = ("tp",),
    log_clip: tuple[float, float] = (0.0, 1000.0),
) -> xr.Dataset | xr.DataArray:
    """Apply log transforms in physical space, then standardize."""

    _validate_std(std)
    dim = _channel_dim(obj, channel_dim)
    result = obj.copy()
    for name in _present_channels(result, dim, log_channels):
        transformed = np.log1p(
            result.sel({dim: name}).clip(min=log_clip[0], max=log_clip[1])
        )
        result.loc[{dim: name}] = transformed
    return (result - mean) / std


def denormalize_xarray(
    obj: xr.Dataset | xr.DataArray,
    mean: Any,
    std: Any,
    *,
    channel_dim: str | None = None,
    log_channels: Sequence[str] = ("tp",),
    exp_clip: tuple[float, float] = (0.0, 7.0),
) -> xr.Dataset | xr.DataArray:
    """Undo standardization and log transforms while preserving channel order."""

    _validate_std(std)
    dim = _channel_dim(obj, channel_dim)
    result = (obj * std + mean).copy()
    for name in _present_channels(result, dim, log_channels):
        restored = np.expm1(
            result.sel({dim: name}).clip(min=exp_clip[0], max=exp_clip[1])
        )
        result.loc[{dim: name}] = restored
    return result
