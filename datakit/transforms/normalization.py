from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr


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
