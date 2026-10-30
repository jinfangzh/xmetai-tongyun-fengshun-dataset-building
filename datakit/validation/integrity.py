"""Completeness and invalid-value checks for in-memory xarray objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class RangeReport:
    """Per-slice range and invalid-value summary."""

    variable: str | None
    axis: str
    axis_values: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    mean: np.ndarray
    nan_count: np.ndarray
    inf_count: np.ndarray

    @property
    def total_nan(self) -> int:
        return int(self.nan_count.sum())

    @property
    def total_inf(self) -> int:
        return int(self.inf_count.sum())

    def to_text(self) -> str:
        header = f"variable={self.variable or '<unnamed>'} axis={self.axis}"
        rows = [
            header,
            f"{'value':<24} {'min':>14} {'max':>14} {'mean':>14} "
            f"{'nan':>10} {'inf':>10}",
        ]
        for index, value in enumerate(self.axis_values):
            rows.append(
                f"{str(value):<24} {self.minimum[index]:>14.6g} "
                f"{self.maximum[index]:>14.6g} {self.mean[index]:>14.6g} "
                f"{self.nan_count[index]:>10d} {self.inf_count[index]:>10d}"
            )
        rows.append(f"total_nan={self.total_nan} total_inf={self.total_inf}")
        return "\n".join(rows)


def describe_data(
    obj: xr.Dataset | xr.DataArray,
    *,
    channels: list[Any] | None = None,
    channel_dim: str | None = None,
) -> str:
    """Return a compact shape, coordinate and value-range description."""

    if isinstance(obj, xr.Dataset):
        return "\n".join(
            describe_data(obj[name], channels=channels, channel_dim=channel_dim)
            for name in obj.data_vars
        )
    lines = [f"{obj.name or '<unnamed>'}: shape={obj.shape}, dims={obj.dims}"]
    if "time" in obj.dims and obj.sizes["time"]:
        lines.append(
            f"time={obj.time.values[0]}..{obj.time.values[-1]} ({obj.sizes['time']})"
        )
    if "lat" in obj.dims and "lon" in obj.dims:
        lines.append(
            f"grid=({obj.lat.values[0]}..{obj.lat.values[-1]}) x "
            f"({obj.lon.values[0]}..{obj.lon.values[-1]})"
        )
    dimension = channel_dim or (
        "channel" if "channel" in obj.dims else "level" if "level" in obj.dims else None
    )
    if dimension is None:
        values = np.asarray(obj.values)
        lines.append(f"range={np.nanmin(values):.6g}..{np.nanmax(values):.6g}")
    else:
        available = obj[dimension].values.tolist()
        selected = [name for name in (channels or available) if name in available]
        for name in selected:
            values = np.asarray(obj.sel({dimension: name}).values)
            lines.append(
                f"{dimension}={name}: {np.nanmin(values):.6g}..{np.nanmax(values):.6g}"
            )
    return "\n".join(lines)


def analyze_range(obj: xr.DataArray, *, axis: str = "level") -> RangeReport:
    """Calculate range and invalid counts for every slice along ``axis``."""

    if axis not in obj.dims:
        raise ValueError(f"Axis {axis!r} not found; available dimensions: {obj.dims}")
    reduce_dims = tuple(dim for dim in obj.dims if dim != axis)
    finite = obj.where(np.isfinite(obj))
    computed = xr.merge(
        [
            finite.min(reduce_dims, skipna=True).rename("minimum"),
            finite.max(reduce_dims, skipna=True).rename("maximum"),
            finite.mean(reduce_dims, skipna=True).rename("mean"),
            obj.isnull().sum(reduce_dims).rename("nan_count"),
            xr.apply_ufunc(np.isinf, obj, dask="allowed")
            .sum(reduce_dims)
            .rename("inf_count"),
        ]
    ).compute()
    return RangeReport(
        variable=obj.name,
        axis=axis,
        axis_values=np.asarray(obj[axis].values),
        minimum=np.asarray(computed.minimum.values),
        maximum=np.asarray(computed.maximum.values),
        mean=np.asarray(computed["mean"].values),
        nan_count=np.asarray(computed.nan_count.values, dtype=np.int64),
        inf_count=np.asarray(computed.inf_count.values, dtype=np.int64),
    )


def count_invalid(obj: xr.Dataset | xr.DataArray) -> tuple[int, int]:
    """Return total NaN and infinite-value counts."""

    if isinstance(obj, xr.Dataset):
        reports = [count_invalid(obj[name]) for name in obj.data_vars]
        return sum(item[0] for item in reports), sum(item[1] for item in reports)
    nan_count = obj.isnull().sum().compute()
    inf_count = xr.apply_ufunc(np.isinf, obj, dask="allowed").sum().compute()
    return int(nan_count.item()), int(inf_count.item())


def check_complete_year(
    obj: xr.Dataset | xr.DataArray,
    *,
    frequency: str = "hourly",
) -> bool:
    """Return whether the time coordinate exactly covers one complete year."""

    frequencies = {"hourly": "1h", "six_hourly": "6h", "daily": "1D"}
    if frequency not in frequencies:
        raise ValueError(f"Unsupported frequency: {frequency}")
    times = pd.DatetimeIndex(pd.to_datetime(obj.time.values))
    if times.empty or len(set(times.year)) != 1:
        return False
    year = int(times.year[0])
    expected = pd.date_range(
        start=pd.Timestamp(year=year, month=1, day=1, tz=times.tz),
        end=pd.Timestamp(year=year + 1, month=1, day=1, tz=times.tz),
        inclusive="left",
        freq=frequencies[frequency],
    )
    return times.equals(expected)
