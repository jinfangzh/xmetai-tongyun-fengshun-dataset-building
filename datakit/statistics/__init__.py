"""Statistics for arrays and calendar-based climatology."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import xarray as xr


XarrayObject = xr.Dataset | xr.DataArray
Dimension = str | Sequence[str] | None
DEFAULT_SAMPLE_DIMS = ("time", "lat", "lon")
__all__ = [
    "climate_slot_mean",
    "compute_mean_std",
    "compute_quantiles",
    "month_day_quantiles",
]


def _reduction_dims(obj: XarrayObject, dim: Dimension) -> tuple[str, ...]:
    if dim is None:
        dimensions = tuple(name for name in DEFAULT_SAMPLE_DIMS if name in obj.dims)
    elif isinstance(dim, str):
        dimensions = (dim,)
    else:
        dimensions = tuple(dim)

    if len(dimensions) != len(set(dimensions)):
        raise ValueError(f"Reduction dimensions must be unique, got {dimensions!r}")
    missing = tuple(name for name in dimensions if name not in obj.dims)
    if missing:
        raise ValueError(f"Dimensions not found in input: {missing!r}")
    if not dimensions:
        raise ValueError(
            "No reduction dimensions were found; pass dim explicitly for this input"
        )
    return dimensions


def _require_time(obj: XarrayObject) -> xr.DataArray:
    if "time" not in obj.dims or "time" not in obj.coords:
        raise ValueError("Input must contain a 'time' dimension and coordinate")
    return obj["time"]


def _february_29_mask(obj: XarrayObject) -> np.ndarray:
    time = _require_time(obj)
    return np.asarray(((time.dt.month == 2) & (time.dt.day == 29)).values)


def compute_mean_std(
    obj: XarrayObject,
    *,
    dim: Dimension = None,
    skipna: bool = True,
    ddof: int = 0,
) -> tuple[XarrayObject, XarrayObject]:
    """Compute mean and standard deviation without loading or writing data.

    By default, the conventional sample dimensions ``time``, ``lat`` and ``lon``
    are reduced when present. Other dimensions, such as ``channel`` or ``level``,
    are preserved. Pass ``dim`` explicitly for data using different conventions.
    """

    if not isinstance(ddof, int) or isinstance(ddof, bool) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    dimensions = _reduction_dims(obj, dim)
    mean = obj.mean(dim=dimensions, skipna=skipna)
    std = obj.std(dim=dimensions, skipna=skipna, ddof=ddof)
    return mean, std


def compute_quantiles(
    obj: XarrayObject,
    quantiles: Sequence[float],
    *,
    dim: str | Sequence[str],
    skipna: bool = True,
) -> XarrayObject:
    """Compute general quantiles without imposing category semantics."""

    try:
        requested = np.asarray(tuple(float(value) for value in quantiles), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("quantiles must contain numeric values") from exc
    if requested.size == 0:
        raise ValueError("quantiles cannot be empty")
    if not np.isfinite(requested).all() or np.any((requested < 0) | (requested > 1)):
        raise ValueError("quantiles must contain finite values in [0, 1]")
    if np.unique(requested).size != requested.size:
        raise ValueError("quantiles must be unique")
    dimensions = (dim,) if isinstance(dim, str) else tuple(dim)
    if not dimensions or any(name not in obj.dims for name in dimensions):
        raise ValueError(f"Quantile dimensions not found in input: {dimensions}")
    return obj.quantile(requested.tolist(), dim=dimensions, skipna=skipna)


def month_day_quantiles(
    obj: XarrayObject,
    *,
    quantiles: tuple[float, ...] = (1 / 3, 2 / 3),
    dim: str | tuple[str, ...] = "time",
    drop_leap_day: bool = True,
) -> XarrayObject:
    """Compute general quantiles grouped by calendar month and day."""

    time = _require_time(obj)
    source = obj
    if drop_leap_day:
        february_29 = _february_29_mask(obj)
        source = obj.isel(time=np.flatnonzero(~february_29))
        time = source["time"]
    if source.sizes.get("time", 0) == 0:
        raise ValueError("No samples remain for month/day quantiles")
    labels = np.asarray(
        [
            f"{month:02d}-{day:02d}"
            for month, day in zip(time.dt.month.values, time.dt.day.values)
        ],
        dtype=str,
    )
    source = source.assign_coords(month_day=("time", labels))
    results = []
    names = []
    for label, group in source.groupby("month_day"):
        results.append(compute_quantiles(group, quantiles, dim=dim, skipna=True))
        names.append(str(label))
    return xr.concat(results, "month_day").assign_coords(month_day=names)


def climate_slot_mean(
    obj: XarrayObject,
    *,
    include_hour: bool = True,
    include_minute: bool = False,
    leap_day: Literal["drop", "keep", "raise"] = "drop",
) -> XarrayObject:
    """Calculate climatology by calendar month/day and optional wall-clock slot.

    Preserve separate 00/06/12/18 (or minute-level) slots. The output dimension
    is ``climatology_slot`` with month, day, hour and optional minute coordinates.
    """

    if leap_day not in {"drop", "keep", "raise"}:
        raise ValueError("leap_day must be 'drop', 'keep', or 'raise'")
    if include_minute and not include_hour:
        raise ValueError("include_minute=True requires include_hour=True")
    time = _require_time(obj)
    february_29 = _february_29_mask(obj)
    if leap_day == "raise" and february_29.any():
        raise ValueError("Input contains February 29")
    source = obj
    if leap_day == "drop" and february_29.any():
        source = obj.isel(time=np.flatnonzero(~february_29))
        time = source["time"]
    if source.sizes.get("time", 0) == 0:
        raise ValueError("No samples remain for slot climatology")

    month = np.asarray(time.dt.month.values, dtype=np.int16)
    day = np.asarray(time.dt.day.values, dtype=np.int16)
    hour = np.asarray(time.dt.hour.values, dtype=np.int16)
    minute = np.asarray(time.dt.minute.values, dtype=np.int16)
    labels = []
    for month_value, day_value, hour_value, minute_value in zip(month, day, hour, minute):
        label = f"{month_value:02d}-{day_value:02d}"
        if include_hour:
            label += f"-{hour_value:02d}"
        if include_minute:
            label += f"-{minute_value:02d}"
        labels.append(label)
    grouped_source = source.assign_coords(climatology_slot=("time", labels))
    result = grouped_source.groupby("climatology_slot").mean("time", skipna=True)

    unique_labels = [str(value) for value in result.climatology_slot.values]
    first_positions = {label: labels.index(label) for label in unique_labels}
    coordinates: dict[str, tuple[str, np.ndarray]] = {
        "month": (
            "climatology_slot",
            np.asarray(
                [month[first_positions[label]] for label in unique_labels],
                dtype=np.int16,
            ),
        ),
        "day": (
            "climatology_slot",
            np.asarray(
                [day[first_positions[label]] for label in unique_labels],
                dtype=np.int16,
            ),
        ),
    }
    if include_hour:
        coordinates["hour"] = (
            "climatology_slot",
            np.asarray(
                [hour[first_positions[label]] for label in unique_labels],
                dtype=np.int16,
            ),
        )
    if include_minute:
        coordinates["minute"] = (
            "climatology_slot",
            np.asarray(
                [minute[first_positions[label]] for label in unique_labels],
                dtype=np.int16,
            ),
        )
    return result.assign_coords(coordinates).sortby("climatology_slot")
