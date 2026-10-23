"""Private statistics-product pipeline implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from datakit import io as dataset_io
from datakit.transforms import resize

XarrayObject = xr.Dataset | xr.DataArray
StatisticOverrides = Mapping[Any, Sequence[float]]


Dimension = str | Sequence[str] | None
DEFAULT_SAMPLE_DIMS = ("time", "lat", "lon")


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


def _read_cached_statistic(path: Path) -> XarrayObject:
    opened = dataset_io.open(path).read(load=True)
    if isinstance(opened, xr.Dataset) and len(opened.data_vars) == 1:
        name = next(iter(opened.data_vars))
        if name in {"mean", "std", "diff_mean", "diff_std"}:
            return opened[name]
    return opened


def _constant_like(value: xr.DataArray, constant: float) -> xr.DataArray:
    return xr.full_like(value, constant)


def _apply_overrides(
    mean: XarrayObject,
    std: XarrayObject,
    overrides: StatisticOverrides | None,
) -> tuple[XarrayObject, XarrayObject]:
    if not overrides:
        return mean, std
    mean = mean.copy()
    std = std.copy()

    for name, pair in overrides.items():
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise ValueError(f"Override for {name!r} must contain (mean, std)")
        mean_value, std_value = (float(pair[0]), float(pair[1]))
        if not np.isfinite(mean_value) or not np.isfinite(std_value) or std_value <= 0:
            raise ValueError(
                f"Override for {name!r} needs a finite mean and positive finite std"
            )

        applied = False
        if isinstance(mean, xr.Dataset):
            if name in mean.data_vars:
                mean[name] = _constant_like(mean[name], mean_value)
                std[name] = _constant_like(std[name], std_value)
                applied = True
            else:
                channel_dim = next(
                    (
                        dim
                        for dim in ("channel", "level")
                        if dim in mean.dims and name in mean[dim].values.tolist()
                    ),
                    None,
                )
                if channel_dim is not None:
                    for variable in mean.data_vars:
                        if channel_dim in mean[variable].dims:
                            mean[variable].loc[{channel_dim: name}] = mean_value
                            std[variable].loc[{channel_dim: name}] = std_value
                            applied = True
        elif mean.name == name:
            mean = _constant_like(mean, mean_value).rename(mean.name)
            std = _constant_like(std, std_value).rename(std.name)
            applied = True
        else:
            channel_dim = next(
                (
                    dim
                    for dim in ("channel", "level")
                    if dim in mean.dims and name in mean[dim].values.tolist()
                ),
                None,
            )
            if channel_dim is not None:
                mean = mean.copy()
                std = std.copy()
                mean.loc[{channel_dim: name}] = mean_value
                std.loc[{channel_dim: name}] = std_value
                applied = True
        if not applied:
            raise KeyError(f"Override target {name!r} not found in statistics")

    return mean, std


def _validate_standard_deviation(std: XarrayObject) -> None:
    invalid = ~np.isfinite(std) | (std <= 0)
    if isinstance(invalid, xr.Dataset):
        invalid = invalid.to_array()
    if bool(invalid.any().compute().item()):
        raise ValueError("Standard deviation contains non-finite or non-positive values")


def _write_statistic(value: XarrayObject, path: Path, *, name: str) -> str:
    output = value.rename(name) if isinstance(value, xr.DataArray) else value
    return dataset_io.open(path).write(output, mode="w")[0]


def build_normalization_statistics(
    data: XarrayObject,
    output_dir: str | Path,
    *,
    resize_resolution: float | None = None,
    overrides: StatisticOverrides | None = None,
    overwrite: bool = False,
    dim: str | Sequence[str] | None = None,
    skipna: bool = True,
    ddof: int = 0,
) -> tuple[XarrayObject, XarrayObject]:
    """Compute and persist normalization statistics as ``mean.nc``/``std.nc``.

    Spatial resizing, policy overrides, cache reuse and file output are explicit
    workflow choices here; the array reductions above do not perform IO.
    """

    destination = Path(output_dir).expanduser()
    mean_path = destination / "mean.nc"
    std_path = destination / "std.nc"
    if not overwrite and mean_path.is_file() and std_path.is_file():
        cached_mean = _read_cached_statistic(mean_path)
        cached_std = _read_cached_statistic(std_path)
        _validate_standard_deviation(cached_std)
        return cached_mean, cached_std

    source = (
        resize(data, resolution=resize_resolution)
        if resize_resolution is not None
        else data
    )
    mean, std = compute_mean_std(source, dim=dim, skipna=skipna, ddof=ddof)
    mean, std = _apply_overrides(mean, std, overrides)
    _validate_standard_deviation(std)
    if isinstance(mean, xr.DataArray):
        mean = mean.rename("mean")
        std = std.rename("std")

    destination.mkdir(parents=True, exist_ok=True)
    _write_statistic(mean, mean_path, name="mean")
    _write_statistic(std, std_path, name="std")
    return mean, std


def build_difference_statistics(
    data: XarrayObject,
    output_dir: str | Path,
    *,
    resize_resolution: float | None = None,
    overwrite: bool = False,
    time_dim: str = "time",
    dim: str | Sequence[str] | None = None,
    skipna: bool = True,
    ddof: int = 0,
) -> tuple[XarrayObject, XarrayObject]:
    """Compute and persist consecutive-difference mean/std statistics."""

    destination = Path(output_dir).expanduser()
    mean_path = destination / "diff_mean.nc"
    std_path = destination / "diff_std.nc"
    if not overwrite and mean_path.is_file() and std_path.is_file():
        cached_mean = _read_cached_statistic(mean_path)
        cached_std = _read_cached_statistic(std_path)
        _validate_standard_deviation(cached_std)
        return cached_mean, cached_std

    source = (
        resize(data, resolution=resize_resolution)
        if resize_resolution is not None
        else data
    )
    if time_dim not in source.dims:
        raise ValueError(f"Time dimension {time_dim!r} not found in input")
    mean, std = compute_mean_std(
        source.diff(time_dim), dim=dim, skipna=skipna, ddof=ddof
    )
    _validate_standard_deviation(std)
    if isinstance(mean, xr.DataArray):
        mean = mean.rename("diff_mean")
        std = std.rename("diff_std")

    destination.mkdir(parents=True, exist_ok=True)
    _write_statistic(mean, mean_path, name="diff_mean")
    _write_statistic(std, std_path, name="diff_std")
    return mean, std


def _write_analysis(
    value: XarrayObject,
    destination: str | Path,
    *,
    overwrite: bool,
) -> str:
    path = Path(destination).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    if path.suffix.lower() == ".nc":
        return dataset_io.open(path).write(value, mode="w")[0]
    if path.suffix.lower() == ".zarr":
        return dataset_io.open(path).write(value, mode="w", consolidated=True)[0]
    raise ValueError("Analysis destination must end with .nc or .zarr")


def build_slot_climatology(
    data: XarrayObject,
    destination: str | Path,
    *,
    include_hour: bool = True,
    include_minute: bool = False,
    leap_day: Literal["drop", "keep", "raise"] = "drop",
    overwrite: bool = False,
) -> XarrayObject:
    """Compute and persist a month/day/wall-clock climatology."""

    result = climate_slot_mean(
        data,
        include_hour=include_hour,
        include_minute=include_minute,
        leap_day=leap_day,
    )
    _write_analysis(result, destination, overwrite=overwrite)
    return result


def build_month_day_quantiles(
    data: XarrayObject,
    destination: str | Path,
    *,
    quantiles: tuple[float, ...] = (1 / 3, 2 / 3),
    dim: str | tuple[str, ...] = "time",
    drop_leap_day: bool = True,
    overwrite: bool = False,
) -> XarrayObject:
    """Compute and persist general quantiles grouped by calendar month/day."""

    result = month_day_quantiles(
        data,
        quantiles=quantiles,
        dim=dim,
        drop_leap_day=drop_leap_day,
    )
    _write_analysis(result, destination, overwrite=overwrite)
    return result


def build_grouped_zarr_quantiles(
    groups: Mapping[str, Sequence[str | Path]],
    output_dir: str | Path,
    *,
    variable: str = "data",
    quantiles: tuple[float, ...] = (1 / 3, 2 / 3),
    dim: str | tuple[str, ...] = "time",
    overwrite: bool = False,
) -> dict[str, str]:
    """Compute one quantile Zarr product for every explicit source group."""

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for label, sources in groups.items():
        if not sources:
            raise ValueError(f"Group {label!r} has no Zarr sources")
        opened = [dataset_io.open(path).read() for path in sources]
        try:
            arrays = []
            for value in opened:
                if isinstance(value, xr.DataArray):
                    arrays.append(value)
                elif variable in value:
                    arrays.append(value[variable])
                elif len(value.data_vars) == 1:
                    arrays.append(value[next(iter(value.data_vars))])
                else:
                    raise KeyError(
                        f"Variable {variable!r} not found in group {label!r}"
                    )
            combined = xr.concat(arrays, "time", join="exact")
            result = compute_quantiles(combined, quantiles, dim=dim)
            path = destination / f"{label}.zarr"
            outputs[str(label)] = _write_analysis(
                result,
                path,
                overwrite=overwrite,
            )
        finally:
            for value in opened:
                value.close()
    return outputs
