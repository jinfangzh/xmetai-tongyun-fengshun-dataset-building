"""Private statistics-product pipeline implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from datakit.io import read_netcdf, read_zarr, write_netcdf, write_zarr
from datakit.statistics import (
    climate_slot_mean,
    compute_mean_std,
    compute_quantiles,
    month_day_quantiles,
)

from datakit.transforms import resize


XarrayObject = xr.Dataset | xr.DataArray
StatisticOverrides = Mapping[Any, Sequence[float]]


def _read_cached_statistic(path: Path) -> XarrayObject:
    opened = read_netcdf(path)
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
    return write_netcdf(output, path)


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
    workflow choices here; :mod:`datakit.statistics` remains pure.
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
        return write_netcdf(value, path)
    if path.suffix.lower() == ".zarr":
        return write_zarr(value, path, mode="w", consolidated=True)
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
        opened = [read_zarr(path) for path in sources]
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
