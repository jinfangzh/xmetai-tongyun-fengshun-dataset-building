"""Reusable helpers for the dataset-building examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from datakit import io as dataset_io
from datakit.validation import RangeReport, analyze_range
from pipelines.dataset import (
    DatasetInfo,
    RadarContinuityReport,
    build_difference_statistics,
    build_grouped_zarr_quantiles,
    build_month_day_quantiles,
    build_normalization_statistics,
    build_slot_climatology,
    convert_cinrad_to_zarr,
    convert_daily_netcdf_to_normalized_zarr,
    convert_ensemble_netcdf_to_zarr,
    convert_meteonet_npz_to_zarr,
    convert_to_zarr,
    extract_raw_truth_day,
    extract_zarr_to_netcdf,
    inspect_zarr_datasets,
)


def save_by_year(
    obj: xr.Dataset | xr.DataArray,
    directory: str | Path,
    *,
    format: str = "zarr",
    dtype: Any = np.float32,
    overwrite: bool = False,
) -> list[str]:
    """Partition a time-indexed object into one local file or store per year."""

    if "time" not in obj.dims or obj.sizes["time"] == 0:
        raise ValueError("Expected a non-empty time dimension")
    format_name = format.strip().lower()
    if format_name not in {"netcdf", "zarr"}:
        raise ValueError("format must be 'zarr' or 'netcdf'")

    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    years = sorted(set(pd.DatetimeIndex(obj.time.values).year.tolist()))
    outputs: list[str] = []
    for year in years:
        subset = obj.sel(time=str(year)).astype(dtype)
        suffix = ".zarr" if format_name == "zarr" else ".nc"
        path = destination / f"{year}{suffix}"
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        outputs.append(dataset_io.open(path).write(subset, mode="w")[0])
    return outputs


def check_zarr_range(
    path: str | Path,
    *,
    variable: str = "data",
    axis: str = "level",
    selection: dict[str, Any] | None = None,
) -> RangeReport:
    """Open a Zarr store and calculate a range report for one variable."""

    opened = dataset_io.open(path).read()
    try:
        if isinstance(opened, xr.DataArray):
            array = opened
            if variable != "data" and opened.name != variable:
                raise KeyError(f"Variable {variable!r} not found")
        else:
            if variable not in opened:
                raise KeyError(
                    f"Variable {variable!r} not found; "
                    f"available: {sorted(opened.data_vars)}"
                )
            array = opened[variable]
        if selection:
            array = array.sel(selection)
        return analyze_range(array, axis=axis)
    finally:
        opened.close()


def split_netcdf_by_time_and_channel(
    source: str | Path | xr.Dataset | xr.DataArray,
    directory: str | Path,
    *,
    time_dim: str = "time",
    channel_dim: str = "channel",
    engine: str | None = "netcdf4",
    overwrite: bool = False,
) -> list[str]:
    """Split one NetCDF-like object into ``<channel>/<year>/<yyyymmdd>.nc`` files."""

    if isinstance(source, (xr.Dataset, xr.DataArray)):
        opened = source
        should_close = False
    else:
        opened = dataset_io.open(source).read(load=True)
        should_close = True

    try:
        dataset = opened if isinstance(opened, xr.Dataset) else opened.to_dataset(name=opened.name or "data")
        if time_dim not in dataset.dims:
            raise ValueError(f"Expected '{time_dim}' dimension")
        if channel_dim not in dataset.dims:
            raise ValueError(
                f"Expected '{channel_dim}' dimension; available dims: {tuple(dataset.dims)}"
            )
        if dataset.sizes.get(time_dim, 0) == 0:
            raise ValueError(f"Expected non-empty '{time_dim}' dimension")
        if dataset.sizes.get(channel_dim, 0) == 0:
            raise ValueError(f"Expected non-empty '{channel_dim}' dimension")

        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)

        outputs: list[str] = []
        for time_idx in range(dataset.sizes[time_dim]):
            timestamp = pd.Timestamp(dataset[time_dim].isel({time_dim: time_idx}).values)
            day_str = timestamp.strftime("%Y%m%d")
            year = timestamp.strftime("%Y")

            day_ds = dataset.isel({time_dim: time_idx})
            for level in day_ds[channel_dim].values.tolist():
                level_ds = day_ds.sel({channel_dim: level}).drop_vars(channel_dim)
                out_dir = destination / str(level) / year
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{day_str}.nc"
                if out_path.exists() and not overwrite:
                    raise FileExistsError(out_path)
                outputs.append(dataset_io.open(out_path).write(level_ds, mode="w", engine=engine)[0])
        return outputs
    finally:
        if should_close and hasattr(opened, "close"):
            opened.close()


__all__ = [
    "DatasetInfo",
    "RadarContinuityReport",
    "build_difference_statistics",
    "build_grouped_zarr_quantiles",
    "build_month_day_quantiles",
    "build_normalization_statistics",
    "build_slot_climatology",
    "check_zarr_range",
    "convert_cinrad_to_zarr",
    "convert_daily_netcdf_to_normalized_zarr",
    "convert_ensemble_netcdf_to_zarr",
    "convert_meteonet_npz_to_zarr",
    "convert_to_zarr",
    "extract_zarr_to_netcdf",
    "extract_raw_truth_day",
    "inspect_zarr_datasets",
    "save_by_year",
    "split_netcdf_by_time_and_channel",
]
