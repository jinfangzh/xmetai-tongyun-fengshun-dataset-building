"""Private data-format conversion implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from datakit.io import (
    read_grib,
    read_netcdf,
    read_numpy,
    read_zarr_array,
    read_zarr,
    write_netcdf,
    write_zarr,
)
from datakit.io._common import infer_format, normalize_format
from datakit.schema import channel_attributes
from datakit.transforms import (
    denormalize_xarray,
    enrich_dataset_attrs,
    normalize_xarray,
    preprocess_channel_units,
)


def _open_source(
    path: str | Path,
    *,
    source_format: str | None,
    variable: str | None,
    options: Mapping[str, Any],
) -> xr.Dataset | xr.DataArray:
    """Open one explicitly supported local source for dataset conversion."""

    format_name = normalize_format(source_format) if source_format else infer_format(path)
    if format_name == "grib":
        return read_grib(path, variable=variable, **options)
    if format_name == "netcdf":
        return read_netcdf(path, variable=variable, **options)
    if format_name == "numpy":
        return read_numpy(path, key=variable, **options)
    if format_name == "zarr":
        return read_zarr(path, variable=variable, **options)
    raise ValueError(f"Unsupported source format: {format_name!r}")


def _reject_reserved_options(
    options: Mapping[str, Any],
    reserved: set[str],
    *,
    name: str,
) -> None:
    conflicts = sorted(reserved & set(options))
    if conflicts:
        raise ValueError(f"{name} contains options not allowed here: {conflicts}")


def _validate_sidecar_layout(
    data: xr.Dataset | xr.DataArray,
    sidecars: Mapping[str, Any],
) -> None:
    if not sidecars:
        return

    has_mean = "mean" in sidecars
    has_std = "std" in sidecars
    if has_mean != has_std:
        raise ValueError("sidecars must provide both 'mean' and 'std' together")

    if not has_mean:
        return

    dataset = data if isinstance(data, xr.Dataset) else data.to_dataset(name=data.name or "data")
    if "channel" not in dataset.dims:
        return

    expected = int(dataset.sizes["channel"])
    for key in ("mean", "std"):
        values = np.asarray(sidecars[key]).reshape(-1)
        if values.size != expected:
            raise ValueError(
                f"sidecar '{key}' length {values.size} does not match channel count {expected}"
            )


def _as_data_array(name: str, value: Any) -> xr.DataArray:
    if isinstance(value, xr.Dataset):
        if len(value.data_vars) != 1:
            raise ValueError(f"Sidecar {name!r} dataset must contain exactly one variable")
        value = value[next(iter(value.data_vars))]
    if isinstance(value, xr.DataArray):
        return value.rename(name)
    values = np.asarray(value)
    if values.size == 0:
        raise ValueError(f"Sidecar {name!r} cannot be empty")
    dims = tuple(f"{name}_dim_{index}" for index in range(values.ndim))
    return xr.DataArray(values, dims=dims, name=name)


def _merge_sidecars_into_dataset(
    data: xr.Dataset | xr.DataArray,
    sidecars: Mapping[str, Any],
) -> xr.Dataset:
    dataset = data if isinstance(data, xr.Dataset) else data.to_dataset(name=data.name or "data")
    if not sidecars:
        return dataset
    result = dataset.copy()
    for name, value in sidecars.items():
        result[name] = _as_data_array(name, value)
    return result


def _apply_channel_attributes(
    dataset: xr.Dataset,
    sidecars: Mapping[str, Any],
    *,
    accumulation_hours: float | None = None,
) -> xr.Dataset:
    if "channel" not in dataset.coords or "mean" not in sidecars or "std" not in sidecars:
        return dataset

    result = dataset.copy()
    channels = [str(name) for name in result["channel"].values.tolist()]
    attrs = channel_attributes(
        channels,
        mean=np.asarray(sidecars["mean"]).reshape(-1),
        std=np.asarray(sidecars["std"]).reshape(-1),
        accumulation_hours=accumulation_hours,
    )
    result["channel"].attrs.update(attrs)
    return result
def _validate_normalized_dataset_layout(dataset: xr.Dataset, *, variable: str = "data") -> None:
    if variable in dataset.data_vars and "channel" in dataset[variable].dims:
        expected = int(dataset.sizes["channel"])
        has_mean = "mean" in dataset.data_vars
        has_std = "std" in dataset.data_vars
        if has_mean != has_std:
            raise ValueError("Dataset must contain both 'mean' and 'std' or neither")
        if has_mean and has_std:
            mean = dataset["mean"]
            std = dataset["std"]
            if "channel" not in mean.dims or "channel" not in std.dims:
                raise ValueError("'mean' and 'std' must use 'channel' dimension")
            if int(mean.sizes["channel"]) != expected or int(std.sizes["channel"]) != expected:
                raise ValueError("'mean'/'std' channel length must match data channel length")

    if "mask" in dataset.data_vars:
        mask = dataset["mask"]
        required = {"lat", "lon"}
        if not required.issubset(mask.dims):
            raise ValueError("'mask' must contain 'lat' and 'lon' dimensions")
        if "mask_channel" not in mask.dims:
            raise ValueError("'mask' must contain 'mask_channel' dimension")
        if int(mask.sizes["mask_channel"]) <= 0:
            raise ValueError("'mask_channel' must be non-empty when present in 'mask'")


def convert_to_zarr(
    source: str | Path | xr.Dataset,
    destination: str | Path,
    *,
    source_format: str | None = None,
    variable: str | None = None,
    chunks: Mapping[str, int] | None = None,
    consolidated: bool | None = True,
    mode: str = "w",
    read_options: Mapping[str, Any] | None = None,
    write_options: Mapping[str, Any] | None = None,
    sidecars: Mapping[str, Any] | None = None,
    dataset_attrs: Mapping[str, Any] | None = None,
    include_channel_attributes: bool = True,
    channel_accumulation_hours: float | None = None,
    add_geospatial_ranges: bool = False,
    lat_coord_names: Sequence[str] = ("lat", "latitude"),
    lon_coord_names: Sequence[str] = ("lon", "longitude"),
) -> str:
    """Convert any supported source format to a Zarr store.

    The source format is inferred from its suffix unless ``source_format`` is
    supplied. Format-specific reader arguments, such as a NetCDF engine or
    NumPy dimensions, belong in ``read_options``. Zarr encoding and version
    arguments belong in ``write_options``.

    When ``sidecars`` is provided, each sidecar is merged as a dataset
    variable before writing so all arrays (for example ``data``/``mean``/
    ``std``/``mask``) share one consistent xarray Zarr schema.

    ``dataset_attrs`` can be used to inject shared metadata. When
    ``add_geospatial_ranges`` is true, coordinate ranges are inferred from
    ``lat_coord_names`` and ``lon_coord_names`` and saved as
    ``geospatial_lat_range`` and ``geospatial_lon_range``.
    When ``include_channel_attributes`` is true and ``mean``/``std`` sidecars
    are present, channel metadata is stored on the ``channel`` coordinate.
    """

    if str(source).rstrip("/") == str(destination).rstrip("/"):
        raise ValueError("source and destination must be different paths")

    reader_kwargs = dict(read_options or {})
    writer_kwargs = dict(write_options or {})
    _reject_reserved_options(
        reader_kwargs,
        {"format", "variable"},
        name="read_options",
    )
    _reject_reserved_options(
        writer_kwargs,
        {"chunks", "consolidated", "mode","zarr_format"},
        name="write_options",
    )

    if isinstance(source, xr.Dataset):
        opened = source
        should_close = False
    else:
        opened = _open_source(
            source,
            source_format=source_format,
            variable=variable,
            options=reader_kwargs,
        )
        should_close = True

    try:
        data_to_write = enrich_dataset_attrs(
            opened,
            dataset_attrs=dataset_attrs,
            add_geospatial_ranges=add_geospatial_ranges,
            lat_coord_names=lat_coord_names,
            lon_coord_names=lon_coord_names,
        )
        _validate_sidecar_layout(data_to_write, sidecars or {})
        to_write = _merge_sidecars_into_dataset(data_to_write, sidecars or {})
        if include_channel_attributes:
            to_write = _apply_channel_attributes(
                to_write,
                sidecars or {},
                accumulation_hours=channel_accumulation_hours,
            )
        _validate_normalized_dataset_layout(to_write, variable="data")
        written = write_zarr(
            to_write,
            destination,
            mode=mode,
            consolidated=consolidated,
            chunks=chunks,
            **writer_kwargs,
        )
        return written
    finally:
        if should_close:
            opened.close()


def _load_channel_statistic(
    directory: str | Path,
    name: str,
    channels: Sequence[Any],
) -> xr.DataArray:
    """Load a one-dimensional channel statistic from Zarr, NetCDF or NumPy.

    Priority order:
    1) Zarr top-level statistics (``<name>``)
    2) ``<name>.nc``
    3) ``<name>.npy``

    The returned array always uses a ``channel`` dimension aligned to ``channels``.
    """

    root = Path(directory).expanduser()
    netcdf_path = root / f"{name}.nc"
    numpy_path = root / f"{name}.npy"
    zarr_values = read_zarr_array(root, name, flatten=True)
    if zarr_values is not None:
        value = xr.DataArray(zarr_values, dims=("channel",))
    elif netcdf_path.is_file():
        opened = read_netcdf(netcdf_path)
        try:
            if isinstance(opened, xr.Dataset):
                if len(opened.data_vars) != 1:
                    raise ValueError(f"Expected one variable in {netcdf_path}")
                value = opened[next(iter(opened.data_vars))]
            else:
                value = opened
            if "level" in value.dims and "channel" not in value.dims:
                value = value.rename({"level": "channel"})
            if "channel" not in value.dims:
                if not value.dims:
                    raise ValueError(
                        f"Statistic {netcdf_path} must contain a channel dimension"
                    )
                value = value.rename({value.dims[0]: "channel"})
            value = value.load()
        finally:
            opened.close()
    elif numpy_path.is_file():
        values = np.load(numpy_path, allow_pickle=False).reshape(-1)
        value = xr.DataArray(values, dims=("channel",))
    else:
        raise FileNotFoundError(
            "Statistics not found. Expected one of: "
            f"Zarr statistic '{name}' in {root}, {netcdf_path}, or {numpy_path}"
        )

    if value.sizes["channel"] != len(channels):
        raise ValueError(f"{name} length does not match data channels")
    return value.assign_coords(channel=list(channels))


def _load_channel_statistic_from_dataset(
    dataset: xr.Dataset,
    name: str,
    channels: Sequence[Any],
) -> xr.DataArray | None:
    if name not in dataset.data_vars:
        return None
    value = dataset[name]
    if "level" in value.dims and "channel" not in value.dims:
        value = value.rename({"level": "channel"})
    if "channel" not in value.dims:
        return None
    selected = value.sel(channel=list(channels)).load()
    if selected.sizes["channel"] != len(channels):
        raise ValueError(f"{name} length does not match data channels")
    return selected


def _parse_day_from_name(path: Path) -> date:
    return datetime.strptime(path.stem, "%Y%m%d").date()


def _parse_day(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    if "-" in text:
        return date.fromisoformat(text)
    return datetime.strptime(text, "%Y%m%d").date()


def _in_day_range(path: Path, start_day: date | None, end_day: date | None) -> bool:
    current = _parse_day_from_name(path)
    if start_day is not None and current < start_day:
        return False
    if end_day is not None and current > end_day:
        return False
    return True


def _load_channel_stat_array(path: str | Path, variable: str) -> xr.DataArray:
    opened = read_netcdf(path)
    try:
        if isinstance(opened, xr.DataArray):
            value = opened
        else:
            if variable not in opened:
                raise KeyError(f"{variable!r} not found in {path}")
            value = opened[variable]
        if "channel" not in value.dims:
            raise ValueError(f"{path} must contain a 'channel' dimension")
        return value.load().astype("float32")
    finally:
        if hasattr(opened, "close"):
            opened.close()


def _load_any_array(path: str | Path, variable: str) -> xr.DataArray:
    opened = read_netcdf(path)
    try:
        if isinstance(opened, xr.DataArray):
            value = opened
        else:
            if variable not in opened:
                raise KeyError(f"{variable!r} not found in {path}")
            value = opened[variable]
        return value.load()
    finally:
        if hasattr(opened, "close"):
            opened.close()


def convert_daily_netcdf_to_normalized_zarr(
    source_dir: str | Path,
    destination: str | Path,
    *,
    mean_nc: str | Path,
    std_nc: str | Path,
    mask_nc: str | Path,
    variable: str = "data",
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    levels: Sequence[str] | None = None,
    accumulation_hours: float = 24.0,
    chunks: Mapping[str, int] | None = None,
    consolidated: bool | None = True,
    mode: str = "w",
    dataset_attrs: Mapping[str, Any] | None = None,
    include_channel_attributes: bool = True,
    tp_zscore: bool = True,
    add_geospatial_ranges: bool = True,
    lat_coord_names: Sequence[str] = ("lat", "latitude"),
    lon_coord_names: Sequence[str] = ("lon", "longitude"),
    overwrite: bool = False,
) -> dict[str, object]:
    """Convert daily NetCDF files into one normalized Zarr dataset.

    This workflow pipeline performs date filtering, per-file channel unit
    preprocessing, channel-wise normalization, concatenation, and final Zarr
    writing with sidecar statistics arrays.
    """

    source = Path(source_dir).expanduser()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    start_day = _parse_day(start_date)
    end_day = _parse_day(end_date)
    if start_day is not None and end_day is not None and start_day > end_day:
        raise ValueError("start_date cannot be greater than end_date")

    selected_levels = [item.strip() for item in (levels or []) if item and item.strip()]
    nc_files = sorted(path for path in source.glob("*.nc") if _in_day_range(path, start_day, end_day))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {source}")

    target = Path(destination).expanduser()
    if target.exists():
        if not overwrite:
            raise FileExistsError(target)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)

    mean = _load_channel_stat_array(mean_nc, "mean")
    std = _load_channel_stat_array(std_nc, "std")
    mask = _load_any_array(mask_nc, "mask")

    datasets: list[xr.Dataset] = []
    for path in nc_files:
        opened = read_netcdf(path)
        try:
            dataset = opened if isinstance(opened, xr.Dataset) else opened.to_dataset(name=opened.name or variable)
            if variable not in dataset:
                raise KeyError(f"{path} does not contain variable {variable!r}")

            data = dataset[variable].astype("float32")
            data = preprocess_channel_units(
                data,
                channel_dim="channel",
                accumulation_hours=float(accumulation_hours),
            ).astype("float32")

            if selected_levels:
                available_channels = [str(name) for name in data.channel.values.tolist()]
                missing = [name for name in selected_levels if name not in available_channels]
                if missing:
                    raise KeyError(f"{path} missing requested levels: {missing}")
                data = data.sel(channel=selected_levels)

            mean_sel = mean.sel(channel=data.channel)
            std_sel = std.sel(channel=data.channel)
            data = normalize_xarray(
                data,
                mean_sel,
                std_sel + 1e-12,
                channel_dim="channel",
                log_channels=(),
                log_clip=(0.0, 1000.0),
            ).astype("float32")

            day_dataset = data.to_dataset(name=variable)
            if "time" in day_dataset.coords and "time" not in day_dataset.dims:
                day_dataset = day_dataset.expand_dims(time=[pd.Timestamp(day_dataset.time.values)])
            elif "time" not in day_dataset.dims:
                day_dataset = day_dataset.expand_dims(time=[pd.Timestamp(_parse_day_from_name(path))])

            datasets.append(day_dataset.load())
        finally:
            if hasattr(opened, "close"):
                opened.close()

    combined = xr.concat(datasets, dim="time")
    combined[variable] = combined[variable].astype("float16")
    channels = [str(name) for name in combined[variable].channel.values.tolist()]

    merged_attrs = dict(dataset_attrs or {})

    sidecars: dict[str, xr.DataArray] = {"mean": mean, "std": std, "mask": mask}

    written = convert_to_zarr(
        combined,
        target,
        chunks=chunks or {"time": 1},
        consolidated=consolidated,
        mode=mode,
        sidecars=sidecars,
        dataset_attrs=merged_attrs,
        include_channel_attributes=include_channel_attributes,
        channel_accumulation_hours=float(accumulation_hours),
        add_geospatial_ranges=add_geospatial_ranges,
        lat_coord_names=lat_coord_names,
        lon_coord_names=lon_coord_names,
    )

    times = pd.DatetimeIndex(combined.time.values)
    result: dict[str, object] = {
        "input": {
            "source_dir": str(source),
            "start_date": start_day.isoformat() if start_day is not None else None,
            "end_date": end_day.isoformat() if end_day is not None else None,
            "variable": variable,
            "accumulation_hours": float(accumulation_hours),
            "levels": selected_levels,
        },
        "output": {
            "zarr_path": str(written),
            "file_count": len(nc_files),
            "time_start": times.min().isoformat(),
            "time_end": times.max().isoformat(),
            "channel_count": len(channels),
            "channels": channels,
            "sidecar_names": sorted(sidecars.keys()),
        },
        "status": "passed",
    }
    return result


def extract_zarr_to_netcdf(
    zarr_path: str | Path,
    save_path: str | Path,
    *,
    start_date: str | pd.Timestamp,
    time_interval: int = 42,
    channels: Sequence[Any] | None = None,
    levels: Sequence[Any] | None = None,
    variable: str = "data",
    stats_path: str | Path | None = None,
    engine: str | None = None,
) -> str:
    """Extract, denormalize and export a half-open Zarr interval.

    This is a workflow rather than a primitive I/O operation: it
    loads normalization statistics, selects channels and time, reverses the
    normalization transform, and finally writes NetCDF.

    Statistics are resolved with priority:
    1) ``mean``/``std`` variables inside the opened Zarr dataset
       (when ``stats_path`` is omitted)
    2) Zarr top-level arrays (``mean``/``std``)
    3) ``mean.nc``/``std.nc``
    4) ``mean.npy``/``std.npy``

    When ``stats_path`` is omitted and in-dataset statistics are missing,
    the Zarr directory itself is used as fallback.
    """

    if time_interval < 1:
        raise ValueError("time_interval must be positive")
    opened = read_zarr(zarr_path)
    dataset = opened
    try:
        if not isinstance(dataset, xr.Dataset):
            raise TypeError("Expected a Zarr Dataset, not a selected DataArray")
        if variable not in dataset.data_vars:
            raise KeyError(
                f"Variable {variable!r} not found; available: "
                f"{sorted(dataset.data_vars)}"
            )
        if "level" in dataset.dims and "channel" not in dataset.dims:
            dataset = dataset.rename({"level": "channel"})
        if "channel" not in dataset.dims:
            raise ValueError("Input data needs a channel dimension")
        if "time" not in dataset.coords:
            raise ValueError("Input data needs a time coordinate")

        all_channels = dataset.channel.values.tolist()
        if stats_path is None:
            mean = _load_channel_statistic_from_dataset(dataset, "mean", all_channels)
            std = _load_channel_statistic_from_dataset(dataset, "std", all_channels)
            if mean is None or std is None:
                mean = _load_channel_statistic(zarr_path, "mean", all_channels)
                std = _load_channel_statistic(zarr_path, "std", all_channels)
        else:
            mean = _load_channel_statistic(stats_path, "mean", all_channels)
            std = _load_channel_statistic(stats_path, "std", all_channels)

        if channels is not None and levels is not None:
            raise ValueError("Use only one of 'channels' or 'levels'")
        selected = list(channels) if channels is not None else list(levels) if levels is not None else []
        if selected:
            missing = [name for name in selected if name not in all_channels]
            if missing:
                raise KeyError(f"Requested channels not found: {missing}")
            dataset = dataset.sel(channel=selected)
            mean = mean.sel(channel=selected)
            std = std.sel(channel=selected)

        start = pd.Timestamp(start_date)
        end = start + pd.Timedelta(days=time_interval)
        dataset = dataset.sel(time=(dataset.time >= start) & (dataset.time < end))
        if dataset.sizes.get("time", 0) == 0:
            raise ValueError(f"No data exists in [{start}, {end})")

        data = denormalize_xarray(
            dataset[variable].astype(np.float32),
            mean,
            std,
            channel_dim="channel",
            log_channels=("tp",),
        ).rename(variable)
        output = data.to_dataset()
        for coordinate in output.coords:
            if output[coordinate].dtype.kind == "O":
                output = output.assign_coords(
                    {coordinate: output[coordinate].astype(str)}
                )

        destination = Path(save_path).expanduser()
        if destination.suffix.lower() != ".nc":
            destination = destination / f"data_{start.strftime('%Y%m%d%H')}.nc"
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoding = (
            {variable: {"zlib": True, "complevel": 4}}
            if engine in {"netcdf4", "h5netcdf"}
            else None
        )
        return write_netcdf(
            output,
            destination,
            engine=engine,
            encoding=encoding,
        )
    finally:
        opened.close()
