"""Private data-format conversion implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from datakit import io as dataset_io
from datakit.io.backends.files import infer_format, normalize_format
from datakit.transforms import (
    denormalize_xarray,
    enrich_dataset_attrs,
)

from .channel_metadata import channel_attributes


def _open_source(
    path: str | Path,
    *,
    source_format: str | None,
    variable: str | None,
    options: Mapping[str, Any],
) -> xr.Dataset | xr.DataArray:
    """Open one explicitly supported local source for dataset conversion."""

    format_name = (
        normalize_format(source_format) if source_format else infer_format(path)
    )
    options = dict(options)
    if format_name == "numpy":
        options["key"] = variable
    else:
        options["variable"] = variable
    return dataset_io.open(path).read(
        format=format_name, load=format_name != "zarr", **options
    )


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

    dataset = (
        data
        if isinstance(data, xr.Dataset)
        else data.to_dataset(name=data.name or "data")
    )
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
            raise ValueError(
                f"Sidecar {name!r} dataset must contain exactly one variable"
            )
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
    dataset = (
        data
        if isinstance(data, xr.Dataset)
        else data.to_dataset(name=data.name or "data")
    )
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
    if (
        "channel" not in dataset.coords
        or "mean" not in sidecars
        or "std" not in sidecars
    ):
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


def _validate_normalized_dataset_layout(
    dataset: xr.Dataset, *, variable: str = "data"
) -> None:
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
            if (
                int(mean.sizes["channel"]) != expected
                or int(std.sizes["channel"]) != expected
            ):
                raise ValueError(
                    "'mean'/'std' channel length must match data channel length"
                )

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
        {"chunks", "consolidated", "mode", "zarr_format"},
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
        written = dataset_io.open(destination).write(
            to_write,
            mode=mode,
            consolidated=consolidated,
            chunks=chunks,
            **writer_kwargs,
        )[0]
        return written
    finally:
        if should_close:
            opened.close()


def _align_channel_statistic(
    value: xr.DataArray, name: str, channels: Sequence[Any]
) -> xr.DataArray:
    """Align labeled statistics by channel; unlabeled arrays use positional order."""
    if value.dims != ("channel",):
        raise ValueError(f"{name} must be one-dimensional along channel")
    channels = list(channels)
    if "channel" in value.coords:
        if value.channel.dtype.kind == "S":
            value = value.assign_coords(channel=value.channel.astype(str))
        if not value.get_index("channel").is_unique:
            raise ValueError(f"{name} channel labels must be unique")
        missing = [
            channel for channel in channels if channel not in value.get_index("channel")
        ]
        if missing:
            raise ValueError(f"{name} missing requested channels: {missing}")
        return value.sel(channel=channels).load()
    if value.sizes["channel"] != len(channels):
        raise ValueError(f"{name} length does not match data channels")
    return value.assign_coords(channel=channels).load()


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
    value = None
    candidate = root / name
    if candidate.exists() and any(
        (root / marker).is_file() for marker in (".zgroup", "zarr.json")
    ):
        import zarr

        node = zarr.open(str(candidate), mode="r")
        if isinstance(node, zarr.Group):
            keys = list(node.array_keys())
            if len(keys) != 1:
                raise ValueError(
                    f"Statistic {candidate} must contain exactly one array"
                )
            node = node[keys[0]]
        coords = None
        for dimension in ("channel", "level"):
            coordinate = root / dimension
            if coordinate.exists():
                coords = {
                    "channel": np.asarray(zarr.open(str(coordinate), mode="r")[...])
                }
                break
        value = xr.DataArray(np.asarray(node[...]), dims=("channel",), coords=coords)
    if value is not None:
        return _align_channel_statistic(value, name, channels)
    if netcdf_path.is_file():
        opened = dataset_io.open(netcdf_path).read(load=True)
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

    return _align_channel_statistic(value, name, channels)


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
    return _align_channel_statistic(value, name, channels)


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
    opened = dataset_io.open(zarr_path).read()
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
        selected = (
            list(channels)
            if channels is not None
            else list(levels)
            if levels is not None
            else []
        )
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
        return dataset_io.open(destination).write(
            output, mode="w", engine=engine, encoding=encoding
        )[0]
    finally:
        opened.close()
