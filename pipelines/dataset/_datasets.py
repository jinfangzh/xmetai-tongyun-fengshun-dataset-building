"""Private dataset inspection implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import xarray as xr

from datakit.io import read_netcdf, read_zarr
from datakit.schema import (
    ACCUMULATED_VARIABLES,
    LAND_VARIABLES,
    OCEAN_VARIABLES,
    PRESSURE_VARIABLES,
    WATER_VARIABLES,
)


@dataclass(frozen=True)
class DatasetInfo:
    """Channel inventory and aligned sidecar arrays for Zarr sources."""

    channels: tuple[str, ...]
    source_channels: tuple[tuple[str, ...], ...]
    groups: Mapping[str, tuple[int, ...]]
    sidecars: Mapping[str, xr.DataArray]
    fields: Mapping[str, xr.DataArray]


def _data_array(dataset: xr.Dataset | xr.DataArray, variable: str) -> xr.DataArray:
    if isinstance(dataset, xr.DataArray):
        return dataset
    if variable in dataset:
        return dataset[variable]
    if len(dataset.data_vars) == 1:
        return dataset[next(iter(dataset.data_vars))]
    raise KeyError(
        f"Variable {variable!r} not found; available: {sorted(dataset.data_vars)}"
    )


def _channel_dimension(array: xr.DataArray) -> str:
    if "channel" in array.dims:
        return "channel"
    if "level" in array.dims:
        return "level"
    raise ValueError("Expected a 'channel' or 'level' dimension")


def _one_dimensional_sidecar(
    value: Any,
    channels: Sequence[str],
    *,
    name: str,
) -> xr.DataArray:
    if isinstance(value, xr.Dataset):
        if len(value.data_vars) != 1:
            raise ValueError(f"Sidecar {name!r} must contain one variable")
        value = value[next(iter(value.data_vars))]
    if isinstance(value, xr.DataArray):
        array = value.squeeze(drop=True)
        for dimension in ("channel", "level"):
            if dimension in array.dims:
                labels = [str(item) for item in array[dimension].values.tolist()]
                if all(channel in labels for channel in channels):
                    selected = array.sel({dimension: list(channels)})
                    if dimension != "channel":
                        selected = selected.rename({dimension: "channel"})
                    return selected.load()
        values = np.asarray(array.values).reshape(-1)
    else:
        values = np.asarray(value).reshape(-1)
    if values.size != len(channels):
        raise ValueError(
            f"Sidecar {name!r} length {values.size} does not match "
            f"{len(channels)} channels"
        )
    return xr.DataArray(
        values,
        dims="channel",
        coords={"channel": list(channels)},
        name=name,
    )


def _load_sidecar(
    source: Path | None,
    dataset: xr.Dataset | xr.DataArray,
    array: xr.DataArray,
    channels: Sequence[str],
    name: str,
) -> xr.DataArray | None:
    netcdf_path = source / f"{name}.nc" if source is not None else None
    numpy_path = source / f"{name}.npy" if source is not None else None
    if netcdf_path is not None and netcdf_path.is_file():
        opened = read_netcdf(netcdf_path)
        try:
            return _one_dimensional_sidecar(opened, channels, name=name)
        finally:
            opened.close()
    if numpy_path is not None and numpy_path.is_file():
        return _one_dimensional_sidecar(
            np.load(numpy_path, allow_pickle=False), channels, name=name
        )
    if isinstance(dataset, xr.Dataset) and name in dataset:
        return _one_dimensional_sidecar(dataset[name], channels, name=name)
    if name in array.attrs:
        return _one_dimensional_sidecar(array.attrs[name], channels, name=name)
    return None


def _load_field(
    source: Path | None,
    dataset: xr.Dataset | xr.DataArray,
    array: xr.DataArray,
    name: str,
) -> xr.DataArray | None:
    netcdf_path = source / f"{name}.nc" if source is not None else None
    numpy_path = source / f"{name}.npy" if source is not None else None
    if netcdf_path is not None and netcdf_path.is_file():
        opened = read_netcdf(netcdf_path)
        try:
            if isinstance(opened, xr.Dataset):
                if len(opened.data_vars) != 1:
                    raise ValueError(f"Field {name!r} must contain one variable")
                value = opened[next(iter(opened.data_vars))]
            else:
                value = opened
            return value.load().rename(name)
        finally:
            opened.close()
    if numpy_path is not None and numpy_path.is_file():
        values = np.load(numpy_path, allow_pickle=False)
        dimensions = tuple(f"dim_{index}" for index in range(values.ndim))
        return xr.DataArray(values, dims=dimensions, name=name)
    if isinstance(dataset, xr.Dataset) and name in dataset:
        return dataset[name].load().rename(name)
    if name in array.attrs:
        values = np.asarray(array.attrs[name])
        dimensions = tuple(f"dim_{index}" for index in range(values.ndim))
        return xr.DataArray(values, dims=dimensions, name=name)
    return None


def inspect_zarr_datasets(
    sources: Sequence[str | Path],
    *,
    variable: str = "data",
    ignore_channels: Sequence[str] = (),
    sidecar_names: Sequence[str] = ("mean", "std"),
    field_names: Sequence[str] = ("weight", "mask", "land_mask", "const"),
    consolidated: bool | None = None,
) -> DatasetInfo:
    """Inspect channels and load channel-aligned sidecar statistics.

    Sidecar precedence is ``<name>.nc``, ``<name>.npy``, a Zarr variable, then
    an attribute on the selected data variable. ``sidecar_names`` must be
    channel-aligned; ``field_names`` retain their original grid dimensions.
    """

    if not sources:
        raise ValueError("sources cannot be empty")
    overlap = sorted(set(sidecar_names) & set(field_names))
    if overlap:
        raise ValueError(f"Sidecar and field names overlap: {overlap}")
    ignored = {str(name) for name in ignore_channels}
    source_channels: list[tuple[str, ...]] = []
    collected: dict[str, list[xr.DataArray]] = {name: [] for name in sidecar_names}
    fields: dict[str, xr.DataArray] = {}

    for raw_source in sources:
        source = Path(raw_source).expanduser()
        dataset = read_zarr(raw_source, consolidated=consolidated)
        try:
            array = _data_array(dataset, variable)
            dimension = _channel_dimension(array)
            original = tuple(str(name) for name in array[dimension].values.tolist())
            if len(original) != len(set(original)):
                raise ValueError(f"Duplicate channels in {raw_source}: {original}")
            kept = tuple(name for name in original if name not in ignored)
            source_channels.append(kept)
            for name in sidecar_names:
                value = _load_sidecar(source, dataset, array, original, name)
                if value is not None:
                    collected[name].append(value.sel(channel=list(kept)))
            for name in field_names:
                value = _load_field(source, dataset, array, name)
                if value is None:
                    continue
                if name in fields and not fields[name].equals(value):
                    raise ValueError(
                        f"Field {name!r} conflicts across Zarr sources"
                    )
                fields[name] = value
        finally:
            dataset.close()

    channels = tuple(name for group in source_channels for name in group)
    if len(channels) != len(set(channels)):
        duplicates = sorted({name for name in channels if channels.count(name) > 1})
        raise ValueError(f"Duplicate channels across sources: {duplicates}")
    incomplete = {
        name: len(values)
        for name, values in collected.items()
        if values and len(values) != len(sources)
    }
    if incomplete:
        details = ", ".join(
            f"{name} ({count}/{len(sources)} sources)"
            for name, count in sorted(incomplete.items())
        )
        raise ValueError(f"Sidecar arrays are incomplete across sources: {details}")
    sidecars = {
        name: xr.concat(values, "channel")
        for name, values in collected.items()
        if values
    }
    groups_source = {
        "pressure": PRESSURE_VARIABLES,
        "accumulated": ACCUMULATED_VARIABLES,
        "ocean": OCEAN_VARIABLES,
        "land": LAND_VARIABLES,
        "water": WATER_VARIABLES,
    }
    base_names = []
    for channel in channels:
        match = re.match(r"^([A-Za-z][A-Za-z0-9]*?)[_-]?\d+$", channel)
        base_names.append(match.group(1) if match is not None else channel)
    groups = {
        name: tuple(
            index
            for index, (channel, base_name) in enumerate(zip(channels, base_names))
            if channel in members or base_name in members
        )
        for name, members in groups_source.items()
    }
    return DatasetInfo(
        channels=channels,
        source_channels=tuple(source_channels),
        groups=groups,
        sidecars=sidecars,
        fields=fields,
    )
