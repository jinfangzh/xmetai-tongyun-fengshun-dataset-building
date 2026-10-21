from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import xarray as xr


PathLike = str | Path

_FORMAT_BY_SUFFIX = {
    ".cdf": "netcdf",
    ".grb": "grib",
    ".grb2": "grib",
    ".grib": "grib",
    ".grib2": "grib",
    ".nc": "netcdf",
    ".netcdf": "netcdf",
    ".netcdf4": "netcdf",
    ".npy": "numpy",
    ".npz": "numpy",
    ".zarr": "zarr",
    ".zarr-v2": "zarr",
    ".zarr-v3": "zarr",
}

_FORMAT_ALIASES = {
    "cdf": "netcdf",
    "grb": "grib",
    "grb2": "grib",
    "grib2": "grib",
    "nc": "netcdf",
    "netcdf4": "netcdf",
    "npy": "numpy",
    "npz": "numpy",
    "zarr-v2": "zarr",
    "zarr-v3": "zarr",
}

SUPPORTED_FORMATS = frozenset({"grib", "netcdf", "numpy", "zarr"})


def normalize_format(format_name: str) -> str:
    normalized = _FORMAT_ALIASES.get(format_name.strip().lower(), format_name.strip().lower())
    if normalized not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported data format: {format_name!r}; supported: {sorted(SUPPORTED_FORMATS)}"
        )
    return normalized


def infer_format(path: PathLike) -> str:
    """Infer a supported data format from a path suffix."""
    suffix = local_path(path).suffix.lower()
    try:
        return normalize_format(_FORMAT_BY_SUFFIX[suffix])
    except KeyError as exc:
        raise ValueError(f"Unsupported data suffix: {suffix or '<none>'}") from exc


def local_path(path: PathLike, *, create_parent: bool = False) -> Path:
    """Normalize a local path and reject URL/object-storage paths."""

    raw_path = os.fspath(path)
    if "://" in raw_path:
        raise ValueError(f"Only local filesystem paths are supported: {raw_path}")
    result = Path(raw_path).expanduser()
    if create_parent:
        result.parent.mkdir(parents=True, exist_ok=True)
    return result


def path_exists(path: PathLike) -> bool:
    return local_path(path).exists()


def as_dataset(data: xr.Dataset | xr.DataArray, *, name: str = "data") -> xr.Dataset:
    return data if isinstance(data, xr.Dataset) else data.to_dataset(name=data.name or name)


def select_variable(
    dataset: xr.Dataset,
    variable: str | None,
    *,
    source: PathLike,
) -> xr.Dataset | xr.DataArray:
    if variable is None:
        return dataset
    if variable not in dataset.data_vars:
        available = sorted(dataset.data_vars)
        dataset.close()
        raise KeyError(f"Variable {variable!r} not found in {source}; available: {available}")
    return dataset[variable]
