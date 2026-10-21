from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import xarray as xr

from ._common import PathLike, local_path, path_exists, select_variable


def read_grib(
    path: PathLike,
    *,
    variable: str | None = None,
    engine: str = "cfgrib",
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Open a GRIB file through xarray/cfgrib."""
    if not path_exists(path):
        raise FileNotFoundError(f"GRIB path not found: {path}")

    dataset = xr.open_dataset(local_path(path), engine=engine, **kwargs)
    dataset.load()
    result = select_variable(dataset, variable, source=path)
    dataset.close()
    return result


def read_grib_editions(
    path: PathLike,
    *,
    editions: Sequence[int] = (2, 1),
    decode_timedelta: bool = False,
) -> xr.Dataset:
    """Read available GRIB editions from one file and merge their datasets."""

    datasets = []
    errors = []
    for edition in editions:
        try:
            datasets.append(
                read_grib(
                    path,
                    decode_timedelta=decode_timedelta,
                    backend_kwargs={"filter_by_keys": {"edition": edition}},
                )
            )
        except Exception as exc:
            errors.append(f"edition {edition}: {exc}")
    if not datasets:
        raise RuntimeError("No GRIB messages could be opened; " + "; ".join(errors))
    return xr.merge(datasets, compat="no_conflicts") if len(datasets) > 1 else datasets[0]


def write_grib(
    data: xr.Dataset | xr.DataArray,
    path: PathLike,
    **kwargs: Any,
) -> str:
    """Write an xarray object to GRIB through cfgrib.xarray_to_grib."""
    try:
        from cfgrib.xarray_to_grib import to_grib
    except ImportError as exc:
        raise ImportError(
            "write_grib requires cfgrib with write support (cfgrib.xarray_to_grib). "
            "Install cfgrib and ecCodes first."
        ) from exc

    if not isinstance(data, (xr.Dataset, xr.DataArray)):
        raise TypeError("data must be xarray.Dataset or xarray.DataArray")

    to_grib(data, local_path(path, create_parent=True), **kwargs)
    return str(path)
