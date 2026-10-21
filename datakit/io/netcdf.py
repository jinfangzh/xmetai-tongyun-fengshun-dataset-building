from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Sequence
from pathlib import Path
import tempfile
from typing import Any
from typing import Iterator
import zipfile

import xarray as xr

from ._common import PathLike, local_path, path_exists, select_variable


def _is_zip_wrapped_netcdf(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(4) == b"PK\x03\x04"


@contextmanager
def open_netcdf_dataset(
    path: PathLike,
    **kwargs: Any,
) -> Iterator[xr.Dataset]:
    """Open one local NetCDF dataset, including zip-wrapped `.nc` payloads."""

    local = local_path(path)
    if not local.exists():
        raise FileNotFoundError(f"NetCDF path not found: {path}")

    if not _is_zip_wrapped_netcdf(local):
        with xr.open_dataset(local, **kwargs) as dataset:
            yield dataset
        return

    with tempfile.TemporaryDirectory(prefix="datakit_netcdf_") as temporary:
        with zipfile.ZipFile(local) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ValueError(
                    f"Zip-wrapped NetCDF must contain exactly one file: {local}"
                )
            archive.extract(members[0], temporary)
            extracted = Path(temporary) / members[0]
        with xr.open_dataset(extracted, **kwargs) as dataset:
            yield dataset


def read_netcdf(
    path: PathLike,
    *,
    variable: str | None = None,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Open a local NetCDF file."""
    if not path_exists(path):
        raise FileNotFoundError(f"NetCDF path not found: {path}")

    with open_netcdf_dataset(path, **kwargs) as dataset:
        dataset.load()
        result = select_variable(dataset, variable, source=path)
        return result


def write_netcdf(
    data: xr.Dataset | xr.DataArray,
    path: PathLike,
    **kwargs: Any,
) -> str:
    """Write an xarray object to NetCDF."""
    data.to_netcdf(local_path(path, create_parent=True), **kwargs)
    return str(path)


def merge_netcdf(
    sources: Sequence[PathLike],
    destination: PathLike,
    *,
    concat_dim: str = "time",
    engine: str | None = None,
) -> str:
    """Concatenate compatible NetCDF files along one dimension."""

    if not sources:
        raise ValueError("sources cannot be empty")
    datasets = [read_netcdf(path, engine=engine) for path in sources]
    try:
        result = xr.concat(
            datasets,
            concat_dim,
            join="exact",
            combine_attrs="drop",
        )
        if concat_dim in result.coords:
            result = result.sortby(concat_dim)
        return write_netcdf(result, destination, engine=engine)
    finally:
        for dataset in datasets:
            dataset.close()
