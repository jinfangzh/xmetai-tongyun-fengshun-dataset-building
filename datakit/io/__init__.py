"""Explicit local IO functions for supported scientific data formats."""

from .grib import read_grib, read_grib_editions, write_grib
from .netcdf import merge_netcdf, open_netcdf_dataset, read_netcdf, write_netcdf
from .numpy import read_numpy, write_numpy
from .zarr import (
    append_zarr,
    merge_zarr,
    read_zarr_array,
    read_zarr,
    update_zarr_coordinate,
    write_zarr_sidecars,
    write_zarr,
)

from ._zarr_inspection import (
    file_sha256,
    read_zarr_metadata,
    open_zarr_array_readonly,
    read_zarr_block,
    expected_zarr_blocks,
)

__all__ = [
    "file_sha256",
    "read_zarr_metadata",
    "open_zarr_array_readonly",
    "read_zarr_block",
    "expected_zarr_blocks",
    "append_zarr",
    "merge_netcdf",
    "open_netcdf_dataset",
    "merge_zarr",
    "read_grib",
    "read_grib_editions",
    "read_netcdf",
    "read_numpy",
    "read_zarr_array",
    "read_zarr",
    "update_zarr_coordinate",
    "write_grib",
    "write_netcdf",
    "write_numpy",
    "write_zarr_sidecars",
    "write_zarr",
]
