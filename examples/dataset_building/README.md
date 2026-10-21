# Dataset-building examples

This directory contains separate, runnable dataset-building workflows:

- `era5_nc_to_zarr.py` builds a normalized Zarr dataset from ERA5 NetCDF files.
- `legacy_zarr_refresh/` contains the self-contained legacy Zarr refresh,
  validation and auxiliary-file migration example.

The refresh workflow deliberately lives outside the `datakit` package while
its dataset contract is under review. See
[`legacy_zarr_refresh/README.md`](legacy_zarr_refresh/README.md) for commands,
scope and test-data policy.
