"""Path-level pipelines for building and inspecting reusable datasets.

This module is the only public pipeline surface.  It combines the package's
explicit IO, in-memory processing and validation modules;
it does not provide a workflow engine, registry, CLI or configuration layer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from datakit import io as dataset_io

from ._conversion import (
    convert_to_zarr,
    extract_zarr_to_netcdf,
)
from ._raw_truth import extract_raw_truth_day
from ._standard_comparison import (
    validate_standard_combination,
    validate_standard_conversion,
)
from ._standard_validation import validate_standard_dataset
from ._validation import validate_zarr_netcdf_consistency


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
        dataset = (
            opened
            if isinstance(opened, xr.Dataset)
            else opened.to_dataset(name=opened.name or "data")
        )
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
            timestamp = pd.Timestamp(
                dataset[time_dim].isel({time_dim: time_idx}).values
            )
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
                outputs.append(
                    dataset_io.open(out_path).write(level_ds, mode="w", engine=engine)[
                        0
                    ]
                )
        return outputs
    finally:
        if should_close and hasattr(opened, "close"):
            opened.close()


__all__ = [
    "convert_to_zarr",
    "extract_zarr_to_netcdf",
    "extract_raw_truth_day",
    "split_netcdf_by_time_and_channel",
    "validate_zarr_netcdf_consistency",
    "validate_standard_dataset",
    "validate_standard_conversion",
    "validate_standard_combination",
]
