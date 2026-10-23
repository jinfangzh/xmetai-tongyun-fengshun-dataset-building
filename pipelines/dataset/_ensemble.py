"""Private ensemble dataset-pipeline implementation."""

from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from datakit import io as dataset_io


def scan_ensemble_netcdf_groups(
    data_root: str | Path,
    *,
    pattern: str = "*.nc",
    min_members: int = 2,
) -> dict[str, list[str]]:
    """Group NetCDF files under ``YYYYMMDD`` directories."""

    if min_members < 1:
        raise ValueError("min_members must be positive")
    root = Path(data_root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(root)
    groups = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if len(directory.name) != 8 or not directory.name.isdigit():
            continue
        files = sorted(str(path) for path in directory.rglob(pattern) if path.is_file())
        if len(files) >= min_members:
            groups[directory.name] = files
    return groups


def validate_ensemble_zarr(
    path: str | Path,
    *,
    required_coordinates: Sequence[str] = ("time", "lat", "lon"),
    required_dimensions: Sequence[str] = ("member", "lead_time"),
) -> bool:
    """Return whether an ensemble Zarr store is readable and non-empty."""

    try:
        dataset = dataset_io.open(path).read(consolidated=True)
        try:
            if not isinstance(dataset, xr.Dataset):
                return False
            if not all(name in dataset.coords for name in required_coordinates):
                return False
            if not all(name in dataset.dims for name in required_dimensions):
                return False
            if not dataset.data_vars:
                return False
            for variable in dataset.data_vars:
                if bool(dataset[variable].isnull().all().compute().item()):
                    return False
            return True
        finally:
            dataset.close()
    except Exception:
        return False


def _default_lead_time(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(
            f"Cannot infer lead time from filename {path.name!r}; pass lead_time_parser"
        ) from exc


def _atomic_write_zarr(
    dataset: xr.Dataset,
    destination: Path,
    *,
    chunks: Mapping[str, int] | None,
    overwrite: bool,
) -> str:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        dataset_io.open(temporary).write(dataset, mode="w", consolidated=True, chunks=chunks)[0]
        if destination.exists():
            if not overwrite:
                raise FileExistsError(destination)
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return str(destination)


def convert_ensemble_netcdf_to_zarr(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*.nc",
    min_members: int = 2,
    lead_time_parser: Callable[[Path], Any] | None = None,
    chunks: Mapping[str, int] | None = None,
    overwrite: bool = False,
    skip_valid_existing: bool = True,
) -> dict[str, str]:
    """Convert date-grouped ensemble NetCDF files into atomic Zarr stores.

    Files with the same parsed lead time are concatenated along ``member``;
    lead-time groups are then concatenated along ``lead_time``. This is a
    one-shot workflow: it does not monitor directories or delete sources.
    """

    groups = scan_ensemble_netcdf_groups(
        data_root,
        pattern=pattern,
        min_members=min_members,
    )
    destination_root = Path(output_dir).expanduser()
    destination_root.mkdir(parents=True, exist_ok=True)
    parser = lead_time_parser or _default_lead_time
    outputs = {}
    for date, files in groups.items():
        destination = destination_root / f"{date}.zarr"
        if destination.exists():
            if (
                skip_valid_existing
                and not overwrite
                and validate_ensemble_zarr(destination)
            ):
                outputs[date] = str(destination)
                continue
            if not overwrite:
                raise FileExistsError(destination)
        by_lead: dict[Any, list[Path]] = defaultdict(list)
        for filename in files:
            path = Path(filename)
            by_lead[parser(path)].append(path)
        lead_datasets = []
        lead_labels = []
        for lead_time, members in sorted(by_lead.items(), key=lambda item: item[0]):
            opened = [dataset_io.open(path).read(load=True) for path in sorted(members)]
            try:
                datasets = [
                    value if isinstance(value, xr.Dataset) else value.to_dataset()
                    for value in opened
                ]
                merged = xr.concat(datasets, "member", join="exact").assign_coords(
                    member=np.arange(len(datasets), dtype=np.int32)
                )
                lead_datasets.append(merged.load())
                lead_labels.append(lead_time)
            finally:
                for value in opened:
                    value.close()
        if not lead_datasets:
            raise ValueError(f"No readable NetCDF members for date {date}")
        combined = xr.concat(lead_datasets, "lead_time", join="exact").assign_coords(
            lead_time=lead_labels
        )
        outputs[date] = _atomic_write_zarr(
            combined,
            destination,
            chunks=chunks,
            overwrite=overwrite,
        )
    return outputs
