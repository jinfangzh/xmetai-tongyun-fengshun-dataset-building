"""Append complete time frames to an existing Zarr store."""

import json
from pathlib import Path

import numpy as np
import xarray as xr

from ._sources import _source


def _target_path(source, target):
    target = Path(target).expanduser().resolve()
    if not isinstance(source, xr.Dataset):
        source = Path(source).expanduser().resolve()
        if source == target or source in target.parents or target in source.parents:
            raise ValueError(
                "Source and target paths must be separate, non-nested locations"
            )
    return target


def _compatible(old, new):
    if set(old.variables) != set(new.variables):
        raise ValueError("Append requires identical variable and coordinate names")
    for name in new.variables:
        before, after = old[name], new[name]
        strings = before.dtype.kind in "OUS" and after.dtype.kind in "OUS"
        if before.dims != after.dims or (before.dtype != after.dtype and not strings):
            raise ValueError(f"{after.name}: dimensions or dtype differ")


def _append_options(options):
    options = dict(options or {})
    reserved = {
        "mode",
        "append_dim",
        "region",
        "encoding",
        "chunks",
        "zarr_format",
        "format",
        "compute",
        "consolidated",
    }
    if reserved & options.keys():
        raise ValueError(
            f"Append options cannot override {sorted(reserved & options.keys())}"
        )
    return options


def _was_consolidated(target):
    metadata = target / "zarr.json"
    return (target / ".zmetadata").is_file() or (
        metadata.is_file()
        and json.loads(metadata.read_text()).get("consolidated_metadata") is not None
    )


def _append_dataset(dataset, target, options):
    import zarr

    consolidated = _was_consolidated(target)
    dataset = dataset.copy(deep=False)
    for name in dataset.variables:
        dataset[name].encoding = {}
        # Raw reads expose _FillValue in attrs. Xarray reuses the target's
        # encoding on append, so keeping both would cause a CF encoding conflict.
        dataset[name].attrs = dict(dataset[name].attrs)
        dataset[name].attrs.pop("_FillValue", None)
    dataset.to_zarr(
        str(target), consolidated=False, mode="a-", append_dim="time", **options
    )
    if consolidated:
        zarr.consolidate_metadata(str(target))


def append_zarr(
    source,
    target,
    *,
    source_config=None,
    selection=None,
    read_options=None,
    write_options=None,
) -> Path:
    """Append complete frames along time; retain static arrays and target attrs.

    Both time coordinates must be datetime64, unique, increasing and free of
    NaT. New times must strictly follow the target. Channels, spatial grids and
    all other non-time arrays must match. No cadence is inferred; coverage,
    revision and evidence must be updated separately. Storage encoding is retained.
    """
    target = _target_path(source, target)
    options = _append_options(write_options)
    with _source(source, source_config, selection, read_options) as new:
        with xr.open_zarr(
            target, chunks=None, consolidated=False, mask_and_scale=False
        ) as old:
            target_attrs = dict(old.attrs)
            _compatible(old, new)
            if set(old.sizes) != set(new.sizes):
                raise ValueError("Source and target dimensions differ")
            for dim in old.sizes:
                if dim != "time" and old.sizes[dim] != new.sizes[dim]:
                    raise ValueError(f"Non-time dimension {dim!r} differs")
            if "time" not in old.indexes or "time" not in new.indexes:
                raise ValueError("Time append requires explicit time coordinate labels")
            for dataset in (old, new):
                time = dataset["time"]
                if (
                    time.dims != ("time",)
                    or time.dtype.kind != "M"
                    or np.isnat(time.values).any()
                ):
                    raise ValueError(
                        "Time coordinates must be one-dimensional datetime64 without NaT"
                    )
            before, after = old.indexes["time"], new.indexes["time"]
            if not len(after) or not all(
                i.is_unique and i.is_monotonic_increasing for i in (before, after)
            ):
                raise ValueError(
                    "New times must be non-empty; old and new times must be unique and increasing"
                )
            if len(before) and not after[0] > before[-1]:
                raise ValueError("New times overlap or precede the existing range")
            for name in old.variables:
                if "time" not in old[name].dims and not old[name].equals(new[name]):
                    raise ValueError(f"Static variable/coordinate {name!r} differs")
        new.attrs = target_attrs
        _append_dataset(new, target, options)
    return target
