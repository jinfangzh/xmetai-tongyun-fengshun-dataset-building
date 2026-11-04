"""Append B + A into a separate Zarr store, then validate the combined result."""

import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from xarray.coding.times import decode_cf_datetime, encode_cf_datetime

from datakit.schema import DatasetSchema, load_schema, verify_schema_unchanged

from ._sources import _source
from .configuration import DatasetValidationSpec, validation_input_files
from .reports import ValidationReport, ensure_report_destination
from .validate import validate_zarr


def _separate(left, right):
    if left == right or left in right.parents or right in left.parents:
        raise ValueError("Input and destination paths must be separate, non-nested locations")


def _input_paths(source, target, source_config):
    paths = [target]
    if isinstance(source, xr.Dataset):
        for obj in [source, *source.variables.values()]:
            if obj.encoding.get("source"):
                paths.append(Path(obj.encoding["source"]).expanduser().resolve())
    else:
        paths.append(Path(source).expanduser().resolve())
    if isinstance(source_config, (str, Path)):
        paths.append(Path(source_config).expanduser().resolve())
    return paths


def _compatible(old, new):
    if set(old.variables) != set(new.variables):
        raise ValueError("Append requires identical variable and coordinate names")
    if set(old.sizes) != set(new.sizes):
        raise ValueError("Source and target dimensions differ")
    for dim in old.sizes:
        if dim != "time" and old.sizes[dim] != new.sizes[dim]:
            raise ValueError(f"Non-time dimension {dim!r} differs")
    for name in new.variables:
        before, after = old[name], new[name]
        strings = before.dtype.kind in "OUST" and after.dtype.kind in "OUST"
        if before.dims != after.dims or (before.dtype != after.dtype and not strings):
            raise ValueError(f"{after.name}: dimensions or dtype differ")
        before_attrs = {k: v for k, v in before.attrs.items() if k != "_FillValue"}
        after_attrs = {k: v for k, v in after.attrs.items() if k != "_FillValue"}
        if not xr.Dataset(attrs=before_attrs).identical(xr.Dataset(attrs=after_attrs)):
            raise ValueError(f"{after.name}: variable/coordinate metadata differ")
    if "time" not in old.indexes or "time" not in new.indexes:
        raise ValueError("Time append requires explicit time coordinate labels")
    for dataset in (old, new):
        time = dataset["time"]
        if time.dims != ("time",) or time.dtype.kind != "M" or np.isnat(time.values).any():
            raise ValueError("Time coordinates must be one-dimensional datetime64 without NaT")
        index = dataset.indexes["time"]
        if not len(index) or not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError("Both time coordinates must be non-empty, unique and increasing")
    if not new.indexes["time"][0] > old.indexes["time"][-1]:
        raise ValueError("New times overlap or precede the existing range")
    for name in old.variables:
        if "time" not in old[name].dims and not old[name].equals(new[name]):
            raise ValueError(f"Static variable/coordinate {name!r} differs")


def _combined_layout(old, new, spec):
    times = np.concatenate([old.time.values, new.time.values])
    if not np.array_equal(times, spec.expected_times()):
        raise ValueError("B + A times must exactly match the combined target contract")
    layouts = spec.array_layouts()
    if set(old.variables) != set(layouts):
        raise ValueError("Input arrays must match the combined target contract")
    for name, (dims, shape, dtype) in layouts.items():
        array = old[name]
        combined_shape = tuple(len(times) if dim == "time" else old.sizes[dim] for dim in array.dims)
        dtype_ok = array.dtype.kind in "OUST" if dtype == "string" else array.dtype == np.dtype(dtype)
        if array.dims != dims or combined_shape != shape or not dtype_ok:
            raise ValueError(f"{name}: combined layout differs from the target contract")
    if tuple(str(v) for v in old.channel.values) != spec.product.channel_names:
        raise ValueError("Input channels differ from the target contract")
    lat, lon = spec.variant.grid.coordinates()
    if not np.array_equal(old.lat.values, lat) or not np.array_equal(old.lon.values, lon):
        raise ValueError("Input grid differs from the target contract")


def _was_consolidated(target):
    metadata = target / "zarr.json"
    return (target / ".zmetadata").is_file() or (
        metadata.is_file()
        and json.loads(metadata.read_text()).get("consolidated_metadata") is not None
    )


def _append_frame(frame, arrays, index):
    for name, array in arrays.items():
        variable = frame[name]
        values = variable.values
        if values.dtype.kind == "M" and array.dtype.kind != "M":
            # CF-encoded integer time and native Zarr datetime use different
            # representations. Never infer a new epoch from a single frame.
            units = array.attrs["units"]
            calendar = array.attrs.get("calendar", "standard")
            values, _, _ = encode_cf_datetime(
                values, units=units, calendar=calendar, dtype=array.dtype)
            if not np.array_equal(
                decode_cf_datetime(values, units, calendar), variable.values
            ):
                raise ValueError("Existing time encoding cannot represent appended timestamps exactly")
        selection = tuple(slice(index, index + 1) if dim == "time" else slice(None)
                          for dim in variable.dims)
        array[selection] = values


def append_zarr(
    source,
    target,
    destination,
    spec,
    *,
    source_config=None,
    selection=None,
    read_options=None,
    output=None,
) -> ValidationReport:
    """Append source A after existing target B into a new destination C.

    B and A must be non-empty and strictly ordered in time. Variables, dtypes,
    grids and static arrays must match. The required spec describes all of BA,
    including cadence, destination identity, storage layout and publication evidence.

    Copy B without loading its arrays, then load/write A one complete time frame
    at a time. Retain B's storage encoding and refresh identity/version attrs from
    spec. Do not overwrite or replace either input or an existing destination.
    Full validation runs after closing inputs and returns a ValidationReport;
    release_passed determines success. Invalid requests and write errors raise.
    Failed output is retained for inspection. Inputs must be quiescent throughout.
    output optionally saves the final validation report to a new .yaml file.
    """
    target = Path(target).expanduser().resolve()
    requested = Path(destination).expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"Refusing to overwrite destination: {requested}")
    destination = requested.resolve()
    inputs = _input_paths(source, target, source_config)
    for path in inputs:
        _separate(path, destination)
    if not isinstance(source, xr.Dataset):
        _separate(Path(source).expanduser().resolve(), target)
    if not target.is_dir():
        raise ValueError("Existing target B must be a Zarr directory")
    reads = dict(read_options or {})
    if reads.get("load", False) or reads.get("chunks") is not None:
        raise ValueError("Append requires lazy, non-Dask input reads (load=False, chunks=None)")
    if not isinstance(source, xr.Dataset):
        reads.update(load=False, chunks=None)
    if output is not None and Path(output).suffix != ".yaml":
        raise ValueError("Validation report output must use the .yaml suffix")
    if isinstance(spec, (str, Path)):
        spec, _ = load_schema(spec)
    if not isinstance(spec, DatasetSchema):
        raise TypeError("spec must describe the combined target dataset")
    if not isinstance(spec, DatasetValidationSpec):
        spec = DatasetValidationSpec.from_schema(spec)
    spec.validate_configuration()
    if spec.directory_name and destination.name != spec.directory_name:
        raise ValueError("Destination name must match the combined target contract")
    files = validation_input_files(spec)
    for path in files:
        if path is not None:
            _separate(path.resolve(), destination)
    if output is not None:
        report_path = Path(output).expanduser().absolute()
        ensure_report_destination(report_path, directories=[*inputs, destination], files=files)
        if report_path.exists() or report_path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite an existing report: {report_path}")
    if spec.schema_path is not None:
        verify_schema_unchanged(spec)
    with _source(source, source_config, selection, reads) as new:
        with xr.open_zarr(target, chunks=None, consolidated=False, mask_and_scale=False) as old:
            _compatible(old, new)
            _combined_layout(old, new, spec)
            if spec.schema_path is not None:
                verify_schema_unchanged(spec)
            # copytree never overwrites an existing directory, including a racing writer.
            # Files are copied incrementally; no complete BA array is materialized.
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(target, destination)
            consolidated = _was_consolidated(destination)
            group = zarr.open_group(str(destination), mode="r+")
            group.attrs.update(dataset_id=spec.dataset_id, schema_version=spec.schema_version,
                               content_version=spec.content_version, data_revision=spec.data_revision)
            arrays = {}
            for name, variable in new.variables.items():
                if "time" not in variable.dims:
                    continue
                array = group[name].with_config({"write_empty_chunks": True})
                shape = tuple(old.sizes["time"] + new.sizes["time"] if dim == "time"
                              else new.sizes[dim] for dim in variable.dims)
                array.resize(shape)
                arrays[name] = array
            for index in range(new.sizes["time"]):
                frame = new.isel(time=slice(index, index + 1)).load()
                _append_frame(frame, arrays, old.sizes["time"] + index)
            if consolidated:
                zarr.consolidate_metadata(str(destination))
    return validate_zarr(destination, spec, output=output, mode="full")
