"""Text partition assembly and layout write planning; no physical format parsing."""

import numpy as np
import xarray as xr

from .._layout import templates
from . import babj, station_tables
from ._writing import (
    check_outputs,
    check_source_coordinates,
    check_source_variables,
    project_coordinates,
    staged_output,
    write_mode,
)


FORMATS = frozenset({"diamond_station", "station_probability", "babj"})


def read_records(records, description, **options):
    track = description["format"]["name"] == "babj"
    fixed_metadata = False if track else options.pop("fixed_station_metadata", False)
    if options:
        label = "BABJ" if track else "Text table"
        raise TypeError(f"{label} options are not supported: {sorted(options)}")
    codec = babj if track else station_tables
    probability = "thresholds" in description
    axis = "cycle_time" if probability else "time"
    frames, manifest, tables = [], [], []
    offset = 0
    for record in records:
        dataset, metadata = codec.read_file(record.path, description)
        if track:
            count = dataset.sizes["record"]
            dataset = dataset.assign_coords(record=np.arange(offset, offset + count))
            offset += count
            metadata.update(filename=record.path.name, records=dataset.record.values)
        elif fixed_metadata and not probability:
            rows = np.column_stack([
                dataset.station.values,
                *(dataset[name].values for name in ("lon", "lat", "altitude", "precipitation")),
            ])
            tables.append((record, metadata["headers"], rows))
            continue
        else:
            dataset = dataset.expand_dims({axis: [record.fields[axis]]})
            metadata["fields"] = record.fields
        frames.append(dataset)
        manifest.append(metadata)
    if fixed_metadata and not probability and not track:
        return _station_collection(tables)
    if not frames:
        raise ValueError("No valid station data parsed")
    if not track:
        labels = [frame[axis].values[0] for frame in frames]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Duplicate {axis} records")
    result = xr.concat(
        frames, dim="record" if track else axis,
        join="outer", coords="minimal", compat="equals",
    )
    if not track:
        result = result.sortby(axis)
    result.encoding["dataset_io_manifest"] = manifest
    return result


def write(dataset, root, description, **options):
    mode = write_mode(options)
    track = description["format"]["name"] == "babj"
    codec = babj if track else station_tables
    if options:
        label = "BABJ" if track else "Text table"
        raise TypeError(f"{label} write options are not supported: {sorted(options)}")
    patterns = templates(description)
    if not track and len(patterns) != 1:
        raise ValueError("Text writing requires one reversible filename template")
    manifest = dataset.encoding.get("dataset_io_manifest")
    if not manifest:
        raise NotImplementedError(
            "BABJ output requires source headers retained by read()" if track else
            "Text writing requires headers and record membership retained by read()"
        )
    check_source_variables(dataset, (name for item in manifest for name in item["columns"]))
    axis = "cycle_time" if "thresholds" in description else "time"
    sources = [
        {"record": item["records"]} if track else
        {axis: [item["fields"][axis]], "station": item["stations"]}
        for item in manifest
    ]
    if not track and "thresholds" in description:
        for source in sources:
            source["threshold"] = description["thresholds"]
    check_source_coordinates(dataset, sources)
    plans = []
    for item, selectors in zip(manifest, sources):
        part = project_coordinates(dataset, selectors)
        if part is None:
            continue
        if not track and axis in part.dims:
            part = part.isel({axis: 0}, drop=True)
        part = codec.prepare_file(part, description, item)
        path = root / (item["filename"] if track else patterns[0][1].render(item["fields"]))
        plans.append((path, item, part))
    if not plans:
        raise ValueError("No data remains for the output layout")
    check_outputs([path for path, *_ in plans], mode)
    for path, metadata, part in plans:
        with staged_output(path, mode) as staged:
            codec.write_file(part, staged, description, metadata)
    return [str(path) for path, *_ in plans]


def _station_collection(tables):
    """Assemble bounded station columns; retain metadata at first appearance."""
    if not tables:
        raise ValueError("No valid station data parsed")
    tables.sort(key=lambda item: item[0].fields["time"])
    stations = np.unique(
        np.concatenate([rows[:, 0].astype("i8") for _, _, rows in tables])
    )
    times = [record.fields["time"] for record, _, _ in tables]
    if len(set(times)) != len(times):
        raise ValueError("Duplicate time records")
    precipitation = np.full((len(tables), len(stations)), np.nan, dtype="f8")
    metadata = np.full((3, len(stations)), np.nan, dtype="f8")
    filled = np.zeros(len(stations), dtype=bool)
    for position, (_, _, rows) in enumerate(tables):
        indexes = np.searchsorted(stations, rows[:, 0].astype("i8"))
        precipitation[position, indexes] = rows[:, 4]
        first = ~filled[indexes]
        metadata[:, indexes[first]] = rows[first, 1:4].T
        filled[indexes] = True
    return xr.Dataset(
        {"precipitation": (("time", "station"), precipitation)},
        coords={
            "time": times,
            "station": stations,
            "lat": ("station", metadata[1]),
            "lon": ("station", metadata[0]),
            "altitude": ("station", metadata[2]),
        },
    )
