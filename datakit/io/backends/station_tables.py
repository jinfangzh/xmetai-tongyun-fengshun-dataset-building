"""Single-file Diamond station and probability table parsing and serialization."""

import numpy as np
import xarray as xr


def read_file(path, description):
    """Return native station arrays and metadata needed to restore this file."""
    probability = "thresholds" in description
    file_format = description.get("format", {})
    columns = file_format.get(
        "columns", ["station", "lon", "lat", "altitude", "precipitation"]
    )
    skip = file_format.get("header_lines", 0 if probability else 2)
    with path.open(encoding=file_format.get("encoding", "gbk")) as stream:
        headers = [stream.readline() for _ in range(skip)]
        rows = np.loadtxt(stream, ndmin=2)
    minimum = len(description["thresholds"]) + 1 if probability else len(columns)
    if (
        rows.size == 0
        or rows.shape[1] < minimum
        or (probability and rows.shape[1] != minimum)
    ):
        raise ValueError(f"Invalid station table shape {rows.shape}: {path}")
    station = rows[:, 0]
    if (
        not np.all(np.isfinite(station))
        or not np.all(station == station.astype("i8"))
        or len(set(station)) != len(station)
    ):
        raise ValueError(f"Station IDs must be unique integers: {path}")
    if probability:
        dataset = xr.Dataset(
            {"probability": (("station", "threshold"), rows[:, 1:])},
            coords={"station": station.astype("i8"), "threshold": description["thresholds"]},
        )
    else:
        names = columns[1:] + [f"column_{i}" for i in range(len(columns), rows.shape[1])]
        dataset = xr.Dataset(
            {name: ("station", rows[:, i + 1]) for i, name in enumerate(names)},
            coords={"station": station.astype("i8")},
        )
    return dataset, {
        "headers": headers,
        "stations": station.astype("i8"),
        "columns": list(dataset.data_vars),
    }


def prepare_file(dataset, description, metadata):
    """Restore native row/column order and check format constraints before writing."""
    missing = set(metadata["columns"]) - dataset.variables.keys()
    if missing:
        raise ValueError(f"Required text columns were removed: {sorted(missing)}")
    if "station" not in dataset.dims:
        dataset = dataset.expand_dims("station")
    stations = metadata["stations"][np.isin(metadata["stations"], dataset.station.values)]
    dataset = dataset.sel(station=stations)
    if "thresholds" in description:
        if "threshold" not in dataset.dims:
            dataset = dataset.expand_dims("threshold")
        if dataset.sizes["threshold"] != len(description["thresholds"]):
            raise ValueError("Probability output requires every declared threshold")
        dataset = dataset.sel(threshold=description["thresholds"])
        if set(dataset.probability.dims) != {"station", "threshold"}:
            raise ValueError("Probability output requires station and threshold dimensions")
    elif any(dataset[name].dims != ("station",) for name in metadata["columns"]):
        raise ValueError("Text columns must have one value per station")
    return dataset


def write_file(dataset, path, description, metadata):
    """Serialize one prepared table to the supplied staging path."""
    values = (
        dataset.probability.transpose("station", "threshold").values
        if "thresholds" in description
        else np.column_stack([dataset[name].values for name in metadata["columns"]])
    )
    rows = np.column_stack([dataset.station.values, values])
    with path.open("x", encoding=description.get("format", {}).get("encoding", "gbk")) as stream:
        stream.writelines(metadata["headers"])
        np.savetxt(stream, rows, fmt="%.17g")
