"""Single-file BABJ track parsing and serialization."""

import numpy as np
import xarray as xr


def read_file(path, description):
    """Parse local record numbers and retain the original header and columns."""
    file_format = description["format"]
    columns = file_format["columns"]
    with path.open(encoding=file_format["encoding"]) as stream:
        headers = [stream.readline() for _ in range(file_format["header_lines"])]
        tokens = headers[1].split()
        if len(tokens) < 2:
            raise ValueError(f"Invalid BABJ header: {path}")
        rows = np.loadtxt(stream, ndmin=2)
    if rows.size == 0 or rows.shape[1] < len(columns):
        raise ValueError(f"Invalid BABJ numeric table: {path}")
    names = list(columns)
    names += [f"column_{i}" for i in range(len(columns), rows.shape[1])]
    dataset = xr.Dataset(
        {name: ("record", rows[:, i]) for i, name in enumerate(names)},
        coords={
            "record": np.arange(len(rows)),
            "storm": ("record", [tokens[1]] * len(rows)),
            "tcname": ("record", [tokens[0]] * len(rows)),
        },
    )
    return dataset, {"headers": headers, "columns": names}


def prepare_file(dataset, description, metadata):
    """Restore native row order and check columns and header identity."""
    missing = set(metadata["columns"]) - dataset.variables.keys()
    if missing:
        raise ValueError(f"Required BABJ columns were removed: {sorted(missing)}")
    if "record" not in dataset.dims:
        dataset = dataset.expand_dims("record")
    labels = metadata["records"][np.isin(metadata["records"], dataset.record.values)]
    dataset = dataset.sel(record=labels)
    if any(dataset[name].dims != ("record",) for name in metadata["columns"]):
        raise ValueError("BABJ columns must have one value per record")
    tokens = metadata["headers"][1].split()
    for name, expected in (("tcname", tokens[0]), ("storm", tokens[1])):
        if name in dataset.coords and not np.all(dataset[name].values == expected):
            raise ValueError(f"BABJ header coordinate {name!r} was changed")
    return dataset


def write_file(dataset, path, description, metadata):
    """Serialize one prepared track to the supplied staging path."""
    values = np.column_stack([dataset[name].values for name in metadata["columns"]])
    with path.open("x", encoding=description["format"]["encoding"]) as stream:
        stream.writelines(metadata["headers"])
        np.savetxt(stream, values, fmt="%.17g")
