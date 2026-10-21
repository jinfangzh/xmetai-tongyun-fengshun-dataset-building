from __future__ import annotations

import xarray as xr

from datakit.schema import SHORT_NAME_MAPPING


XarrayObject = xr.Dataset | xr.DataArray


def standardize_variable_names(obj: XarrayObject) -> XarrayObject:
    """Rename known Dataset variables or a DataArray name to canonical names."""

    if isinstance(obj, xr.DataArray):
        canonical = SHORT_NAME_MAPPING.get(obj.name, obj.name)
        return obj.rename(canonical) if canonical != obj.name else obj
    mapping = {
        old: new
        for old, new in SHORT_NAME_MAPPING.items()
        if old in obj.data_vars and new not in obj.data_vars
    }
    return obj.rename_vars(mapping) if mapping else obj


def level_to_channel(obj: XarrayObject, short_name: str) -> xr.Dataset:
    """Flatten a pressure ``level`` dimension into canonical channel labels."""

    if isinstance(obj, xr.DataArray):
        array = obj
    else:
        if short_name not in obj.data_vars:
            raise KeyError(f"Variable '{short_name}' not found")
        array = obj[short_name]

    if "channel" in array.dims:
        return array.rename("data").to_dataset()
    has_level = "level" in array.dims
    if not has_level:
        axis = 1 if "time" in array.dims else 0
        array = array.expand_dims({"level": [1]}, axis=axis)

    levels = array["level"].values.tolist()
    if not has_level:
        channels = [SHORT_NAME_MAPPING.get(short_name, short_name)]
    else:
        channels = [f"{short_name}{(int(level))}" for level in levels]
    array = (
        array.rename({"level": "channel"})
        .assign_coords(channel=channels)
        .rename("data")
    )
    array.attrs = {}
    return array.to_dataset()


def channel_to_level(obj: XarrayObject, short_name: str) -> XarrayObject:
    """Restore numeric levels from channels such as ``t500`` and ``t850``."""

    if "channel" not in obj.dims:
        axis = 1 if "time" in obj.dims else 0
        return obj.expand_dims({"level": [1]}, axis=axis)
    labels = [str(value) for value in obj["channel"].values.tolist()]
    prefix = SHORT_NAME_MAPPING.get(short_name, short_name)
    try:
        levels = [
            1 if label == prefix else int(label.removeprefix(prefix))
            for label in labels
        ]
    except ValueError as exc:
        raise ValueError(
            f"Channels cannot be parsed as '{prefix}<level>': {labels}"
        ) from exc
    return obj.rename({"channel": "level"}).assign_coords(level=levels)
