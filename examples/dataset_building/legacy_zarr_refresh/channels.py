"""Channel ordering and metadata used by legacy Zarr refresh."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
BASE_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
W_LEVELS = BASE_LEVELS + (10,)
PRESSURE_VARIABLES = ("z", "t", "u", "v", "q")
SURFACE = (
    "d2m", "t2m", "msl", "sp", "skt", "u10m", "v10m", "u100m", "v100m",
    "sst", "tcw", "tp", "sd",
)
DERIVED = ("ws10m", "ws100m")
CLOUD_RADIATION = ("lcc", "mcc", "hcc", "tcc", "ssr", "ssrd", "fdir", "ttr")
SOIL = ("swvl1", "swvl2", "stl1", "stl2")
WAVE = ("wmb", "mwd", "cdww", "mwp", "swh")
RADIATION = frozenset(("ssr", "ssrd", "fdir", "ttr"))
OCEAN_DEPTHS_CM = (
    50, 266, 514, 809, 1177, 1652, 2275, 3087, 4118, 5385, 6902,
    8692, 10803, 13307, 16316, 19978, 22114, 27135, 33386, 41179, 50863,
)
OCEAN_DEPTH_VARIABLES = ("os", "ot", "ou", "ov")
OCEAN_SURFACE = ("sit", "ssh", "sic", "mlt")
OCEAN_CHANNELS = (
    tuple(f"{variable}{depth}" for variable in OCEAN_DEPTH_VARIABLES for depth in OCEAN_DEPTHS_CM)
    + OCEAN_SURFACE
)

# Exact 116-channel order from the approved ERA5 v3 template.
TEMPLATE_CHANNELS = (
    tuple(f"{variable}{level}" for variable in PRESSURE_VARIABLES for level in BASE_LEVELS)
    + SURFACE
    + tuple(f"{variable}10" for variable in PRESSURE_VARIABLES)
    + tuple(f"w{level}" for level in W_LEVELS)
    + DERIVED + CLOUD_RADIATION + SOIL + WAVE
)

# Verified legacy-only channels that are absent from the 116-channel template.
# They are inserted after the closest template channel without changing the
# relative order of any template channels.
SUPPLEMENTS_AFTER: dict[str, tuple[str, ...]] = {
    "t2m": ("t2m_avg", "t2m_min", "t2m_max", "q2m"),
    "tcw": ("tcwv",),
    "swvl2": ("swvl3", "swvl4"),
    "stl2": ("stl3", "stl4"),
}


def _complete_channel_order() -> tuple[str, ...]:
    result: list[str] = []
    for name in TEMPLATE_CHANNELS:
        result.append(name)
        result.extend(SUPPLEMENTS_AFTER.get(name, ()))
    return tuple(result)


CHANNEL_ORDER = _complete_channel_order() + OCEAN_CHANNELS
CHANNEL_RANK = {name: index for index, name in enumerate(CHANNEL_ORDER)}

if len(TEMPLATE_CHANNELS) != 116 or len(set(TEMPLATE_CHANNELS)) != 116:
    raise RuntimeError("template channel definition must contain 116 unique labels")
if len(CHANNEL_RANK) != len(CHANNEL_ORDER):
    raise RuntimeError("complete channel order contains duplicate labels")


def canonicalize_channels(
    source_channels: Sequence[str], rename: Mapping[str, str]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    renamed = tuple(rename.get(name, name) for name in source_channels)
    if len(set(renamed)) != len(renamed):
        raise ValueError(f"channel rename creates duplicate labels: {renamed}")
    unknown = [name for name in renamed if name not in CHANNEL_RANK]
    if unknown:
        raise ValueError(
            "channels are absent from the template/supplement registry: "
            + ", ".join(unknown)
        )
    indices = tuple(sorted(range(len(renamed)), key=lambda index: CHANNEL_RANK[renamed[index]]))
    return tuple(renamed[index] for index in indices), indices


VARIABLE_METADATA = {
    "z": ("m2 s-2", "Geopotential"),
    "t": ("K", "Temperature"),
    "u": ("m s-1", "Eastward wind"),
    "v": ("m s-1", "Northward wind"),
    "q": ("g kg-1", "Specific humidity"),
    "w": ("Pa s-1", "Vertical velocity"),
    "d2m": ("K", "2 metre dewpoint temperature"),
    "t2m": ("K", "2 metre temperature"),
    "t2m_avg": ("K", "Daily average 2 metre temperature"),
    "t2m_min": ("K", "Daily minimum 2 metre temperature"),
    "t2m_max": ("K", "Daily maximum 2 metre temperature"),
    "q2m": ("g kg-1", "2 metre specific humidity"),
    "msl": ("Pa", "Mean sea level pressure"),
    "sp": ("Pa", "Surface pressure"),
    "skt": ("K", "Skin temperature"),
    "u10m": ("m s-1", "10 metre eastward wind"),
    "v10m": ("m s-1", "10 metre northward wind"),
    "u100m": ("m s-1", "100 metre eastward wind"),
    "v100m": ("m s-1", "100 metre northward wind"),
    "sst": ("K", "Sea surface temperature"),
    "tcw": ("kg m-2", "Total column water"),
    "tcwv": ("kg m-2", "Total column water vapour"),
    "tp": ("1", "Log-transformed total precipitation"),
    "sd": ("m", "Snow depth water equivalent"),
    "ws10m": ("m s-1", "10 metre wind speed"),
    "ws100m": ("m s-1", "100 metre wind speed"),
    "lcc": ("1", "Low cloud cover"),
    "mcc": ("1", "Medium cloud cover"),
    "hcc": ("1", "High cloud cover"),
    "tcc": ("1", "Total cloud cover"),
    "ssr": ("W m-2", "Surface net short-wave radiation"),
    "ssrd": ("W m-2", "Surface short-wave radiation downwards"),
    "fdir": ("W m-2", "Surface direct short-wave radiation"),
    "ttr": ("W m-2", "Top net long-wave radiation"),
    "swvl1": ("m3 m-3", "Volumetric soil water layer 1"),
    "swvl2": ("m3 m-3", "Volumetric soil water layer 2"),
    "swvl3": ("m3 m-3", "Volumetric soil water layer 3"),
    "swvl4": ("m3 m-3", "Volumetric soil water layer 4"),
    "stl1": ("K", "Soil temperature level 1"),
    "stl2": ("K", "Soil temperature level 2"),
    "stl3": ("K", "Soil temperature level 3"),
    "stl4": ("K", "Soil temperature level 4"),
    "wmb": ("m", "Model bathymetry"),
    "mwd": ("degree", "Mean wave direction"),
    "cdww": ("1", "Coefficient of drag with waves"),
    "mwp": ("s", "Mean wave period"),
    "swh": ("m", "Significant height of combined wind waves and swell"),
    "os": ("1", "Ocean salinity"),
    "ot": ("degree_Celsius", "Ocean potential temperature"),
    "ou": ("m s-1", "Eastward ocean current"),
    "ov": ("m s-1", "Northward ocean current"),
    "sit": ("m", "Sea-ice thickness"),
    "ssh": ("m", "Sea-surface height"),
    "sic": ("1", "Sea-ice concentration"),
    "mlt": ("m", "Ocean mixed-layer thickness"),
}
SOURCE_UNITS = {name: units for name, (units, _) in VARIABLE_METADATA.items()}
SOURCE_UNITS.update({
    "q": "kg kg-1", "q2m": "kg kg-1",
    "ssr": "J m-2", "ssrd": "J m-2", "fdir": "J m-2", "ttr": "J m-2",
    "tp": "m",
})
SOURCE_NAMES: dict[str, str | list[str]] = {name: name for name in VARIABLE_METADATA}
SOURCE_NAMES.update({
    "ws10m": ["u10m", "v10m"],
    "ws100m": ["u100m", "v100m"],
    "swvl1": "vsw1", "swvl2": "vsw2",
    "stl1": "sot1", "stl2": "sot2",
})


def split_channel(channel: str) -> tuple[str, int | None]:
    for variable in OCEAN_DEPTH_VARIABLES:
        suffix = channel.removeprefix(variable)
        if channel.startswith(variable) and suffix.isdigit():
            return variable, int(suffix)
    for variable in (*PRESSURE_VARIABLES, "w"):
        suffix = channel.removeprefix(variable)
        if channel.startswith(variable) and suffix.isdigit():
            return variable, int(suffix)
    return channel, None


def preprocessing_metadata(
    channel: str, variable: str, *, radiation_seconds: int | None
) -> list[dict[str, Any]]:
    zscore = {"operation": "zscore", "mean": "/mean", "std": "/std"}
    if variable in {"q", "q2m"}:
        return [{
            "operation": "multiply", "factor": 1000.0,
            "input_units": "kg kg-1", "output_units": "g kg-1",
        }, zscore]
    if variable in RADIATION:
        if radiation_seconds is None:
            raise ValueError(f"radiation_seconds is required for {channel}")
        return [{
            "operation": "divide", "divisor": float(radiation_seconds),
            "accumulation_window_seconds": int(radiation_seconds),
            "input_units": "J m-2", "output_units": "W m-2",
        }, zscore]
    if channel == "tp":
        return [
            {"operation": "multiply", "factor": 1000.0, "input_units": "m", "output_units": "mm"},
            {"operation": "clip_min", "minimum": 0.0},
            {"operation": "log1p"},
        ]
    if channel == "ws10m":
        return [{"operation": "hypot", "inputs": ["u10m", "v10m"]}, zscore]
    if channel == "ws100m":
        return [{"operation": "hypot", "inputs": ["u100m", "v100m"]}, zscore]
    return [zscore]


def channel_level_metadata(variable: str, level: int | None) -> dict[str, Any]:
    if variable in OCEAN_DEPTH_VARIABLES and level is not None:
        return {
            "level_type": "depth_below_sea_surface",
            "level": level / 100.0,
            "level_units": "m",
        }
    if level is not None:
        return {"level_type": "pressure", "level": level, "level_units": "hPa"}
    height = {
        "d2m": 2, "t2m": 2, "t2m_avg": 2, "t2m_min": 2, "t2m_max": 2,
        "q2m": 2, "u10m": 10, "v10m": 10, "ws10m": 10,
        "u100m": 100, "v100m": 100, "ws100m": 100,
    }.get(variable)
    if height is not None:
        return {"level_type": "height_above_ground", "level": height, "level_units": "m"}
    if variable.startswith("stl") or variable.startswith("swvl"):
        return {"level_type": "soil_layer", "level": int(variable[-1]), "level_units": "layer_index"}
    return {"level_type": "surface"}


def build_channel_info(
    channels: Sequence[str], mean: np.ndarray, std: np.ndarray, radiation_seconds: int | None
) -> dict[str, dict[str, Any]]:
    mean_values = np.asarray(mean, dtype="f4").reshape(-1)
    std_values = np.asarray(std, dtype="f4").reshape(-1)
    if len(mean_values) != len(channels) or len(std_values) != len(channels):
        raise ValueError("mean/std must align positionally with channels")
    result: dict[str, dict[str, Any]] = {}
    for index, channel in enumerate(channels):
        variable, level = split_channel(str(channel))
        if variable not in VARIABLE_METADATA:
            raise ValueError(f"missing channel metadata for {channel}")
        units, long_name = VARIABLE_METADATA[variable]
        if level is not None and variable in OCEAN_DEPTH_VARIABLES:
            long_name = f"{long_name} at {level / 100.0:g} m"
        elif level is not None:
            long_name = f"{long_name} at {level} hPa"
        result[str(channel)] = {
            "long_name": long_name,
            "source_name": SOURCE_NAMES[variable],
            "source_units": SOURCE_UNITS[variable],
            "units": units,
            "scale_factor": float(std_values[index]),
            "add_offset": float(mean_values[index]),
            "scale_offset_formula": "preprocessed_value = normalized_value * scale_factor + add_offset",
            **channel_level_metadata(variable, level),
            "variable_type": (
                "derived" if variable in DERIVED
                else "accumulated" if variable in RADIATION or variable == "tp"
                else "static" if variable == "wmb"
                else "instantaneous"
            ),
            "preprocessing": preprocessing_metadata(
                str(channel), variable, radiation_seconds=radiation_seconds
            ),
        }
    return result
