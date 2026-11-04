"""Channel-coordinate attributes for weather Zarr products."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np

from datakit.schema.variables import ACCUMULATED_VARIABLES

from .raw_truth import DERIVED, DYNAMIC_CHANNELS, SOIL, WAVE

CHANNEL_INFO_SCHEMA_VERSION = "1.0"
DEFAULT_RADIATION_SECONDS = 21600.0
DEFAULT_RADIATION_VARIABLES = frozenset({"ssr", "ssrd", "fdir", "ttr"})
DEFAULT_WIND_COMPONENTS: Mapping[str, tuple[str, str]] = {
    "ws10m": ("u10m", "v10m"),
    "ws100m": ("u100m", "v100m"),
}
ERA5_VARIABLE_METADATA: Mapping[str, tuple[str, str]] = {
    "z": ("m2 s-2", "Geopotential"),
    "t": ("K", "Temperature"),
    "u": ("m s-1", "Eastward wind"),
    "v": ("m s-1", "Northward wind"),
    "q": ("g kg-1", "Specific humidity"),
    "w": ("Pa s-1", "Vertical velocity"),
    "d2m": ("K", "2 metre dewpoint temperature"),
    "t2m": ("K", "2 metre temperature"),
    "q2m": ("g kg-1", "2 metre specific humidity"),
    "t2m_min": ("K", "Daily minimum 2 metre temperature"),
    "t2m_max": ("K", "Daily maximum 2 metre temperature"),
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
}
ORAS5_VARIABLE_METADATA: Mapping[str, tuple[str, str]] = {
    "sit": ("m", "Sea ice thickness"),
    "sic": ("1", "Sea ice concentration"),
    "ssh": ("m", "Sea surface height"),
    "mlt": ("m", "Mixed layer thickness"),
    "so": ("1e-3", "Sea water salinity"),
    "to": ("degrees_C", "Sea water potential temperature"),
    "uo": ("m s-1", "Eastward sea water velocity"),
    "vo": ("m s-1", "Northward sea water velocity"),
}
VARIABLE_METADATA: Mapping[str, tuple[str, str]] = {
    **ERA5_VARIABLE_METADATA,
    **ORAS5_VARIABLE_METADATA,
}
ORAS5_OCEAN_DEPTH_VARIABLES = frozenset({"so", "to", "uo", "vo"})
SOURCE_UNITS = {
    **{name: units for name, (units, _) in VARIABLE_METADATA.items()},
    "q": "kg kg-1",
    "q2m": "kg kg-1",
    "ssr": "J m-2",
    "ssrd": "J m-2",
    "fdir": "J m-2",
    "ttr": "J m-2",
    "tp": "m",
}
SOURCE_NAMES: dict[str, str | list[str]] = {name: name for name in VARIABLE_METADATA}
SOURCE_NAMES.update(
    {
        "sit": "sithick_oras",
        "sic": "siconc_oras",
        "ssh": "zos_oras",
        "mlt": "mlotst_oras",
        "so": "so_oras",
        "to": "thetao_oras",
        "uo": "uo_oras",
        "vo": "vo_oras",
        "ws10m": ["u10m", "v10m"],
        "ws100m": ["u100m", "v100m"],
        "swvl1": "vsw1",
        "swvl2": "vsw2",
        "stl1": "sot1",
        "stl2": "sot2",
    }
)


def split_channel(
    channel: str,
    *,
    pressure_variables: Collection[str] = ("z", "t", "u", "v", "q", "w"),
) -> tuple[str, int | None]:
    """Split a channel into base variable and pressure level when possible."""

    text = str(channel)
    # Known surface/soil labels containing digits are complete variable names.
    # The generic suffix fallback remains for legacy callers; strict product
    # declarations are maintained separately in the dataset-building workflow.
    if text in VARIABLE_METADATA:
        return text, None
    for variable in pressure_variables:
        if text.startswith(variable):
            suffix = text[len(variable) :]
            if suffix.isdigit():
                return variable, int(suffix)
    match = re.fullmatch(r"([A-Za-z_]+?)(\d+)", text)
    if match is not None:
        return match.group(1), int(match.group(2))
    return text, None


def preprocessing_metadata(
    channel: str,
    variable: str,
    *,
    radiation_seconds: float = DEFAULT_RADIATION_SECONDS,
    accumulation_hours: float | None = None,
    humidity_prefix: str = "q",
    radiation_variables: Collection[str] = DEFAULT_RADIATION_VARIABLES,
    wind_components: Mapping[str, tuple[str, str]] = DEFAULT_WIND_COMPONENTS,
    log1p_variables: Collection[str] = ("tp",),
    tp_zscore: bool = False,
) -> list[dict[str, Any]]:
    """Build preprocessing steps metadata for one channel."""

    zscore = {"operation": "zscore", "mean": "/mean", "std": "/std"}
    if channel in wind_components:
        u_name, v_name = wind_components[channel]
        return [{"operation": "hypot", "inputs": [u_name, v_name]}, zscore]
    if channel in set(log1p_variables):
        steps: list[dict[str, Any]] = [
            {
                "operation": "multiply",
                "factor": 1000.0,
                "input_units": "m",
                "output_units": "mm",
            },
            {"operation": "clip_min", "minimum": 0.0},
            {"operation": "log1p"},
        ]
        if tp_zscore:
            steps.append(zscore)
        return steps
    if channel in set(radiation_variables) or variable in set(ACCUMULATED_VARIABLES):
        seconds = (
            float(radiation_seconds)
            if accumulation_hours is None
            else 3600.0 * float(accumulation_hours)
        )
        return [
            {
                "operation": "divide",
                "divisor": seconds,
                "accumulation_window_seconds": int(round(seconds)),
                "input_units": "J m-2",
                "output_units": "W m-2",
            },
            zscore,
        ]
    if channel.startswith(humidity_prefix) or variable.startswith(humidity_prefix):
        return [
            {
                "operation": "multiply",
                "factor": 1000.0,
                "input_units": "kg kg-1",
                "output_units": "g kg-1",
            },
            zscore,
        ]
    return [zscore]


def channel_level_metadata(variable: str, level: int | None) -> dict[str, Any]:
    """Build vertical level metadata for one channel."""

    if level is not None:
        if variable in ORAS5_OCEAN_DEPTH_VARIABLES:
            level_m = level / 100.0
            return {
                "level_type": "depth_below_sea",
                "level": level_m,
                "level_units": "m",
                "level_code": level,
            }
        return {"level_type": "pressure", "level": level, "level_units": "hPa"}
    height = {
        "d2m": 2,
        "t2m": 2,
        "q2m": 2,
        "t2m_min": 2,
        "t2m_max": 2,
        "u10m": 10,
        "v10m": 10,
        "ws10m": 10,
        "u100m": 100,
        "v100m": 100,
        "ws100m": 100,
    }.get(variable)
    if height is not None:
        return {
            "level_type": "height_above_ground",
            "level": height,
            "level_units": "m",
        }
    if variable in SOIL or re.fullmatch(r"(?:swvl|stl)[1-4]", variable):
        return {
            "level_type": "soil_layer",
            "level": int(variable[-1]),
            "level_units": "layer_index",
        }
    return {"level_type": "surface"}


def build_channel_info(
    channels: Sequence[str] = DYNAMIC_CHANNELS,
    *,
    mean: Sequence[float] | np.ndarray,
    std: Sequence[float] | np.ndarray,
    radiation_seconds: float = DEFAULT_RADIATION_SECONDS,
    accumulation_hours: float | None = None,
    variable_metadata: Mapping[str, tuple[str, str]] = VARIABLE_METADATA,
) -> dict[str, dict[str, Any]]:
    """Build ``channel_info`` aligned positionally with the channel coordinate."""

    mean_values = np.asarray(mean, dtype="f4").reshape(-1)
    std_values = np.asarray(std, dtype="f4").reshape(-1)
    if len(mean_values) != len(channels) or len(std_values) != len(channels):
        raise ValueError("mean/std must align positionally with channels")
    if (
        not np.isfinite(mean_values).all()
        or not np.isfinite(std_values).all()
        or not np.all(std_values > 0)
    ):
        raise ValueError("channel scale_factor/add_offset values are invalid")

    result: dict[str, dict[str, Any]] = {}
    for index, raw_channel in enumerate(channels):
        channel = str(raw_channel)
        variable, level = split_channel(channel)
        units, long_name = variable_metadata.get(variable, ("unknown", variable))
        level_meta = channel_level_metadata(variable, level)
        if level is not None:
            if level_meta.get("level_type") == "pressure":
                long_name = f"{long_name} at {level} hPa"
            elif level_meta.get("level_type") == "depth_below_sea":
                long_name = f"{long_name} at {float(level_meta['level']):g} m depth"
            else:
                long_name = f"{long_name} at {level}"
        result[channel] = {
            "long_name": long_name,
            "source_name": SOURCE_NAMES.get(variable, variable),
            "source_units": SOURCE_UNITS.get(variable, "unknown"),
            "units": units,
            "scale_factor": float(std_values[index]),
            "add_offset": float(mean_values[index]),
            "scale_offset_formula": (
                "preprocessed_value = normalized_value * scale_factor + add_offset"
            ),
            **level_meta,
            "variable_type": (
                "derived"
                if variable in DERIVED
                else (
                    "accumulated"
                    if variable in DEFAULT_RADIATION_VARIABLES or variable == "tp"
                    else "static"
                    if variable in WAVE and variable == "wmb"
                    else "instantaneous"
                )
            ),
            "preprocessing": preprocessing_metadata(
                channel,
                variable,
                radiation_seconds=radiation_seconds,
                accumulation_hours=accumulation_hours,
            ),
        }
    return result


def channel_attributes(
    channels: Sequence[str] = DYNAMIC_CHANNELS,
    *,
    mean: Sequence[float] | np.ndarray,
    std: Sequence[float] | np.ndarray,
    radiation_seconds: float = DEFAULT_RADIATION_SECONDS,
    accumulation_hours: float | None = None,
) -> dict[str, Any]:
    """Build attributes stored on the Zarr ``/channel`` coordinate array."""

    return {
        "long_name": "Weather variable channel code",
        "channel_info_schema_version": CHANNEL_INFO_SCHEMA_VERSION,
        "channel_info": build_channel_info(
            channels,
            mean=mean,
            std=std,
            radiation_seconds=radiation_seconds,
            accumulation_hours=accumulation_hours,
        ),
    }


__all__ = [
    "CHANNEL_INFO_SCHEMA_VERSION",
    "DEFAULT_RADIATION_SECONDS",
    "DEFAULT_RADIATION_VARIABLES",
    "DEFAULT_WIND_COMPONENTS",
    "ERA5_VARIABLE_METADATA",
    "ORAS5_VARIABLE_METADATA",
    "VARIABLE_METADATA",
    "build_channel_info",
    "channel_attributes",
    "channel_level_metadata",
    "preprocessing_metadata",
    "split_channel",
]
