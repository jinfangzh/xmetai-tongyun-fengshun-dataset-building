"""Refresh existing normalized Zarr stores without repeating unit conversions.

Keep dataset paths and correction rules in an external JSON configuration.
Inspect or dry-run first, write a pilot, then refresh and validate the full store.
Optional auxiliary migration preserves the embedded mask unless requested.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from numcodecs import Blosc
import xarray as xr
import zarr


__all__ = ['LegacyZarrSource', 'RefreshProfile', 'build_refresh_plan', 'load_profiles', 'migrate_auxiliary_files', 'read_manifest', 'refresh_legacy_zarr', 'select_profiles', 'validate_refreshed_store']


@dataclass(frozen=True)
class RefreshProfile:
    key: str
    input: Path
    output_name: str
    cadence_hours: int
    radiation_seconds: int | None = None
    channel_coordinate: str = "auto"
    statistics: str = "auto"
    statistics_divide: dict[str, float] = field(default_factory=dict)
    statistics_multiply: dict[str, float] = field(default_factory=dict)
    channel_rename: dict[str, str] = field(default_factory=dict)
    mask: dict[str, Any] = field(default_factory=dict)
    expected_channels: int | None = None
    expected_grid: tuple[int, int] | None = None
    already_refreshed: str = "error"
    notes: str = ""

    @classmethod
    def from_dict(cls, key: str, value: dict[str, Any]) -> "RefreshProfile":
        grid = value.get("expected_grid")
        return cls(
            key=key,
            input=Path(value["input"]),
            output_name=str(value["output_name"]),
            cadence_hours=int(value["cadence_hours"]),
            radiation_seconds=(
                None
                if value.get("radiation_seconds") is None
                else int(value["radiation_seconds"])
            ),
            channel_coordinate=str(value.get("channel_coordinate", "auto")),
            statistics=str(value.get("statistics", "auto")),
            statistics_divide={
                str(name): float(divisor)
                for name, divisor in value.get("statistics_divide", {}).items()
            },
            statistics_multiply={
                str(pattern): float(factor)
                for pattern, factor in value.get("statistics_multiply", {}).items()
            },
            channel_rename={
                str(source): str(target)
                for source, target in value.get("channel_rename", {}).items()
            },
            mask=dict(value.get("mask", {})),
            expected_channels=(
                None
                if value.get("expected_channels") is None
                else int(value["expected_channels"])
            ),
            expected_grid=None if grid is None else (int(grid[0]), int(grid[1])),
            already_refreshed=str(value.get("already_refreshed", "error")),
            notes=str(value.get("notes", "")),
        )

    def serializable(self) -> dict[str, Any]:
        result = {
            "key": self.key,
            "input": str(self.input),
            "output_name": self.output_name,
            "cadence_hours": self.cadence_hours,
            "radiation_seconds": self.radiation_seconds,
            "channel_coordinate": self.channel_coordinate,
            "statistics": self.statistics,
            "statistics_divide": self.statistics_divide,
            "statistics_multiply": self.statistics_multiply,
            "channel_rename": self.channel_rename,
            "mask": self.mask,
            "expected_channels": self.expected_channels,
            "expected_grid": self.expected_grid,
            "already_refreshed": self.already_refreshed,
            "notes": self.notes,
        }
        return result


def load_profiles(path: Path) -> dict[str, RefreshProfile]:
    document = json.loads(path.read_text(encoding="utf-8"))
    raw = document.get("datasets")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a non-empty 'datasets' object")
    return {key: RefreshProfile.from_dict(key, value) for key, value in raw.items()}


def select_profiles(
    profiles: dict[str, RefreshProfile], requested: list[str]
) -> list[RefreshProfile]:
    if not requested or requested == ["all"]:
        return list(profiles.values())
    missing = [name for name in requested if name not in profiles]
    if missing:
        raise KeyError("unknown dataset profile(s): " + ", ".join(missing))
    return [profiles[name] for name in requested]


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


CF_TIME = re.compile(
    r"^(?P<unit>hours?|days?|seconds?)\s+since\s+(?P<origin>.+)$",
    re.IGNORECASE,
)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_cf_time(values: np.ndarray, attributes: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")
    units = str(attributes.get("units", ""))
    match = CF_TIME.match(units.strip())
    if not match:
        raise ValueError(f"unsupported numeric time units: {units!r}")
    origin_text = match.group("origin").strip().replace(" UTC", "")
    origin = np.datetime64(origin_text).astype("datetime64[ns]")
    unit = match.group("unit").lower()
    code = "h" if unit.startswith("hour") else "D" if unit.startswith("day") else "s"
    numeric = np.asarray(values, dtype="i8")
    return origin + numeric.astype(f"timedelta64[{code}]").astype("timedelta64[ns]")


def metadata_file(path: Path) -> dict[str, Any]:
    v3 = path / "zarr.json"
    if v3.is_file():
        return json.loads(v3.read_text(encoding="utf-8"))
    v2 = path / ".zattrs"
    if v2.is_file():
        return {"attributes": json.loads(v2.read_text(encoding="utf-8")), "zarr_format": 2}
    raise FileNotFoundError(f"no Zarr root metadata found in {path}")


def correct_statistics(
    channels: tuple[str, ...],
    mean_source: np.ndarray,
    std_source: np.ndarray,
    divide: dict[str, float],
    multiply: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    mean = mean_source.astype("f4", copy=True)
    std = std_source.astype("f4", copy=True)
    for channel, divisor in divide.items():
        if channel not in channels:
            raise ValueError(f"statistics correction channel missing: {channel}")
        if not np.isfinite(divisor) or divisor <= 0:
            raise ValueError(f"invalid statistics divisor for {channel}: {divisor}")
        if divisor == 1:
            continue
        index = channels.index(channel)
        mean[index] /= np.float32(divisor)
        std[index] /= np.float32(divisor)
    for pattern, factor in multiply.items():
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"invalid statistics multiplier for {pattern}: {factor}")
        indices = [index for index, name in enumerate(channels) if fnmatchcase(name, pattern)]
        if not indices:
            raise ValueError(f"statistics multiplier pattern matched no channels: {pattern}")
        if factor == 1:
            continue
        mean[indices] *= np.float32(factor)
        std[indices] *= np.float32(factor)
    return mean, std


@dataclass
class LegacyZarrSource:
    profile: RefreshProfile

    def __post_init__(self) -> None:
        self.path = self.profile.input
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)
        if self.profile.already_refreshed not in {"error", "force"}:
            raise ValueError("already_refreshed must be either 'error' or 'force'")
        previous_manifest = self.path / "refresh_manifest.json"
        if previous_manifest.is_file() and self.profile.already_refreshed != "force":
            raise ValueError(
                f"source already contains a refresh manifest: {previous_manifest}; "
                "set already_refreshed='force' only after verifying the source"
            )
        self.data = zarr.open_array(str(self.path / "data"), mode="r")
        coordinate = self.profile.channel_coordinate
        if coordinate == "auto":
            coordinate = "channel" if (self.path / "channel").is_dir() else "level"
        if coordinate not in {"channel", "level"}:
            raise ValueError(f"invalid channel coordinate {coordinate!r}")
        self.channel_coordinate = coordinate
        self.channel_array = zarr.open_array(str(self.path / coordinate), mode="r")
        self.lat_array = zarr.open_array(str(self.path / "lat"), mode="r")
        self.lon_array = zarr.open_array(str(self.path / "lon"), mode="r")
        self.time_array = zarr.open_array(str(self.path / "time"), mode="r")

        self.source_channels = tuple(_text(value) for value in self.channel_array[:].tolist())
        self.channels, self.channel_indices = canonicalize_channels(
            self.source_channels, self.profile.channel_rename
        )
        self.lat_source = np.asarray(self.lat_array[:], dtype="f8")
        self.lon = np.asarray(self.lon_array[:], dtype="f8")
        self.times = decode_cf_time(
            np.asarray(self.time_array[:]), dict(self.time_array.attrs)
        )
        mean_source = self._read_statistic("mean")
        std_source = self._read_statistic("std")
        order = np.asarray(self.channel_indices, dtype="i8")
        self.mean_source = mean_source[order]
        self.std_source = std_source[order]
        self.mean, self.std = self._correct_statistics()
        self.flip_latitude = bool(self.lat_source[0] < self.lat_source[-1])
        self.lat = self.lat_source[::-1].copy() if self.flip_latitude else self.lat_source.copy()
        self.mask = self._read_mask()
        if self.flip_latitude:
            self.mask = self.mask[::-1].copy()
        self._validate()

    def _read_statistic(self, statistic: str) -> np.ndarray:
        selected = self.profile.statistics
        candidates = []
        if selected == "auto":
            candidates = ["npy", "nc", "zarr", "root"]
        else:
            candidates = [selected]
        for backend in candidates:
            if backend == "npy" and (self.path / f"{statistic}.npy").is_file():
                return np.asarray(
                    np.load(self.path / f"{statistic}.npy"), dtype="f4"
                ).reshape(-1)
            if backend == "nc" and (self.path / f"{statistic}.nc").is_file():
                dataset = xr.open_dataset(self.path / f"{statistic}.nc")
                try:
                    variable = statistic if statistic in dataset.data_vars else next(
                        iter(dataset.data_vars)
                    )
                    values = np.asarray(dataset[variable].values, dtype="f4").reshape(-1)
                    if "channel" in dataset.coords:
                        labels = [_text(value) for value in dataset.channel.values.tolist()]
                        if len(labels) == len(values) and set(labels) == set(self.source_channels):
                            mapping = dict(zip(labels, values, strict=True))
                            values = np.asarray([mapping[name] for name in self.source_channels], dtype="f4")
                    return values
                finally:
                    dataset.close()
            if backend == "zarr" and (self.path / statistic).is_dir():
                array = zarr.open_array(str(self.path / statistic), mode="r")
                return np.asarray(array[:], dtype="f4").reshape(-1)
            if backend == "root":
                attributes = metadata_file(self.path).get("attributes", {})
                if statistic in attributes:
                    values = np.asarray(attributes[statistic], dtype="f4").reshape(-1)
                    channel_count = len(self.source_channels)
                    if values.size % channel_count != 0:
                        raise ValueError(
                            f"{statistic} root attribute length {values.size} is not "
                            f"a multiple of {channel_count} channels"
                        )
                    blocks = values.reshape(-1, channel_count)
                    if blocks.shape[0] > 1 and not np.allclose(
                        blocks, blocks[:1], rtol=0.0, atol=0.0
                    ):
                        raise ValueError(
                            f"{statistic} root attribute contains inconsistent repeated blocks"
                        )
                    return blocks[0]
        raise FileNotFoundError(f"cannot locate {statistic} for {self.path}")

    def _correct_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        return correct_statistics(
            self.channels,
            self.mean_source,
            self.std_source,
            self.profile.statistics_divide,
            self.profile.statistics_multiply,
        )
    def _read_mask(self) -> np.ndarray:
        spec = self.profile.mask
        kind = str(spec.get("kind", ""))
        relative = Path(str(spec.get("path", "")))
        path = self.path / relative
        if kind == "npy":
            values = np.load(path)
        elif kind == "netcdf":
            dataset = xr.open_dataset(path)
            try:
                variable = str(spec["variable"])
                values = np.asarray(dataset[variable].values)
            finally:
                dataset.close()
        elif kind == "const_channel":
            dataset = xr.open_dataset(path)
            try:
                variable = str(spec["variable"])
                channel = str(spec.get("channel", "lsm"))
                field = dataset[variable]
                if "channel" not in field.dims:
                    raise ValueError(f"{path}: {variable} has no channel dimension")
                values = np.asarray(field.sel(channel=channel).values)
            finally:
                dataset.close()
        else:
            raise ValueError(f"unsupported mask kind {kind!r} for {self.profile.key}")
        values = np.asarray(values).squeeze()
        if values.shape != tuple(self.data.shape[-2:]):
            raise ValueError(
                f"mask shape {values.shape} differs from data grid {self.data.shape[-2:]}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"mask contains non-finite values: {path}")
        if values.dtype == np.bool_ or np.isin(values, (0, 1)).all():
            self.mask_binarized = False
            return values.astype("u1")
        threshold = float(spec.get("threshold", 0.5))
        if not np.isfinite(threshold):
            raise ValueError(f"mask threshold must be finite, got {threshold}")
        self.mask_binarized = True
        return (values > threshold).astype("u1")

    def _validate(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(f"data must be 4-D, got {self.data.shape}")
        expected = (len(self.times), len(self.channels), len(self.lat), len(self.lon))
        if tuple(self.data.shape) != expected:
            raise ValueError(f"data shape {self.data.shape} != coordinates {expected}")
        if self.profile.expected_channels is not None and len(self.channels) != self.profile.expected_channels:
            raise ValueError("unexpected channel count")
        if self.profile.expected_grid is not None and tuple(self.data.shape[-2:]) != self.profile.expected_grid:
            raise ValueError("unexpected grid shape")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channel labels are not unique")
        if self.mean.shape != (len(self.channels),) or self.std.shape != (len(self.channels),):
            raise ValueError("mean/std are not aligned with channels")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("mean/std contain invalid values")
        if len(self.times) > 1:
            differences = np.diff(self.times)
            expected_step = np.timedelta64(self.profile.cadence_hours, "h")
            if not np.all(differences == expected_step):
                bad = np.flatnonzero(differences != expected_step)[:10].tolist()
                raise ValueError(f"time cadence mismatch at indices {bad}")
        if not np.all(np.diff(self.lon) > 0):
            raise ValueError("longitude must be strictly increasing")
        if not np.all(np.diff(self.lat) < 0):
            raise ValueError("target latitude must be strictly decreasing")
        if not np.isin(self.mask, (0, 1)).all():
            raise ValueError("mask is not binary")

    def read_data(self, index: int) -> np.ndarray:
        values = np.asarray(self.data[index], dtype="f2")
        identity = tuple(range(len(self.channel_indices)))
        if self.channel_indices != identity:
            values = values[np.asarray(self.channel_indices, dtype="i8")]
        if self.flip_latitude:
            values = values[:, ::-1, :]
        return np.ascontiguousarray(values)

    def fingerprint(self) -> str:
        payload = {
            "profile": self.profile.serializable(),
            "shape": tuple(int(v) for v in self.data.shape),
            "chunks": tuple(int(v) for v in self.data.chunks),
            "source_channels": self.source_channels,
            "channels": self.channels,
            "time_first": str(self.times[0]),
            "time_last": str(self.times[-1]),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def describe(self) -> dict[str, Any]:
        radiation = {}
        for name in ("ssr", "ssrd", "fdir", "ttr"):
            if name in self.channels:
                index = self.channels.index(name)
                radiation[name] = {
                    "source_mean": float(self.mean_source[index]),
                    "source_std": float(self.std_source[index]),
                    "target_mean": float(self.mean[index]),
                    "target_std": float(self.std[index]),
                }
        return {
            "key": self.profile.key,
            "input": str(self.path),
            "data_shape": [int(v) for v in self.data.shape],
            "data_chunks": [int(v) for v in self.data.chunks],
            "data_dtype": str(self.data.dtype),
            "channel_coordinate": self.channel_coordinate,
            "channel_count": len(self.channels),
            "source_channels": list(self.source_channels),
            "channels": list(self.channels),
            "channel_source_indices": list(self.channel_indices),
            "grid": [len(self.lat), len(self.lon)],
            "source_latitude_order": "ascending" if self.flip_latitude else "descending",
            "target_latitude_order": "descending",
            "time_count": len(self.times),
            "time_start": str(self.times[0]),
            "time_end": str(self.times[-1]),
            "cadence_hours": self.profile.cadence_hours,
            "radiation_seconds": self.profile.radiation_seconds,
            "radiation_statistics": radiation,
            "mask_land_fraction": float(self.mask.mean()),
            "fingerprint": self.fingerprint(),
        }


SCHEMA_VERSION = "2.0"
CONTENT_VERSION = "v3"
MASK_CHANNELS = ("land_mask", "sea_mask")
DATA_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["time", "channel", "lat", "lon"],
    "long_name": "meteorological fields",
    "data_representation": "normalized",
    "normalization_method": "channel_dependent",
    "mean_variable": "mean",
    "std_variable": "std",
    "scale_factor_location": "channel.channel_info[*].scale_factor",
    "add_offset_location": "channel.channel_info[*].add_offset",
    "inverse_transform": "preprocessed_value = data * scale_factor + add_offset",
}
LAT_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["lat"],
    "standard_name": "latitude",
    "long_name": "latitude",
    "units": "degrees_north",
    "axis": "Y",
}
LON_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["lon"],
    "standard_name": "longitude",
    "long_name": "longitude",
    "units": "degrees_east",
    "axis": "X",
}
TIME_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["time"],
    "standard_name": "time",
    "long_name": "time",
    "axis": "T",
    "timezone": "UTC",
}
MEAN_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["channel"],
    "alignment": "index-aligned with /channel",
    "long_name": "Channel normalization mean",
}
STD_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["channel"],
    "alignment": "index-aligned with /channel",
    "long_name": "Channel normalization standard deviation",
}
MASK_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["mask_channel", "lat", "lon"],
    "long_name": "Binary spatial masks",
    "mask_type": "binary",
    "mask_channel_variable": "mask_channel",
    "source_variable": "lsm",
    "derivation": "land_mask = lsm > 0.5; sea_mask = 1 - land_mask",
}
MASK_CHANNEL_ATTRIBUTES = {
    "_ARRAY_DIMENSIONS": ["mask_channel"],
    "long_name": "Mask channel name",
    "description": "Labels aligned with mask[mask_channel, :, :]",
}


def codec() -> Blosc:
    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def channel_info(
    channels: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    radiation_seconds: int | None,
) -> dict[str, dict[str, Any]]:
    return build_channel_info(channels, mean, std, radiation_seconds)


def create_refreshed_store(
    path: Path,
    *,
    dataset_id: str,
    data_revision: str,
    times: np.ndarray,
    channels: Sequence[str],
    lat: np.ndarray,
    lon: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    land_mask: np.ndarray,
    radiation_seconds: int | None,
) -> zarr.Group:
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.attrs.update(
        {
            "dataset_id": dataset_id,
            "schema_version": SCHEMA_VERSION,
            "content_version": CONTENT_VERSION,
            "data_revision": data_revision,
            "geospatial_lat_range": [float(np.min(lat)), float(np.max(lat))],
            "geospatial_lon_range": [float(np.min(lon)), float(np.max(lon))],
        }
    )
    channel_count = len(channels)
    ny, nx = len(lat), len(lon)
    group.create_array(
        "data",
        shape=(len(times), channel_count, ny, nx),
        chunks=(1, channel_count, ny, nx),
        dtype="f2",
        fill_value=np.nan,
        compressor=codec(),
        chunk_key_encoding={"name": "v2", "configuration": {"separator": "."}},
        attributes=DATA_ATTRIBUTES,
    )
    group.create_array(
        "channel",
        data=np.asarray(channels, dtype=str),
        chunks=(channel_count,),
        fill_value="",
        attributes={
            "_ARRAY_DIMENSIONS": ["channel"],
            "long_name": "Weather variable channel code",
            "channel_info_schema_version": "1.0",
            "channel_info": channel_info(channels, mean, std, radiation_seconds),
        },
    )
    group.create_array(
        "lat",
        data=np.asarray(lat, dtype="f4"),
        chunks=(ny,),
        fill_value=np.nan,
        attributes=LAT_ATTRIBUTES,
    )
    group.create_array(
        "lon",
        data=np.asarray(lon, dtype="f4"),
        chunks=(nx,),
        fill_value=np.nan,
        attributes=LON_ATTRIBUTES,
    )
    times_ns = np.asarray(times).astype("datetime64[ns]")
    group.create_array(
        "time",
        data=times_ns,
        chunks=times_ns.shape,
        fill_value=np.datetime64("NaT", "ns"),
        attributes=TIME_ATTRIBUTES,
    )
    group.create_array(
        "mean",
        data=np.asarray(mean, dtype="f4"),
        chunks=(channel_count,),
        compressor=codec(),
        fill_value=np.nan,
        attributes=MEAN_ATTRIBUTES,
    )
    group.create_array(
        "std",
        data=np.asarray(std, dtype="f4"),
        chunks=(channel_count,),
        compressor=codec(),
        fill_value=np.nan,
        attributes=STD_ATTRIBUTES,
    )
    group.create_array(
        "mask_channel",
        data=np.asarray(MASK_CHANNELS, dtype=str),
        chunks=(2,),
        fill_value="",
        attributes=MASK_CHANNEL_ATTRIBUTES,
    )
    masks = np.stack((land_mask.astype("u1"), np.uint8(1) - land_mask.astype("u1")))
    group.create_array(
        "mask",
        data=masks,
        chunks=(1, ny, nx),
        compressor=codec(),
        fill_value=np.uint8(0),
        attributes=MASK_ATTRIBUTES,
    )
    return group


EXPECTED_CHILDREN = {
    "data",
    "channel",
    "lat",
    "lon",
    "time",
    "mean",
    "std",
    "mask",
    "mask_channel",
}


def default_sample_indices(size: int) -> list[int]:
    if size <= 3:
        return list(range(size))
    return sorted({0, size // 2, size - 1})


def _strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values.tolist()]


def validate_refreshed_store(
    path: Path,
    source: LegacyZarrSource,
    source_indices: list[int],
    *,
    full_data_check: bool = False,
) -> dict[str, Any]:
    group = zarr.open_group(str(path), mode="r", use_consolidated=False)
    if set(group.keys()) != EXPECTED_CHILDREN:
        raise ValueError(f"unexpected root nodes: {sorted(group.keys())}")
    attrs = dict(group.attrs)
    if attrs.get("schema_version") != SCHEMA_VERSION or attrs.get("content_version") != CONTENT_VERSION:
        raise ValueError("root schema/content version mismatch")
    expected_shape = (
        len(source_indices),
        len(source.channels),
        len(source.lat),
        len(source.lon),
    )
    if tuple(group["data"].shape) != expected_shape:
        raise ValueError(f"target data shape {group['data'].shape} != {expected_shape}")
    if group["data"].dtype != np.dtype("f2"):
        raise ValueError("target data dtype is not float16")
    if tuple(group["data"].chunks) != (1, *expected_shape[1:]):
        raise ValueError("target data chunks do not follow the v2 contract")
    if _strings(group["channel"][:]) != list(source.channels):
        raise ValueError("channel values/order changed")
    np.testing.assert_array_equal(group["lat"][:], source.lat.astype("f4"))
    np.testing.assert_array_equal(group["lon"][:], source.lon.astype("f4"))
    np.testing.assert_array_equal(
        np.asarray(group["time"][:]).astype("datetime64[ns]"),
        source.times[source_indices].astype("datetime64[ns]"),
    )
    np.testing.assert_array_equal(group["mean"][:], source.mean.astype("f4"))
    np.testing.assert_array_equal(group["std"][:], source.std.astype("f4"))
    if _strings(group["mask_channel"][:]) != list(MASK_CHANNELS):
        raise ValueError("mask channel registry mismatch")
    masks = np.asarray(group["mask"][:], dtype="u1")
    if masks.shape != (2, len(source.lat), len(source.lon)):
        raise ValueError("mask shape mismatch")
    if not np.isin(masks, (0, 1)).all() or not np.array_equal(
        masks[0] + masks[1], np.ones_like(masks[0])
    ):
        raise ValueError("masks are not binary complements")
    np.testing.assert_array_equal(masks[0], source.mask)

    output_indices: Iterable[int]
    output_indices = range(len(source_indices)) if full_data_check else default_sample_indices(len(source_indices))
    checked = []
    radiation_report: dict[str, Any] = {}
    for output_index in output_indices:
        source_index = source_indices[output_index]
        expected = source.read_data(source_index)
        actual = np.asarray(group["data"][output_index], dtype="f2")
        if not np.array_equal(actual, expected, equal_nan=True):
            difference = np.nanmax(np.abs(actual.astype("f4") - expected.astype("f4")))
            raise ValueError(
                f"normalized data mismatch output={output_index} source={source_index}; max_abs={difference}"
            )
        checked.append({"output_index": output_index, "source_index": source_index})

    for name in ("ssr", "ssrd", "fdir", "ttr"):
        if name not in source.channels:
            continue
        ci = source.channels.index(name)
        samples = []
        for output_index in default_sample_indices(len(source_indices)):
            normalized = np.asarray(group["data"][output_index, ci], dtype="f4")
            physical = normalized * source.std[ci] + source.mean[ci]
            finite = physical[np.isfinite(physical)]
            if finite.size:
                samples.append(
                    {
                        "output_index": output_index,
                        "min": float(np.min(finite)),
                        "mean": float(np.mean(finite)),
                        "max": float(np.max(finite)),
                    }
                )
        radiation_report[name] = samples

    return {
        "path": str(path),
        "shape": list(expected_shape),
        "checked_data_chunks": checked,
        "full_data_check": full_data_check,
        "radiation_physical_samples": radiation_report,
        "status": "ok",
    }


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = path / "refresh_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    return json.loads(manifest.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def selection_indices(size: int, start: int, count: int | None) -> list[int]:
    if start < 0 or start >= size:
        raise ValueError(f"start index {start} outside [0, {size})")
    stop = size if count is None else min(size, start + count)
    if count is not None and count <= 0:
        raise ValueError("count must be positive")
    return list(range(start, stop))


def output_path(
    profile: RefreshProfile,
    output_root: Path,
    *,
    pilot: bool,
    start: int,
    count: int | None,
) -> Path:
    name = profile.output_name
    if pilot:
        assert count is not None
        name += f".pilot.{start}_{start + count - 1}"
    return output_root / name


def build_refresh_plan(
    source: LegacyZarrSource,
    destination: Path,
    indices: list[int],
) -> dict[str, Any]:
    """Return a read-only summary of transformations that would be applied."""
    identity = tuple(range(len(source.channel_indices)))
    return {
        "input": str(source.path),
        "output": str(destination),
        "source_indices": {
            "start": indices[0],
            "end": indices[-1],
            "count": len(indices),
        },
        "actions": {
            "channel_rename": bool(source.profile.channel_rename),
            "channel_reorder": source.channel_indices != identity,
            "latitude_flip": source.flip_latitude,
            "longitude_reorder": False,
            "statistics_divide": source.profile.statistics_divide,
            "statistics_multiply": source.profile.statistics_multiply,
            "data_dtype_conversion": str(source.data.dtype) != "float16",
            "mask_binarization": source.mask_binarized,
        },
    }


def refresh_legacy_zarr(
    profile: RefreshProfile,
    output_root: Path,
    *,
    start: int = 0,
    count: int | None = None,
    pilot: bool = False,
    workers: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    full_data_check: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    source = LegacyZarrSource(profile)
    indices = selection_indices(len(source.times), start, count)
    final = output_path(profile, output_root, pilot=pilot, start=start, count=len(indices))
    if final.resolve() == source.path.resolve():
        raise ValueError("output path must not equal the read-only source path")
    staging = final.with_name("." + final.name + ".staging")
    state_path = final.with_name("." + final.name + ".refresh_state.json")
    fingerprint_payload = {
        "source_fingerprint": source.fingerprint(),
        "source_indices": [indices[0], indices[-1]],
        "count": len(indices),
        "output_name": final.name,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    completed = 0
    if staging.exists():
        if not resume:
            raise FileExistsError(f"staging exists and resume is disabled: {staging}")
        if not state_path.is_file():
            raise RuntimeError(f"staging has no state file: {staging}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError("staging fingerprint differs from current source/config")
        completed = int(state.get("completed", 0))
        group = zarr.open_group(str(staging), mode="r+", use_consolidated=False)
    else:
        revision = datetime.now(timezone.utc).strftime("%Y%m%d")
        group = create_refreshed_store(
            staging,
            dataset_id=final.name.removesuffix(".zarr"),
            data_revision=revision,
            times=source.times[indices],
            channels=source.channels,
            lat=source.lat,
            lon=source.lon,
            mean=source.mean,
            std=source.std,
            land_mask=source.mask,
            radiation_seconds=profile.radiation_seconds,
        )
        atomic_json(
            state_path,
            {
                "fingerprint": fingerprint,
                "completed": 0,
                "total": len(indices),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    target = group["data"]
    session_start = completed
    started = time.monotonic()

    def load(item: tuple[int, int]) -> tuple[int, np.ndarray]:
        output_index, source_index = item
        return output_index, source.read_data(source_index)

    while completed < len(indices):
        block_stop = min(len(indices), completed + workers)
        work = [(i, indices[i]) for i in range(completed, block_stop)]
        if workers == 1:
            loaded = [load(work[0])]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                loaded = list(executor.map(load, work))
        for output_index, values in sorted(loaded):
            target[output_index] = values
        completed = block_stop
        atomic_json(
            state_path,
            {
                "fingerprint": fingerprint,
                "completed": completed,
                "total": len(indices),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        elapsed = time.monotonic() - started
        processed = completed - session_start
        rate = processed / elapsed if elapsed else 0.0
        eta_seconds = (len(indices) - completed) / rate if rate else float("inf")
        print(
            f"[{profile.key}] {completed}/{len(indices)} "
            f"({completed / len(indices):.2%}) rate={rate:.3f} steps/s "
            f"eta={eta_seconds / 3600:.2f}h",
            flush=True,
        )

    del target, group
    zarr.consolidate_metadata(str(staging))
    validation = validate_refreshed_store(
        staging, source, indices, full_data_check=full_data_check
    )
    manifest = {
        "tool": "data-format-refresh",
        "profile": profile.serializable(),
        "source_fingerprint": source.fingerprint(),
        "source_indices": {"start": indices[0], "end": indices[-1], "count": len(indices)},
        "source_time": {"start": str(source.times[indices[0]]), "end": str(source.times[indices[-1]])},
        "transformations": {
            "normalized_data": "unchanged",
            "latitude_reversed": source.flip_latitude,
            "statistics_divide": profile.statistics_divide,
            "statistics_multiply": profile.statistics_multiply,
            "channel_rename": profile.channel_rename,
            "channel_order": "template_intersection",
            "channel_source_indices": list(source.channel_indices),
            "codec": "blosc-zstd-clevel5-bitshuffle",
        },
        "validation": validation,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(staging / "refresh_manifest.json", manifest)
    state_path.unlink(missing_ok=True)
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {final}")
        backup = final.with_name(
            final.name + ".backup." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        os.replace(final, backup)
    os.replace(staging, final)
    return final, manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npy(path: Path, values: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"destination exists (use --overwrite): {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_netcdf(path: Path, dataset: xr.Dataset, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"destination exists (use --overwrite): {path}")
    temporary = path.with_name(path.stem + ".tmp.nc")
    temporary.unlink(missing_ok=True)
    dataset.to_netcdf(temporary)
    os.replace(temporary, path)


def read_zarr_coordinate(store: Path, name: str) -> np.ndarray:
    return np.asarray(zarr.open_array(str(store / name), mode="r")[:])


def index_for_coordinate(
    source: np.ndarray, target: np.ndarray, *, longitude: bool = False
) -> np.ndarray:
    source = np.asarray(source, dtype="f8").reshape(-1)
    target = np.asarray(target, dtype="f8").reshape(-1)
    if source.size != target.size:
        raise ValueError(f"coordinate size differs: {source.size} != {target.size}")
    source_key = np.mod(source, 360.0) if longitude else source
    target_key = np.mod(target, 360.0) if longitude else target
    if np.allclose(source_key, target_key, rtol=0, atol=1e-6):
        return np.arange(source.size, dtype="i8")
    if np.allclose(source_key[::-1], target_key, rtol=0, atol=1e-6):
        return np.arange(source.size - 1, -1, -1, dtype="i8")
    lookup = {round(float(value), 7): index for index, value in enumerate(source_key)}
    try:
        indices = np.asarray(
            [lookup[round(float(value), 7)] for value in target_key], dtype="i8"
        )
    except KeyError as exc:
        raise ValueError(f"target coordinate value missing from source: {exc}") from exc
    if not np.allclose(source_key[indices], target_key, rtol=0, atol=1e-6):
        raise ValueError("coordinate mapping failed")
    return indices


def const_data_variable(dataset: xr.Dataset) -> str:
    matches = [
        name
        for name, variable in dataset.data_vars.items()
        if {"channel", "lat", "lon"}.issubset(variable.dims)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one const field, found {matches}")
    return matches[0]


def build_const(source_file: Path, target_store: Path) -> tuple[xr.Dataset, dict[str, Any]]:
    target_lat = read_zarr_coordinate(target_store, "lat").astype("f8")
    target_lon = read_zarr_coordinate(target_store, "lon").astype("f8")
    source = xr.open_dataset(source_file, decode_cf=False)
    try:
        variable_name = const_data_variable(source)
        field = source[variable_name].transpose("channel", "lat", "lon")
        source_channels = [str(value) for value in source["channel"].values.tolist()]
        lat_index = index_for_coordinate(source["lat"].values, target_lat)
        lon_index = index_for_coordinate(
            source["lon"].values, target_lon, longitude=True
        )
        values = np.asarray(field.values, dtype="f4")
        values = values[:, lat_index, :][:, :, lon_index]
        result = xr.Dataset(
            data_vars={variable_name: (("channel", "lat", "lon"), values)},
            coords={
                "channel": np.asarray(source_channels, dtype=str),
                "lat": target_lat,
                "lon": target_lon,
            },
            attrs=dict(source.attrs),
        )
        result[variable_name].attrs.update(dict(field.attrs))
        result["channel"].attrs.update(dict(source["channel"].attrs))
        result["lat"].attrs.update(dict(source["lat"].attrs))
        result["lon"].attrs.update(dict(source["lon"].attrs))
        details = {
            "source_variable": variable_name,
            "source_channels": source_channels,
            "target_channels": source_channels,
            "latitude_reordered": not np.array_equal(
                lat_index, np.arange(lat_index.size)
            ),
            "longitude_reordered": not np.array_equal(
                lon_index, np.arange(lon_index.size)
            ),
            "generated_channels": [],
            "channel_policy": "preserved from legacy const.nc",
            "shape": list(values.shape),
        }
        return result, details
    finally:
        source.close()

def build_weight(source_file: Path, target_store: Path) -> tuple[np.ndarray, dict[str, Any]]:
    target_lat = read_zarr_coordinate(target_store, "lat").astype("f8")
    source = xr.open_dataset(source_file, decode_cf=False)
    try:
        candidates = [v for v in source.data_vars.values() if v.dims == ("lat",)]
        if len(candidates) != 1:
            raise ValueError("expected one weight variable with dimensions (lat,)")
        field = candidates[0]
        lat_index = index_for_coordinate(source["lat"].values, target_lat)
        values = np.asarray(field.values, dtype="f4")[lat_index].reshape(1, -1, 1)
        return values, {
            "source_variable": field.name,
            "latitude_reordered": not np.array_equal(
                lat_index, np.arange(lat_index.size)
            ),
            "shape": list(values.shape),
        }
    finally:
        source.close()


def build_weight_npy(
    source_file: Path, source_root: Path, target_store: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    source_lat, _ = source_spatial_coordinates(source_root)
    target_lat = read_zarr_coordinate(target_store, "lat").astype("f8")
    raw = np.asarray(np.load(source_file, allow_pickle=False), dtype="f4")
    if raw.shape == (source_lat.size,):
        vector = raw
    elif raw.shape == (1, source_lat.size, 1):
        vector = raw[0, :, 0]
    else:
        raise ValueError(
            f"unsupported weight.npy shape {raw.shape}; expected "
            f"{(source_lat.size,)} or {(1, source_lat.size, 1)}"
        )
    lat_index = index_for_coordinate(source_lat, target_lat)
    values = vector[lat_index].reshape(1, -1, 1).astype("f4")
    return values, {
        "source_format": "npy",
        "latitude_reordered": not np.array_equal(
            lat_index, np.arange(lat_index.size)
        ),
        "shape": list(values.shape),
    }


def source_spatial_coordinates(source_root: Path) -> tuple[np.ndarray, np.ndarray]:
    const_file = source_root / "const.nc"
    if const_file.is_file():
        dataset = xr.open_dataset(const_file, decode_cf=False)
        try:
            if "lat" in dataset.coords and "lon" in dataset.coords:
                return (
                    np.asarray(dataset["lat"].values),
                    np.asarray(dataset["lon"].values),
                )
        finally:
            dataset.close()
    return (
        np.asarray(zarr.open_array(str(source_root / "lat"), mode="r")[:]),
        np.asarray(zarr.open_array(str(source_root / "lon"), mode="r")[:]),
    )


def build_mask(source_file: Path, source_root: Path, target_store: Path) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(np.load(source_file, allow_pickle=False)).squeeze()
    source_lat, source_lon = source_spatial_coordinates(source_root)
    target_lat = read_zarr_coordinate(target_store, "lat")
    target_lon = read_zarr_coordinate(target_store, "lon")
    if values.shape != (source_lat.size, source_lon.size):
        raise ValueError(
            f"mask shape {values.shape} != source grid {(source_lat.size, source_lon.size)}"
        )
    lat_index = index_for_coordinate(source_lat, target_lat)
    lon_index = index_for_coordinate(source_lon, target_lon, longitude=True)
    values = values[lat_index, :][:, lon_index].astype(bool)
    return values, {
        "latitude_reordered": not np.array_equal(
            lat_index, np.arange(lat_index.size)
        ),
        "longitude_reordered": not np.array_equal(
            lon_index, np.arange(lon_index.size)
        ),
        "shape": list(values.shape),
    }


def build_mask_netcdf(
    source_file: Path,
    target_store: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align one two-dimensional NetCDF mask to the refreshed grid."""

    target_lat = read_zarr_coordinate(target_store, "lat")
    target_lon = read_zarr_coordinate(target_store, "lon")
    dataset = xr.open_dataset(source_file, decode_cf=False)
    try:
        candidates = [
            variable
            for variable in dataset.data_vars.values()
            if {"lat", "lon"}.issubset(variable.dims)
        ]
        if len(candidates) != 1:
            names = [str(variable.name) for variable in candidates]
            raise ValueError(
                f"expected one mask variable with lat/lon dimensions, found {names}"
            )
        field = candidates[0].transpose("lat", "lon")
        lat_index = index_for_coordinate(dataset["lat"].values, target_lat)
        lon_index = index_for_coordinate(
            dataset["lon"].values,
            target_lon,
            longitude=True,
        )
        values = np.asarray(field.values)[lat_index, :][:, lon_index].astype(bool)
        return values, {
            "source_format": "netcdf",
            "source_variable": str(field.name),
            "latitude_reordered": not np.array_equal(
                lat_index, np.arange(lat_index.size)
            ),
            "longitude_reordered": not np.array_equal(
                lon_index, np.arange(lon_index.size)
            ),
            "shape": list(values.shape),
        }
    finally:
        dataset.close()

def build_land_mask_from_const(
    source_file: Path, target_store: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    target_lat = read_zarr_coordinate(target_store, "lat")
    target_lon = read_zarr_coordinate(target_store, "lon")
    source = xr.open_dataset(source_file, decode_cf=False)
    try:
        variable_name = const_data_variable(source)
        field = source[variable_name].transpose("channel", "lat", "lon")
        channels = [str(value) for value in source["channel"].values.tolist()]
        if "lsm" not in channels:
            raise ValueError(f"const.nc has no lsm channel: {source_file}")
        lat_index = index_for_coordinate(source["lat"].values, target_lat)
        lon_index = index_for_coordinate(
            source["lon"].values, target_lon, longitude=True
        )
        values = np.asarray(field.values[channels.index("lsm")])
        land = values[lat_index, :][:, lon_index] > 0.5
        return land, {
            "source": str(source_file),
            "derivation": "land_mask = const.nc/lsm > 0.5; sea_mask = 1 - land_mask",
            "latitude_reordered": not np.array_equal(
                lat_index, np.arange(lat_index.size)
            ),
            "longitude_reordered": not np.array_equal(
                lon_index, np.arange(lon_index.size)
            ),
            "shape": [2, int(land.shape[0]), int(land.shape[1])],
        }
    finally:
        source.close()


def write_embedded_masks(target_store: Path, land: np.ndarray) -> None:
    array = zarr.open_array(str(target_store / "mask"), mode="r+")
    expected = (2, *land.shape)
    if tuple(array.shape) != expected:
        raise ValueError(f"embedded mask shape {array.shape} != {expected}")
    replacement = np.stack((land, ~land)).astype("u1")
    original = np.asarray(array[:]).copy()
    try:
        array[:] = replacement
        if not np.array_equal(np.asarray(array[:]), replacement):
            raise IOError("embedded mask verification failed after write")
    except Exception:
        array[:] = original
        raise


def choose_output(output_root: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = output_root / name
        if candidate.is_dir():
            return candidate
    return None


def verify_target_store(path: Path) -> None:
    if path.name.startswith(".") or path.name.endswith(".staging"):
        raise ValueError(f"refusing to modify staging directory: {path}")
    for name in (".zgroup", "data", "lat", "lon", "channel"):
        if not (path / name).exists():
            raise ValueError(f"not a completed refreshed store; missing {name}: {path}")


def migrate_auxiliary_files(
    key: str,
    spec: dict[str, Any],
    output_root: Path,
    *,
    dry_run: bool,
    overwrite: bool,
    strict: bool,
) -> dict[str, Any]:
    source = Path(spec["source"])
    target = choose_output(output_root, spec["outputs"])
    if target is None:
        message = f"no completed output found for {key}: {spec['outputs']}"
        if strict:
            raise FileNotFoundError(message)
        return {"dataset": key, "status": "skipped", "reason": message}
    verify_target_store(target)
    report: dict[str, Any] = {
        "dataset": key,
        "source": str(source),
        "target": str(target),
        "dry_run": dry_run,
        "artifacts": {},
    }

    const_source = source / "const.nc"
    if const_source.is_file():
        dataset, details = build_const(const_source, target)
        destination = target / "const.nc"
        if not dry_run:
            atomic_netcdf(destination, dataset, overwrite)
            details["sha256"] = sha256(destination)
        dataset.close()
        details.update({"source": str(const_source), "destination": str(destination)})
        report["artifacts"]["const.nc"] = details
    else:
        report["artifacts"]["const.nc"] = {"status": "source_missing_skipped"}

    weight_nc_source = source / "weight.nc"
    weight_npy_source = source / "weight.npy"
    if weight_nc_source.is_file():
        weight_source = weight_nc_source
        values, details = build_weight(weight_source, target)
        details["source_format"] = "netcdf"
        destination = target / "weight.npy"
        if not dry_run:
            atomic_npy(destination, values, overwrite)
            details["sha256"] = sha256(destination)
        details.update({"source": str(weight_source), "destination": str(destination)})
        report["artifacts"]["weight.npy"] = details
    elif weight_npy_source.is_file():
        weight_source = weight_npy_source
        values, details = build_weight_npy(weight_source, source, target)
        destination = target / "weight.npy"
        if not dry_run:
            atomic_npy(destination, values, overwrite)
            details["sha256"] = sha256(destination)
        details.update({"source": str(weight_source), "destination": str(destination)})
        report["artifacts"]["weight.npy"] = details
    else:
        report["artifacts"]["weight.npy"] = {"status": "source_missing_skipped"}

    mask_npy_source = source / "mask.npy"
    mask_nc_source = source / "mask.nc"
    mask_source: Path | None = None
    if mask_npy_source.is_file():
        mask_source = mask_npy_source
        values, details = build_mask(mask_source, source, target)
        details["source_format"] = "numpy"
    elif mask_nc_source.is_file():
        mask_source = mask_nc_source
        values, details = build_mask_netcdf(mask_source, target)
    else:
        values = None
        details = {"status": "source_missing_skipped"}

    if mask_source is not None and values is not None:
        destination = target / "mask.npy"
        if not dry_run:
            atomic_npy(destination, values, overwrite)
            details["sha256"] = sha256(destination)
        details["semantics"] = (
            "preserved from the legacy mask source; not assumed to be land_mask"
        )
        details.update({"source": str(mask_source), "destination": str(destination)})
        report["artifacts"]["mask.npy"] = details

        refresh_embedded_mask = bool(spec.get("refresh_embedded_mask", False))
        if (
            refresh_embedded_mask
            and const_source.is_file()
            and (target / "mask").is_dir()
        ):
            land, embedded_details = build_land_mask_from_const(
                const_source, target
            )
            if not dry_run:
                write_embedded_masks(target, land)
                embedded_details["verified"] = True
            report["artifacts"]["embedded_mask"] = embedded_details
        else:
            report["artifacts"]["embedded_mask"] = {
                "status": "preserved",
                "reason": (
                    "embedded mask replacement requires explicit "
                    "refresh_embedded_mask=true"
                ),
            }
    else:
        report["artifacts"]["mask.npy"] = details

    report["status"] = "planned" if dry_run else "ok"
    if not dry_run:
        manifest = target / "auxiliary_refresh_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    **report,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return report


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    result = argparse.ArgumentParser(
        description="Refresh legacy meteorological Zarr stores"
    )
    result.add_argument("--config", type=Path, required=True)
    sub = result.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="read-only source inventory")
    inspect.add_argument("--dataset", action="append", default=[])
    inspect.add_argument("--report", type=Path)

    for name, help_text in (
        ("pilot", "write a small contiguous test slice and validate it"),
        ("convert", "refresh a complete dataset"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--dataset", required=True)
        command.add_argument("--output-root", type=Path, required=True)
        if name == "pilot":
            command.add_argument("--start-index", type=int, default=0)
            command.add_argument("--count", type=int, default=4)
        command.add_argument(
            "--dry-run",
            action="store_true",
            help="show the selected data and transformations without writing",
        )
        command.add_argument("--workers", type=int, default=1)
        command.add_argument("--no-resume", action="store_true")
        command.add_argument("--overwrite", action="store_true")
        command.add_argument("--full-data-check", action="store_true")

    validate = sub.add_parser("validate", help="reopen and compare an existing output")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--zarr", type=Path, required=True)
    validate.add_argument("--full-data-check", action="store_true")

    auxiliary = sub.add_parser(
        "auxiliary", help="migrate const, weight and mask sidecar files"
    )
    auxiliary.add_argument("--dataset", action="append", default=[])
    auxiliary.add_argument("--output-root", type=Path, required=True)
    auxiliary.add_argument(
        "--target-name",
        help="completed output directory name; defaults to the profile output_name",
    )
    auxiliary.add_argument("--dry-run", action="store_true")
    auxiliary.add_argument(
        "--refresh-embedded-mask",
        action="store_true",
        help="replace the Zarr mask from const.nc/lsm after semantic verification",
    )
    auxiliary.add_argument("--overwrite", action="store_true")
    auxiliary.add_argument("--strict", action="store_true")
    return result.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = get_args(argv)
    profiles = load_profiles(args.config)

    if args.command == "inspect":
        selected = select_profiles(profiles, args.dataset or ["all"])
        report = {
            profile.key: LegacyZarrSource(profile).describe() for profile in selected
        }
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        return

    if args.command == "auxiliary":
        selected = select_profiles(profiles, args.dataset or ["all"])
        reports = [
            migrate_auxiliary_files(
                profile.key,
                {
                    "source": str(profile.input),
                    "outputs": [args.target_name or profile.output_name],
                    "refresh_embedded_mask": args.refresh_embedded_mask,
                },
                args.output_root,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                strict=args.strict,
            )
            for profile in selected
        ]
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    profile = profiles[args.dataset]
    if args.command in {"pilot", "convert"}:
        source = LegacyZarrSource(profile)
        count = args.count if args.command == "pilot" else None
        start = args.start_index if args.command == "pilot" else 0
        indices = selection_indices(len(source.times), start, count)
        destination = output_path(
            profile,
            args.output_root,
            pilot=args.command == "pilot",
            start=start,
            count=len(indices),
        )
        if args.dry_run:
            print(
                json.dumps(
                    build_refresh_plan(source, destination, indices),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        path, manifest = refresh_legacy_zarr(
            profile,
            args.output_root,
            start=start,
            count=count,
            pilot=args.command == "pilot",
            workers=args.workers,
            resume=not args.no_resume,
            overwrite=args.overwrite,
            full_data_check=args.full_data_check,
        )
        print(
            json.dumps(
                {"output": str(path), "validation": manifest["validation"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    manifest = read_manifest(args.zarr)
    source_range = manifest["source_indices"]
    indices = list(range(int(source_range["start"]), int(source_range["end"]) + 1))
    report = validate_refreshed_store(
        args.zarr,
        LegacyZarrSource(profile),
        indices,
        full_data_check=args.full_data_check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
