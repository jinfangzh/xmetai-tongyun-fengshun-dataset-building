"""ERA5 c116 production channels and raw-source mappings, not acceptance rules."""

from __future__ import annotations

import numpy as np

from datakit.schema.dimensions import LEVELS_13

BASE_LEVELS = tuple(reversed(LEVELS_13))
PRESSURE_VARIABLES = ("z", "t", "u", "v", "q")

W_LEVELS = BASE_LEVELS + (10,)
SURFACE = (
    "d2m",
    "t2m",
    "msl",
    "sp",
    "skt",
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "sst",
    "tcw",
    "tp",
    "sd",
)
DERIVED = ("ws10m", "ws100m")
CLOUD_RADIATION = ("lcc", "mcc", "hcc", "tcc", "ssr", "ssrd", "fdir", "ttr")
SOIL = ("swvl1", "swvl2", "stl1", "stl2")
WAVE = ("wmb", "mwd", "cdww", "mwp", "swh")
STATIC = ("lsm", "slor", "sdor", "z_sfc")

CHANNELS: tuple[str, ...] = (
    tuple(f"{variable}{level}" for variable in PRESSURE_VARIABLES for level in BASE_LEVELS)
    + SURFACE
    + tuple(f"{variable}10" for variable in PRESSURE_VARIABLES)
    + tuple(f"w{level}" for level in W_LEVELS)
    + DERIVED
    + CLOUD_RADIATION
    + SOIL
    + WAVE
)

DYNAMIC_CHANNELS = CHANNELS
CHANNEL_COUNT = len(CHANNELS)
TARGET_LAT = np.linspace(90.0, -90.0, 721, dtype=np.float32)
TARGET_LON = np.arange(0.0, 360.0, 0.25, dtype=np.float32)

SOURCE_VARIABLES = {
    "u10m": ("u10m", "u10"),
    "v10m": ("v10m", "v10"),
    "u100m": ("u100m", "u100"),
    "v100m": ("v100m", "v100"),
    "swvl1": ("swvl1", "vsw1"),
    "swvl2": ("swvl2", "vsw2"),
    "stl1": ("stl1", "sot1"),
    "stl2": ("stl2", "sot2"),
    "z_sfc": ("z_sfc", "z"),
}

SOURCE_DIRECTORIES = {
    "swvl1": "vsw1",
    "swvl2": "vsw2",
    "stl1": "sot1",
    "stl2": "sot2",
}

GROUPS = {
    **{name: "pl" for name in PRESSURE_VARIABLES + ("w",)},
    **{name: "sfc" for name in SURFACE + DERIVED},
    **{name: "cldrad" for name in CLOUD_RADIATION},
    **{name: "soil" for name in SOIL},
    **{name: "wave" for name in WAVE},
    **{name: "static" for name in STATIC},
}

if len(CHANNELS) != 116 or len(set(CHANNELS)) != 116:
    raise RuntimeError("canonical ERA5 channel definition must contain 116 unique labels")
