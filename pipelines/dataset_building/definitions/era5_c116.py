"""ERA5 c116 product standard and production source mappings."""

from __future__ import annotations

from .standard_dataset import (
    DatasetProduct,
    MaskDefinition,
    DatasetVariant,
    GRID_P25,
    STANDARD_LAND_SEA_MASK,
)
from ._standard_products import (
    BASE_LEVELS,
    PRESSURE_VARIABLES,
    CONST_OROGRAPHY,
    WEIGHT_NPY,
    MASK_NPY,
    define_channels,
)

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

MASK: MaskDefinition = STANDARD_LAND_SEA_MASK
VARIANTS: tuple[DatasetVariant, ...] = (
    DatasetVariant("p25.h6", GRID_P25, 6, (CONST_OROGRAPHY, WEIGHT_NPY, MASK_NPY)),
)
PRODUCT: DatasetProduct = DatasetProduct(
    product_id="era5.c116",
    version="1.0",
    channels=define_channels(CHANNELS, radiation_seconds=21600),
    variants=VARIANTS,
    mask=MASK,
)

# Existing production callers retain the same names and values.
DYNAMIC_CHANNELS = CHANNELS
CHANNEL_COUNT = len(CHANNELS)
TARGET_LAT, TARGET_LON = GRID_P25.coordinates()

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
