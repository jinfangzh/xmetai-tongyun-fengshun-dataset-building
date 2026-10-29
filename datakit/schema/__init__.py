"""Canonical weather-data dimensions and variable metadata."""

from .mapping import DatasetMapping, VarSpec, load_mapping
from .dimensions import (
    DIMENSION_ALIASES,
    LEVELS_6,
    LEVELS_13,
    LEVELS_20,
    LEVELS_30,
    LEVELS_37,
)
from .variables import (
    ACCUMULATED_VARIABLES,
    LAND_VARIABLES,
    OCEAN_VARIABLES,
    PRESSURE_VARIABLES,
    SHORT_NAME_MAPPING,
    STATIONS_NORM,
    UNIT_SCALE,
    WATER_VARIABLES,
    is_accumulated,
    is_land,
    is_ocean,
    is_pressure,
    is_water,
)

__all__ = [
    "DatasetMapping",
    "VarSpec",
    "load_mapping",
    "ACCUMULATED_VARIABLES",
    "DIMENSION_ALIASES",
    "LAND_VARIABLES",
    "LEVELS_6",
    "LEVELS_13",
    "LEVELS_20",
    "LEVELS_30",
    "LEVELS_37",
    "OCEAN_VARIABLES",
    "PRESSURE_VARIABLES",
    "SHORT_NAME_MAPPING",
    "STATIONS_NORM",
    "UNIT_SCALE",
    "WATER_VARIABLES",
    "is_accumulated",
    "is_land",
    "is_ocean",
    "is_pressure",
    "is_water",
]
