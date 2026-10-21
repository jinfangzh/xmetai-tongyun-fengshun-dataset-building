"""Canonical variable names, groups and unit-conversion metadata."""

from types import MappingProxyType


SHORT_NAME_MAPPING = MappingProxyType(
    {
        "prmsl": "msl",
        "tp24hr": "tp",
        "u10": "u10m",
        "v10": "v10m",
        "ws10": "ws10m",
        "u100": "u100m",
        "v100": "v100m",
        "ws100": "ws100m",
    }
)

UNIT_SCALE = MappingProxyType(
    {
        "ciwc": 1000.0,
        "clwc": 1000.0,
        "crwc": 1000.0,
        "cswc": 1000.0,
        "q": 1000.0,
        "q2m": 1000.0,
        "tp": 1000.0,
        "ttr": 1 / 3600,
        "ssr": 1 / 3600,
        "ssrd": 1 / 3600,
        "fdir": 1 / 3600,
    }
)

STATIONS_NORM = MappingProxyType(
    {
        "tp": (0.0, 100.0),
        "ws": (0.0, 35.0),
        "gs": (0.0, 40.0),
        "cr": (0.0, 70.0),
        "ld": (0.0, 100.0),
    }
)

PRESSURE_VARIABLES = frozenset(
    {
        "omega",
        "h2o",
        "o3",
        "gh",
        "z",
        "s",
        "t",
        "u",
        "v",
        "q",
        "cc",
        "ciwc",
        "clwc",
        "crwc",
        "cswc",
    }
)
ACCUMULATED_VARIABLES = frozenset({"ssr", "ssrd", "fdir", "ttr", "tp"})
OCEAN_VARIABLES = frozenset(
    {"sst", "mdts", "mdww", "mpts", "mpww", "shts", "shww"}
)
LAND_VARIABLES = frozenset(
    {"stl1", "stl2", "stl3", "stl4", "swvl1", "swvl2", "swvl3", "swvl4"}
)
WATER_VARIABLES = frozenset({"ciwc", "clwc", "crwc", "cswc", "tp"})


def is_pressure(name: str) -> bool:
    return name in PRESSURE_VARIABLES


def is_accumulated(name: str) -> bool:
    return name in ACCUMULATED_VARIABLES


def is_ocean(name: str) -> bool:
    return name in OCEAN_VARIABLES


def is_land(name: str) -> bool:
    return name in LAND_VARIABLES


def is_water(name: str) -> bool:
    return name in WATER_VARIABLES
