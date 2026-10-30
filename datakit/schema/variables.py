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

ACCUMULATED_VARIABLES = frozenset({"ssr", "ssrd", "fdir", "ttr", "tp"})
WATER_VARIABLES = frozenset({"ciwc", "clwc", "crwc", "cswc", "tp"})
