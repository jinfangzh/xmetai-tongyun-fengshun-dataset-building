"""Canonical dimension names and standard pressure-level collections."""

from types import MappingProxyType


LEVELS_13 = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

DIMENSION_ALIASES = MappingProxyType(
    {
        "valid_time": "time",
        "valid-time": "time",
        "dayofyear": "doy",
        "lead_time": "step",
        "prediction_timedelta": "step",
        "pressure_level": "level",
        "isobaricInhPa": "level",
        "heightAboveGround": "level",
        "heightabovefround": "level",  # Legacy misspelling.
        "latitude": "lat",
        "Latitude": "lat",
        "nav_lat": "lat",
        "longitude": "lon",
        "Longitude": "lon",
        "nav_lon": "lon",
        "number": "member",
        "numrer": "member",  # Legacy misspelling.
        "plev": "level",
        "lev": "level",
    }
)
