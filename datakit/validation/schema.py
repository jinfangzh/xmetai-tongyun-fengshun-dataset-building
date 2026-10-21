"""Schema and coordinate validation for in-memory xarray objects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def validate_weather_array(
    obj: xr.Dataset | xr.DataArray,
    *,
    require_channel: bool = True,
) -> None:
    """Raise ``ValueError`` when a weather array violates core conventions."""

    dims = set(obj.dims)
    missing = {"time", "lat", "lon"} - dims
    if missing:
        raise ValueError(f"Missing required dimensions: {sorted(missing)}")
    if require_channel and not ({"level", "channel"} & dims):
        raise ValueError("Expected a 'level' or 'channel' dimension")
    for coordinate in ("time", "lat", "lon"):
        values = np.asarray(obj[coordinate].values)
        if values.size == 0:
            raise ValueError(f"Coordinate '{coordinate}' is empty")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Coordinate '{coordinate}' contains duplicates")
    times = pd.DatetimeIndex(pd.to_datetime(obj.time.values))
    if not times.is_monotonic_increasing:
        raise ValueError("Time coordinate is not monotonically increasing")
