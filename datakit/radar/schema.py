"""Static grid conventions for composite weather-radar products."""

from __future__ import annotations

from dataclasses import dataclass
import math


RadarBBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class RadarGridSpec:
    """Explicit grid metadata for a headerless compressed radar array."""

    height: int
    width: int
    left_lon: float
    top_lat: float
    lon_step: float
    lat_step: float

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Radar grid height and width must be positive")
        if self.lon_step <= 0 or self.lat_step <= 0:
            raise ValueError("Radar grid steps must be positive")
        if not all(
            math.isfinite(value)
            for value in (self.left_lon, self.top_lat, self.lon_step, self.lat_step)
        ):
            raise ValueError("Radar grid coordinates and steps must be finite")
