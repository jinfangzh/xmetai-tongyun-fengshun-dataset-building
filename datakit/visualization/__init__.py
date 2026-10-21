"""Meteorological colormaps and lightweight xarray plotting helpers."""

from .animations import save_field_gif
from .colormaps import (
    HAIL_COLORS,
    PRECIPITATION_COLORS,
    RADAR_BOUNDARIES,
    RADAR_COLORS,
    SNOW_COLORS,
    radar_cmap,
    radar_norm,
)
from .plots import plot_contour, plot_field, plot_profile, plot_series
from .radar import (
    overlay_geojson_boundaries,
    render_radar_rgba,
    save_radar_gif,
    save_radar_png,
)

__all__ = [
    "HAIL_COLORS",
    "PRECIPITATION_COLORS",
    "RADAR_BOUNDARIES",
    "RADAR_COLORS",
    "SNOW_COLORS",
    "radar_cmap",
    "radar_norm",
    "plot_contour",
    "plot_field",
    "plot_profile",
    "plot_series",
    "overlay_geojson_boundaries",
    "render_radar_rgba",
    "save_field_gif",
    "save_radar_gif",
    "save_radar_png",
]
