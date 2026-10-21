from __future__ import annotations

PRECIPITATION_COLORS = (
    "#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00",
    "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000",
    "#f800fd", "#9854c6", "#fdfdfd",
)
RADAR_COLORS = (
    "#FFFFFF", "#808080", "#00FFFF", "#00BFFF", "#0000FF", "#00FF00",
    "#32CD32", "#008000", "#FFFF00", "#DAA520", "#FFA500", "#FF0000",
    "#8B0000", "#800000", "#FF00FF", "#8A2BE2", "#FFFFFF",
)
RADAR_BOUNDARIES = (-32768, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 32767)
SNOW_COLORS = (
    "#FFFFFF", "#C2BFFF", "#867FFF", "#493FFF", "#0D00FF", "#0C3FBF",
    "#0B7F7F", "#0ABF3F", "#09FF00", "#46E800", "#84D200", "#C1BC00",
    "#FFA600", "#FF7C00", "#FF5300", "#FF2900", "#FF0000", "#FF004E",
    "#FF009C", "#FF00EA",
)
HAIL_COLORS = (
    "#FFFFFF", "#AEAAFF", "#5D55FF", "#0D00FF", "#0B55AA", "#0AAA55",
    "#09FF00", "#5BE100", "#ADC300", "#FFA600", "#FF6E00", "#FF3700",
    "#FF0000", "#FF0075", "#FF00EA",
)


def _matplotlib_colors():
    try:
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib is required for colormap objects; install xmetai-datakit[visualization]"
        ) from exc
    return BoundaryNorm, ListedColormap


def radar_cmap():
    """Return the standard discrete radar ``ListedColormap``."""

    _, listed_colormap = _matplotlib_colors()
    return listed_colormap(RADAR_COLORS)


def radar_norm():
    """Return the standard radar reflectivity ``BoundaryNorm``."""

    boundary_norm, _ = _matplotlib_colors()
    return boundary_norm(RADAR_BOUNDARIES, ncolors=len(RADAR_COLORS), clip=True)
