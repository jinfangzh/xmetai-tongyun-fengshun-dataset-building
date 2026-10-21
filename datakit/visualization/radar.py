"""Radar image rendering and lightweight GIF export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .colormaps import RADAR_BOUNDARIES, RADAR_COLORS


def _rgba_colors(colors: Sequence[str]) -> np.ndarray:
    result = []
    for color in colors:
        text = color.removeprefix("#")
        if len(text) != 6:
            raise ValueError(f"Expected #RRGGBB color, got {color!r}")
        result.append([int(text[index : index + 2], 16) for index in (0, 2, 4)] + [255])
    return np.asarray(result, dtype=np.uint8)


def render_radar_rgba(
    data: xr.DataArray | np.ndarray,
    *,
    colors: Sequence[str] = RADAR_COLORS,
    boundaries: Sequence[float] = RADAR_BOUNDARIES,
    missing_alpha: int = 0,
) -> np.ndarray:
    """Map one 2-D dBZ field directly to an RGBA image using NumPy."""

    values = np.asarray(data.values if isinstance(data, xr.DataArray) else data, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Radar rendering expects a 2-D field, got {values.shape}")
    if not 0 <= missing_alpha <= 255:
        raise ValueError("missing_alpha must be in [0, 255]")
    palette = _rgba_colors(colors)
    edges = np.asarray(boundaries, dtype=float)
    if edges.ndim != 1 or edges.size != palette.shape[0] + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("boundaries must contain len(colors)+1 increasing values")
    indices = np.searchsorted(edges, values, side="right") - 1
    indices = np.clip(indices, 0, palette.shape[0] - 1)
    rgba = palette[indices]
    rgba[~np.isfinite(values), 3] = missing_alpha
    return rgba


def _geometry_lines(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if coordinates is None:
        return []
    if kind == "LineString":
        return [[(float(point[0]), float(point[1])) for point in coordinates]]
    if kind == "MultiLineString":
        return [[(float(point[0]), float(point[1])) for point in line] for line in coordinates]
    if kind == "Polygon":
        return [[(float(point[0]), float(point[1])) for point in ring] for ring in coordinates]
    if kind == "MultiPolygon":
        return [
            [(float(point[0]), float(point[1])) for point in ring]
            for polygon in coordinates
            for ring in polygon
        ]
    if kind == "GeometryCollection":
        return [
            line
            for child in geometry.get("geometries", [])
            for line in _geometry_lines(child)
        ]
    return []


def _geojson_lines(path: str | Path) -> list[list[tuple[float, float]]]:
    content = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if content.get("type") == "FeatureCollection":
        geometries = [feature.get("geometry") for feature in content.get("features", [])]
    elif content.get("type") == "Feature":
        geometries = [content.get("geometry")]
    else:
        geometries = [content]
    return [
        line
        for geometry in geometries
        if geometry is not None
        for line in _geometry_lines(geometry)
        if len(line) >= 2
    ]


def overlay_geojson_boundaries(
    rgba: np.ndarray,
    geojson_path: str | Path,
    *,
    bbox: tuple[float, float, float, float],
    color: Sequence[int] = (31, 41, 55, 204),
    width: int = 1,
) -> np.ndarray:
    """Overlay GeoJSON line/polygon boundaries on a georeferenced RGBA image."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError(
            "Boundary rendering requires Pillow; install xmetai-datakit[visualization]"
        ) from exc
    image_array = np.asarray(rgba, dtype=np.uint8)
    if image_array.ndim != 3 or image_array.shape[-1] != 4:
        raise ValueError("rgba must have shape (height, width, 4)")
    if len(color) not in {3, 4} or any(not 0 <= int(value) <= 255 for value in color):
        raise ValueError("color must be an RGB or RGBA sequence in [0, 255]")
    if width <= 0:
        raise ValueError("width must be positive")
    lon_min, lat_min, lon_max, lat_max = (float(value) for value in bbox)
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(f"Invalid bbox: {bbox}")
    height, image_width = image_array.shape[:2]
    x_scale = (image_width - 1) / (lon_max - lon_min) if image_width > 1 else 0.0
    y_scale = (height - 1) / (lat_max - lat_min) if height > 1 else 0.0
    image = Image.fromarray(image_array)
    drawing = ImageDraw.Draw(image, mode="RGBA")
    rgba_color = tuple(int(value) for value in color) + ((255,) if len(color) == 3 else ())
    for line in _geojson_lines(geojson_path):
        points = [
            ((lon - lon_min) * x_scale, (lat_max - lat) * y_scale)
            for lon, lat in line
        ]
        drawing.line(points, fill=rgba_color, width=width, joint="curve")
    return np.asarray(image)


def save_radar_png(
    data: xr.DataArray | np.ndarray,
    output_path: str | Path,
    *,
    geojson_path: str | Path | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    **render_options: Any,
) -> str:
    """Render and save one radar field as PNG."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "PNG export requires Pillow; install xmetai-datakit[visualization]"
        ) from exc
    rgba = render_radar_rgba(data, **render_options)
    if geojson_path is not None:
        if bbox is None:
            if isinstance(data, xr.DataArray) and {"lat", "lon"} <= set(data.coords):
                bbox = (
                    float(data.lon.min()),
                    float(data.lat.min()),
                    float(data.lon.max()),
                    float(data.lat.max()),
                )
            else:
                raise ValueError("bbox is required for a non-georeferenced radar array")
        rgba = overlay_geojson_boundaries(rgba, geojson_path, bbox=bbox)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(path)
    return str(path)


def save_radar_gif(
    data: xr.DataArray | Sequence[xr.DataArray | np.ndarray],
    output_path: str | Path,
    *,
    frame_dim: str = "time",
    duration_ms: int = 200,
    loop: int = 0,
    geojson_path: str | Path | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    boundary_options: Mapping[str, Any] | None = None,
    **render_options: Any,
) -> str:
    """Save a time-indexed radar array or explicit field sequence as GIF."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "GIF export requires Pillow; install xmetai-datakit[visualization]"
        ) from exc
    if duration_ms <= 0 or loop < 0:
        raise ValueError("duration_ms must be positive and loop non-negative")
    if isinstance(data, xr.DataArray):
        if frame_dim not in data.dims:
            raise ValueError(f"Frame dimension {frame_dim!r} not found")
        frames = [data.isel({frame_dim: index}) for index in range(data.sizes[frame_dim])]
    else:
        frames = list(data)
    if not frames:
        raise ValueError("At least one frame is required")
    resolved_bbox = bbox
    if geojson_path is not None and resolved_bbox is None:
        first = frames[0]
        if isinstance(first, xr.DataArray) and {"lat", "lon"} <= set(first.coords):
            resolved_bbox = (
                float(first.lon.min()),
                float(first.lat.min()),
                float(first.lon.max()),
                float(first.lat.max()),
            )
        else:
            raise ValueError("bbox is required for non-georeferenced radar frames")
    images = []
    for frame in frames:
        rgba = render_radar_rgba(frame, **render_options)
        if geojson_path is not None and resolved_bbox is not None:
            rgba = overlay_geojson_boundaries(
                rgba,
                geojson_path,
                bbox=resolved_bbox,
                **dict(boundary_options or {}),
            )
        images.append(Image.fromarray(rgba))
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,
    )
    return str(path)
