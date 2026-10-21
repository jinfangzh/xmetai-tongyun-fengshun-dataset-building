"""Small, format-agnostic animation helpers for gridded fields."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def save_field_gif(
    data: xr.DataArray | Sequence[xr.DataArray | np.ndarray],
    output_path: str | Path,
    *,
    frame_dim: str = "time",
    duration_ms: int = 200,
    loop: int = 0,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    missing_color: Sequence[int] = (0, 0, 0, 0),
    normalize_options: Mapping[str, Any] | None = None,
) -> str:
    """Render a gridded DataArray or a sequence of 2-D fields as a GIF.

    All frames share the same color limits. This makes the function suitable
    for quickly inspecting NetCDF, GRIB or Zarr fields after they are opened.
    """

    try:
        from matplotlib import colormaps
        from matplotlib.colors import Normalize
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "GIF export requires matplotlib and Pillow; install "
            "xmetai-datakit[visualization]"
        ) from exc
    if duration_ms <= 0 or loop < 0:
        raise ValueError("duration_ms must be positive and loop non-negative")
    if len(missing_color) != 4 or any(not 0 <= int(value) <= 255 for value in missing_color):
        raise ValueError("missing_color must be an RGBA sequence in [0, 255]")
    if isinstance(data, xr.DataArray):
        if frame_dim not in data.dims:
            raise ValueError(f"Frame dimension {frame_dim!r} not found")
        frame_values = [
            np.asarray(data.isel({frame_dim: index}).squeeze(drop=True).values, dtype=float)
            for index in range(data.sizes[frame_dim])
        ]
    else:
        frame_values = [
            np.asarray(frame.values if isinstance(frame, xr.DataArray) else frame, dtype=float)
            for frame in data
        ]
    if not frame_values:
        raise ValueError("At least one frame is required")
    if any(frame.ndim != 2 for frame in frame_values):
        raise ValueError("Every animation frame must be two-dimensional")
    finite_parts = [frame[np.isfinite(frame)] for frame in frame_values]
    finite_values = np.concatenate(finite_parts)
    if finite_values.size == 0:
        raise ValueError("Animation frames contain no finite values")
    lower = float(np.min(finite_values)) if vmin is None else float(vmin)
    upper = float(np.max(finite_values)) if vmax is None else float(vmax)
    if lower == upper and vmin is None and vmax is None:
        padding = max(abs(lower) * 0.01, 0.5)
        lower -= padding
        upper += padding
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("vmin and vmax must be finite and satisfy vmin < vmax")
    options = dict(normalize_options or {})
    conflicts = sorted({"vmin", "vmax"} & set(options))
    if conflicts:
        raise ValueError(f"normalize_options cannot override {conflicts}")
    normalizer = Normalize(vmin=lower, vmax=upper, **options)
    color_map = colormaps.get_cmap(cmap)
    images = []
    for frame in frame_values:
        rgba = np.rint(color_map(normalizer(frame)) * 255).astype(np.uint8)
        rgba[~np.isfinite(frame)] = np.asarray(missing_color, dtype=np.uint8)
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
