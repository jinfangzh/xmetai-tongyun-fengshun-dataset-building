"""Pure-array quality control for composite radar reflectivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class RadarQCConfig:
    min_dbz: float = 0.0
    max_dbz: float = 75.0
    speckle_dbz: float = 5.0
    speckle_window: int = 3
    speckle_min_neighbors: int = 2
    spike_dbz: float = 15.0
    spike_delta_dbz: float = 25.0
    spike_max_neighbors: int = 2
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_dbz) or not np.isfinite(self.max_dbz):
            raise ValueError("Radar QC range must be finite")
        if self.min_dbz > self.max_dbz:
            raise ValueError("min_dbz cannot exceed max_dbz")
        if self.speckle_window < 3 or self.speckle_window % 2 != 1:
            raise ValueError("speckle_window must be an odd integer >= 3")
        if self.speckle_min_neighbors < 0 or self.spike_max_neighbors < 0:
            raise ValueError("Radar neighbor thresholds cannot be negative")
        if self.speckle_min_neighbors > self.speckle_window**2:
            raise ValueError("speckle_min_neighbors exceeds the speckle window")
        if self.spike_max_neighbors > self.speckle_window**2:
            raise ValueError("spike_max_neighbors exceeds the speckle window")
        if not all(
            np.isfinite(value)
            for value in (self.speckle_dbz, self.spike_dbz, self.spike_delta_dbz)
        ):
            raise ValueError("Radar QC thresholds must be finite")
        if self.spike_delta_dbz < 0:
            raise ValueError("spike_delta_dbz cannot be negative")


def _neighbor_count(mask: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    padded = np.pad(mask.astype(np.uint8), radius, mode="constant")
    dtype = np.uint16 if window * window > 255 else np.uint8
    result = np.zeros(mask.shape, dtype=dtype)
    height, width = mask.shape
    for row in range(window):
        for column in range(window):
            result += padded[row : row + height, column : column + width]
    return result


def _neighbor_max(values: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    padded = np.pad(values, radius, mode="constant", constant_values=-np.inf)
    result = np.full(values.shape, -np.inf, dtype=np.float32)
    height, width = values.shape
    for row in range(window):
        for column in range(window):
            if row == radius and column == radius:
                continue
            np.maximum(
                result,
                padded[row : row + height, column : column + width],
                out=result,
            )
    return result


def quality_control_reflectivity(
    value: xr.DataArray | np.ndarray,
    *,
    config: RadarQCConfig | None = None,
    static_clutter_mask: xr.DataArray | np.ndarray | None = None,
) -> xr.DataArray | np.ndarray:
    """Remove invalid values, unsupported speckles and isolated dBZ spikes.

    The last two dimensions are treated as the radar grid. Leading dimensions
    such as ``time`` are processed independently.
    """

    settings = config or RadarQCConfig()
    source = np.asarray(value.values if isinstance(value, xr.DataArray) else value)
    if source.ndim < 2:
        raise ValueError("Radar reflectivity must contain at least two dimensions")
    mask = None
    if static_clutter_mask is not None:
        mask = np.asarray(
            static_clutter_mask.values
            if isinstance(static_clutter_mask, xr.DataArray)
            else static_clutter_mask,
            dtype=bool,
        )
        if mask.shape != source.shape[-2:]:
            raise ValueError(
                f"static_clutter_mask shape {mask.shape} does not match "
                f"radar grid {source.shape[-2:]}"
            )

    output = np.empty(source.shape, dtype=np.float32)
    for index in np.ndindex(source.shape[:-2]):
        field = np.asarray(source[index], dtype=np.float32).copy()
        invalid = (
            ~np.isfinite(field)
            | (field < settings.min_dbz)
            | (field > settings.max_dbz)
        )
        field[invalid] = settings.fill_value
        if mask is not None:
            field[mask] = settings.fill_value
        echo = field >= settings.speckle_dbz
        if np.any(echo):
            support = _neighbor_count(echo, settings.speckle_window)
            field[echo & (support < settings.speckle_min_neighbors)] = settings.fill_value
            if settings.spike_delta_dbz > 0:
                remaining = field >= settings.speckle_dbz
                support = _neighbor_count(remaining, settings.speckle_window)
                neighbor_max = _neighbor_max(field, settings.speckle_window)
                spikes = (
                    (field >= settings.spike_dbz)
                    & (field - neighbor_max >= settings.spike_delta_dbz)
                    & (support <= settings.spike_max_neighbors)
                )
                field[spikes] = settings.fill_value
        output[index] = field

    if isinstance(value, xr.DataArray):
        result = value.copy(data=output)
        result.attrs = dict(value.attrs)
        result.attrs["quality_control"] = "range,speckle,spike,static_clutter"
        return result
    return output
