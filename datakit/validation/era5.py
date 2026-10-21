"""ERA5-specific validation helpers for day-level channel datasets."""

from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np
import xarray as xr


def validate_era5_dataset_schema(
    dataset: xr.Dataset,
    *,
    variable: str = "data",
    require_mask: bool = True,
    required_attrs: Sequence[str] = (
        "dataset_id",
        "schema_version",
        "content_version",
        "data_revision",
        "geospatial_lat_range",
        "geospatial_lon_range",
    ),
) -> None:
    required_dims = ("time", "channel", "lat", "lon")
    for dim in required_dims:
        if dim not in dataset.dims:
            raise ValueError(f"Missing required dimension: {dim!r}")
        if int(dataset.sizes[dim]) <= 0:
            raise ValueError(f"Dimension {dim!r} must be non-empty")

    required_coords = ("time", "channel", "lat", "lon")
    for coord in required_coords:
        if coord not in dataset.coords:
            raise ValueError(f"Missing required coordinate: {coord!r}")

    if variable not in dataset.data_vars:
        raise ValueError(f"Missing required data variable: {variable!r}")
    data = dataset[variable]
    expected_dims = ("time", "channel", "lat", "lon")
    if tuple(data.dims) != expected_dims:
        raise ValueError(f"{variable!r} dims must be {expected_dims}, got {tuple(data.dims)}")

    for name in ("mean", "std"):
        if name not in dataset.data_vars:
            raise ValueError(f"Missing required data variable: {name!r}")
        value = dataset[name]
        if tuple(value.dims) != ("channel",):
            raise ValueError(f"{name!r} dims must be ('channel',), got {tuple(value.dims)}")
        if int(value.sizes["channel"]) != int(dataset.sizes["channel"]):
            raise ValueError(f"{name!r} length must match channel dimension")

    if require_mask:
        if "mask" not in dataset.data_vars:
            raise ValueError("Missing required data variable: 'mask'")
        mask = dataset["mask"]
        for dim in ("mask_channel", "lat", "lon"):
            if dim not in mask.dims:
                raise ValueError(f"'mask' missing required dimension: {dim!r}")
            if int(mask.sizes[dim]) <= 0:
                raise ValueError(f"'mask' dimension {dim!r} must be non-empty")

    for attr in required_attrs:
        if attr not in dataset.attrs:
            raise ValueError(f"Missing required attribute: {attr!r}")


def expected_six_hourly_times(day: date) -> np.ndarray:
    base = np.datetime64(day.isoformat(), "h")
    return base + np.asarray([0, 6, 12, 18], dtype="timedelta64[h]")


def validate_six_hourly_day(
    dataset: xr.Dataset,
    day: date,
    *,
    time_coord: str = "time",
) -> None:
    if time_coord not in dataset.coords:
        raise ValueError(f"Missing coordinate {time_coord!r}")
    actual = np.asarray(dataset[time_coord].values).astype("datetime64[h]")
    expected = expected_six_hourly_times(day)
    if not np.array_equal(actual, expected):
        raise ValueError(f"Unexpected day slots: got={actual.tolist()}, expected={expected.tolist()}")


def validate_era5_layout(
    *,
    channels: Sequence[str],
    data_shape: tuple[int, int, int, int],
    lat: np.ndarray,
    lon: np.ndarray,
    expected_channels: Sequence[str],
    expected_lat: np.ndarray,
    expected_lon: np.ndarray,
    expected_time_steps: int = 4,
) -> None:
    if list(channels) != list(expected_channels):
        raise ValueError("Channel names/order verification failed")
    expected_shape = (
        int(expected_time_steps),
        len(expected_channels),
        int(np.asarray(expected_lat).shape[0]),
        int(np.asarray(expected_lon).shape[0]),
    )
    if data_shape != expected_shape:
        raise ValueError(f"Unexpected data shape: {data_shape}")
    if not np.array_equal(np.asarray(lat, dtype="f4"), np.asarray(expected_lat, dtype="f4")):
        raise ValueError("Latitude coordinate verification failed")
    if not np.array_equal(np.asarray(lon, dtype="f4"), np.asarray(expected_lon, dtype="f4")):
        raise ValueError("Longitude coordinate verification failed")


def validate_era5_c116_layout(
    *,
    channels: Sequence[str],
    data_shape: tuple[int, int, int, int],
    lat: np.ndarray,
    lon: np.ndarray,
    expected_channels: Sequence[str],
    expected_lat: np.ndarray,
    expected_lon: np.ndarray,
) -> None:
    """Backward-compatible wrapper for the historical c116 contract."""

    validate_era5_layout(
        channels=channels,
        data_shape=data_shape,
        lat=lat,
        lon=lon,
        expected_channels=expected_channels,
        expected_lat=expected_lat,
        expected_lon=expected_lon,
        expected_time_steps=4,
    )
