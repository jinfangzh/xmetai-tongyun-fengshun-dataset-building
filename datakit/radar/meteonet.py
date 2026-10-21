"""Reader for MeteoNet radar NPZ chunk collections."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import warnings
import zipfile

import numpy as np
from numpy.lib import format as npy_format
import pandas as pd
import xarray as xr

from .schema import RadarGridSpec


DEFAULT_SCALE_FACTORS = {"data": 10.0, "prob": 1.0, "height": 1.0}
DEFAULT_MISSING_VALUES = {"data": (-200,), "prob": (-100,), "height": (-200,)}


def _paths(
    paths: str | Path | Iterable[str | Path],
    pattern: str,
) -> list[Path]:
    values = [paths] if isinstance(paths, (str, Path)) else list(paths)
    files: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            files.extend(item for item in path.rglob(pattern) if item.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No MeteoNet NPZ files matched {pattern!r}")
    return files


def _npy_header(
    archive: zipfile.ZipFile,
    member: str,
) -> tuple[tuple[int, ...], bool, np.dtype]:
    with archive.open(member) as stream:
        version = npy_format.read_magic(stream)
        if version == (1, 0):
            return npy_format.read_array_header_1_0(stream)
        if version == (2, 0):
            return npy_format.read_array_header_2_0(stream)
        return npy_format._read_array_header(stream, version)  # noqa: SLF001


def _metadata(
    path: Path,
    keys: Sequence[str],
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    result = {}
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        for key in keys:
            member = f"{key}.npy"
            if member not in members:
                raise KeyError(f"NPZ file {path} does not contain {key!r}")
            shape, _, dtype = _npy_header(archive, member)
            result[key] = (tuple(int(size) for size in shape), np.dtype(dtype))
    return result


def _dates(path: Path) -> tuple[pd.DatetimeIndex, set[pd.Timestamp]]:
    warning_class = getattr(
        getattr(np, "exceptions", object),
        "VisibleDeprecationWarning",
        Warning,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", warning_class)
        with np.load(path, allow_pickle=True) as archive:
            if "dates" not in archive.files:
                raise KeyError(f"NPZ file {path} does not contain 'dates'")
            dates = pd.DatetimeIndex(pd.to_datetime(archive["dates"]))
            missing = (
                pd.DatetimeIndex(pd.to_datetime(archive["miss_dates"]))
                if "miss_dates" in archive.files
                else pd.DatetimeIndex([])
            )
    if dates.hasnans or not dates.is_unique:
        raise ValueError(f"MeteoNet dates must be valid and unique within {path}")
    return dates, {pd.Timestamp(value) for value in missing if not pd.isna(value)}


def _scale_mapping(
    channels: Sequence[str],
    value: float | Mapping[str, float] | None,
    defaults: Mapping[str, float],
) -> dict[str, float]:
    if isinstance(value, Mapping):
        result = {
            channel: float(value.get(channel, defaults.get(channel, 1.0)))
            for channel in channels
        }
    elif value is None:
        result = {channel: float(defaults.get(channel, 1.0)) for channel in channels}
    else:
        result = {channel: float(value) for channel in channels}
    if any(factor == 0 or not np.isfinite(factor) for factor in result.values()):
        raise ValueError("scale_factors must be finite and non-zero")
    return result


def _fill_mapping(
    channels: Sequence[str],
    value: float | Mapping[str, float | None] | None,
) -> dict[str, float | None]:
    if isinstance(value, Mapping):
        result = {}
        for channel in channels:
            item = value.get(channel, 0.0)
            result[channel] = None if item is None else float(item)
        return result
    if value is None:
        return {channel: None for channel in channels}
    return {channel: float(value) for channel in channels}


def _missing_mapping(
    channels: Sequence[str],
    value: Mapping[str, Sequence[float] | float] | Sequence[float] | float | None,
) -> dict[str, tuple[float, ...]]:
    def values(item: object) -> tuple[float, ...]:
        if isinstance(item, (int, float, np.integer, np.floating)):
            return (float(item),)
        return tuple(float(member) for member in np.asarray(item).reshape(-1))

    if isinstance(value, Mapping):
        return {
            channel: values(value.get(channel, DEFAULT_MISSING_VALUES.get(channel, ())))
            for channel in channels
        }
    if value is None:
        return {
            channel: values(DEFAULT_MISSING_VALUES.get(channel, ()))
            for channel in channels
        }
    shared = values(value)
    return {channel: shared for channel in channels}


def read_meteonet_npz(
    paths: str | Path | Iterable[str | Path],
    *,
    pattern: str = "*.npz",
    channels: Sequence[str] = ("data",),
    time_range: tuple[object, object] | None = None,
    exclude_missing_dates: bool = True,
    scale_factors: float | Mapping[str, float] | None = None,
    missing_values: (
        Mapping[str, Sequence[float] | float]
        | Sequence[float]
        | float
        | None
    ) = None,
    fill_values: float | Mapping[str, float | None] | None = 0.0,
    crop: Sequence[int] | None = None,
    stride: int = 1,
    grid: RadarGridSpec | None = None,
) -> xr.Dataset:
    """Read MeteoNet NPZ chunks into one time-sorted xarray Dataset.

    ``fill_values=0.0`` replaces encoded missing values, ``None`` preserves
    their decoded numeric values, and ``np.nan`` converts them to NaN.
    """

    requested = tuple(str(channel) for channel in channels)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("channels must be non-empty and unique")
    if stride <= 0:
        raise ValueError("stride must be positive")
    files = _paths(paths, pattern)
    scales = _scale_mapping(requested, scale_factors, DEFAULT_SCALE_FACTORS)
    fills = _fill_mapping(requested, fill_values)
    missing = _missing_mapping(requested, missing_values)
    start = pd.Timestamp(time_range[0]) if time_range else None
    end = pd.Timestamp(time_range[1]) if time_range else None
    if start is not None and end is not None and start > end:
        raise ValueError("time_range start cannot be after end")

    values: dict[str, list[np.ndarray]] = {channel: [] for channel in requested}
    times: list[pd.Timestamp] = []
    sources: list[str] = []
    source_indices: list[int] = []
    grid_shape: tuple[int, int] | None = None
    seen: set[pd.Timestamp] = set()

    for path in files:
        metadata = _metadata(path, ("dates", *requested))
        dates, missing_dates = _dates(path)
        if metadata["dates"][0] != (len(dates),):
            raise ValueError(f"dates metadata does not match date count in {path}")
        for channel in requested:
            shape, _ = metadata[channel]
            if len(shape) != 3 or shape[0] != len(dates):
                raise ValueError(
                    f"MeteoNet {channel!r} must have shape (time, y, x), got {shape}"
                )
            if grid_shape is None:
                grid_shape = (shape[1], shape[2])
            elif grid_shape != (shape[1], shape[2]):
                raise ValueError(
                    f"MeteoNet grids do not match: {grid_shape} and {shape[1:]}"
                )
        if grid is not None and grid_shape != (grid.height, grid.width):
            raise ValueError(
                f"MeteoNet grid {grid_shape} does not match grid specification "
                f"{(grid.height, grid.width)}"
            )

        keep = []
        for index, timestamp in enumerate(dates):
            timestamp = pd.Timestamp(timestamp)
            if exclude_missing_dates and timestamp in missing_dates:
                continue
            if start is not None and timestamp < start:
                continue
            if end is not None and timestamp > end:
                continue
            if timestamp in seen:
                raise ValueError(f"Duplicate MeteoNet timestamp: {timestamp}")
            seen.add(timestamp)
            keep.append(index)
            times.append(timestamp)
            sources.append(str(path))
            source_indices.append(index)
        if not keep:
            continue
        with np.load(path, allow_pickle=True) as archive:
            for channel in requested:
                selected = np.asarray(archive[channel])[keep]
                if crop is not None:
                    if len(crop) != 4:
                        raise ValueError("crop must be (row_start, row_stop, col_start, col_stop)")
                    row_start, row_stop, column_start, column_stop = (int(item) for item in crop)
                    if (
                        row_start < 0
                        or column_start < 0
                        or row_stop <= row_start
                        or column_stop <= column_start
                    ):
                        raise ValueError(f"Invalid crop: {crop}")
                    if row_stop > selected.shape[1] or column_stop > selected.shape[2]:
                        raise ValueError(
                            f"crop {crop} exceeds MeteoNet grid {selected.shape[1:]}"
                        )
                    selected = selected[:, row_start:row_stop, column_start:column_stop]
                selected = selected[:, ::stride, ::stride]
                invalid = np.isin(selected, np.asarray(missing[channel], dtype=selected.dtype))
                decoded = selected.astype(np.float32) / scales[channel]
                if fills[channel] is not None:
                    decoded[invalid] = fills[channel]
                values[channel].append(decoded)

    if not times or grid_shape is None:
        raise ValueError("No MeteoNet frames remain after filtering")
    order = np.argsort(np.asarray(times, dtype="datetime64[ns]"))
    time_coord = np.asarray(times, dtype="datetime64[ns]")[order]
    combined = {
        channel: np.concatenate(parts, axis=0)[order]
        for channel, parts in values.items()
    }
    height, width = next(iter(combined.values())).shape[-2:]
    if grid is not None:
        if crop is None:
            row_start = column_start = 0
        else:
            row_start, _, column_start, _ = (int(item) for item in crop)
        latitude = (grid.top_lat - np.arange(grid.height) * grid.lat_step)[
            row_start::stride
        ][:height]
        longitude = (grid.left_lon + np.arange(grid.width) * grid.lon_step)[
            column_start::stride
        ][:width]
        dimensions = ("time", "lat", "lon")
        coordinates = {"time": time_coord, "lat": latitude, "lon": longitude}
    else:
        dimensions = ("time", "y", "x")
        coordinates = {"time": time_coord, "y": np.arange(height), "x": np.arange(width)}
    dataset = xr.Dataset(
        {channel: (dimensions, data) for channel, data in combined.items()},
        coords=coordinates,
    )
    dataset = dataset.assign_coords(
        source_file=("time", np.asarray(sources, dtype=str)[order]),
        source_index=("time", np.asarray(source_indices, dtype=np.int64)[order]),
    )
    for channel in requested:
        dataset[channel].attrs.update(
            scale_factor_applied=scales[channel],
            source_format="MeteoNet NPZ",
        )
    return dataset
