"""Private radar dataset-pipeline implementation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from datakit.io import write_zarr
from datakit.radar import (
    CinradRecord,
    RadarBBox,
    RadarGridSpec,
    RadarQCConfig,
    discover_cinrad_files,
    quality_control_reflectivity,
    read_cinrad,
    read_meteonet_npz,
)


@dataclass(frozen=True)
class RadarContinuityReport:
    expected_frequency: pd.Timedelta
    sample_count: int
    missing_count: int
    gaps: tuple[tuple[pd.Timestamp, pd.Timestamp, int], ...]

    @property
    def continuous(self) -> bool:
        return not self.gaps


def analyze_radar_continuity(
    values: Sequence[object] | Sequence[CinradRecord],
    *,
    frequency: str | pd.Timedelta = "6min",
) -> RadarContinuityReport:
    """Describe missing steps in a timestamp or radar-record sequence."""

    expected = pd.Timedelta(frequency)
    if pd.isna(expected) or expected <= pd.Timedelta(0):
        raise ValueError("frequency must be a fixed positive interval")
    record_flags = [isinstance(value, CinradRecord) for value in values]
    if any(record_flags) and not all(record_flags):
        raise TypeError("values cannot mix CinradRecord objects and timestamps")
    if record_flags and all(record_flags):
        times = pd.DatetimeIndex(
            [value.time for value in values if isinstance(value, CinradRecord)]
        )
    else:
        times = pd.DatetimeIndex(pd.to_datetime(values))
    if times.hasnans or not times.is_monotonic_increasing or not times.is_unique:
        raise ValueError("Radar timestamps must be valid, sorted and unique")
    gaps = []
    missing = 0
    for first, second in zip(times[:-1], times[1:]):
        difference = second - first
        if difference == expected:
            continue
        if difference < expected or difference % expected != pd.Timedelta(0):
            gaps.append((first, second, -1))
            continue
        count = int(difference / expected) - 1
        missing += count
        gaps.append((first, second, count))
    return RadarContinuityReport(expected, len(times), missing, tuple(gaps))


def _validate_common_grid(records: Sequence[CinradRecord]) -> None:
    first = records[0].header
    for record in records[1:]:
        current = record.header
        if not (
            current.height == first.height
            and current.width == first.width
            and np.isclose(current.left_lon, first.left_lon)
            and np.isclose(current.top_lat, first.top_lat)
            and np.isclose(current.lon_step, first.lon_step)
            and np.isclose(current.lat_step, first.lat_step)
        ):
            raise ValueError(
                f"Radar grids do not match: {first.path} and {current.path}"
            )


def convert_cinrad_to_zarr(
    paths: str | Path | Iterable[str | Path],
    destination: str | Path,
    *,
    pattern: str = "*.bin",
    time_range: tuple[object, object] | None = None,
    raw_grid: RadarGridSpec | None = None,
    bbox: RadarBBox | Sequence[float] | None = None,
    crop: Sequence[int] | None = None,
    stride: int = 1,
    fill_value: float | None = np.nan,
    qc: bool = True,
    qc_config: RadarQCConfig | None = None,
    static_clutter_mask: xr.DataArray | np.ndarray | None = None,
    frequency: str | pd.Timedelta = "6min",
    chunks: Mapping[str, int] | None = None,
    overwrite: bool = False,
) -> tuple[str, RadarContinuityReport]:
    """Decode timestamped radar binaries and write one CF-like Zarr product."""

    output = Path(destination).expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    records = discover_cinrad_files(
        paths,
        pattern=pattern,
        time_range=time_range,
        raw_grid=raw_grid,
    )
    _validate_common_grid(records)
    continuity = analyze_radar_continuity(records, frequency=frequency)
    fields = []
    for record in records:
        field = read_cinrad(
            record.path,
            header=record.header,
            fill_value=fill_value,
            bbox=bbox,
            crop=crop,
            stride=stride,
        )
        if qc:
            field = quality_control_reflectivity(
                field,
                config=qc_config,
                static_clutter_mask=static_clutter_mask,
            )
        fields.append(field.expand_dims(time=[record.time.to_datetime64()]))
    combined = xr.concat(fields, "time", join="exact").to_dataset(name="reflectivity")
    combined.attrs.update(
        source_format="CINRAD composite reflectivity binary",
        expected_frequency=str(continuity.expected_frequency),
        missing_time_steps=continuity.missing_count,
    )
    written = write_zarr(
        combined,
        output,
        mode="w",
        consolidated=True,
        chunks=chunks,
    )
    return written, continuity


def convert_meteonet_npz_to_zarr(
    paths: str | Path | Iterable[str | Path],
    destination: str | Path,
    *,
    chunks: Mapping[str, int] | None = None,
    overwrite: bool = False,
    **read_options: Any,
) -> str:
    """Read a MeteoNet NPZ collection and persist its xarray representation."""

    output = Path(destination).expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    dataset = read_meteonet_npz(paths, **read_options)
    return write_zarr(
        dataset,
        output,
        mode="w",
        consolidated=True,
        chunks=chunks,
    )
