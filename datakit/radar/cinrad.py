"""CINRAD composite-reflectivity binary metadata and decoding."""

from __future__ import annotations

import bz2
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import struct

import numpy as np
import pandas as pd
import xarray as xr

from .schema import RadarBBox, RadarGridSpec


_TIME_PATTERN = re.compile(r"(?P<date>\d{8})[_-]?(?P<time>\d{4}(?:\d{2})?)")
_PRODUCT_PATTERN = re.compile(r"RADR_(?P<product>[A-Z0-9]+)_", re.IGNORECASE)


@dataclass(frozen=True)
class CinradGridHeader:
    """Metadata decoded from a composite-reflectivity product."""

    path: Path
    product: str
    product_name: str
    header_size: int
    compressed_size: int
    data_bytes: int
    timestamp: pd.Timestamp | None
    left_lon: float
    top_lat: float
    right_lon: float
    center_lon: float
    center_lat: float
    width: int
    height: int
    lon_step: float
    lat_step: float

    @property
    def bottom_lat(self) -> float:
        return self.top_lat - self.lat_step * (self.height - 1)

    @property
    def lon(self) -> np.ndarray:
        return self.left_lon + np.arange(self.width, dtype=np.float32) * np.float32(
            self.lon_step
        )

    @property
    def lat(self) -> np.ndarray:
        return self.top_lat - np.arange(self.height, dtype=np.float32) * np.float32(
            self.lat_step
        )


@dataclass(frozen=True)
class CinradRecord:
    """One timestamped radar file and its decoded grid metadata."""

    time: pd.Timestamp
    path: Path
    header: CinradGridHeader


def _decode_ascii(value: bytes) -> str:
    return value.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()


def _read_i32(value: bytes, offset: int) -> int:
    return struct.unpack_from("<i", value, offset)[0]


def _read_u32(value: bytes, offset: int) -> int:
    return struct.unpack_from("<I", value, offset)[0]


def _header_time(value: bytes) -> pd.Timestamp | None:
    year, month, day, hour, minute, second = struct.unpack_from("<6H", value, 0x64)
    if year <= 0 or month <= 0 or day <= 0:
        return None
    try:
        return pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
        )
    except ValueError:
        return None


def _filename_time(path: Path) -> pd.Timestamp | None:
    match = _TIME_PATTERN.search(path.name)
    if match is None:
        return None
    clock = match.group("time")
    clock_format = "%H%M%S" if len(clock) == 6 else "%H%M"
    return pd.to_datetime(
        f"{match.group('date')}{clock}", format=f"%Y%m%d{clock_format}"
    )


def read_cinrad_header(
    path: str | Path,
    *,
    raw_grid: RadarGridSpec | None = None,
) -> CinradGridHeader:
    """Read a 256-byte ACHN header or describe a headerless BZip2 grid."""

    radar_path = Path(path).expanduser()
    if not radar_path.is_file():
        raise FileNotFoundError(radar_path)
    with radar_path.open("rb") as stream:
        header = stream.read(256)
    if header.startswith(b"BZh"):
        if raw_grid is None:
            raise ValueError("Headerless BZip2 radar data requires raw_grid")
        match = _PRODUCT_PATTERN.search(radar_path.name)
        product = match.group("product").upper() if match else radar_path.stem
        return CinradGridHeader(
            path=radar_path,
            product=product,
            product_name=product,
            header_size=0,
            compressed_size=radar_path.stat().st_size,
            data_bytes=raw_grid.height * raw_grid.width * np.dtype("<i2").itemsize,
            timestamp=_filename_time(radar_path),
            left_lon=raw_grid.left_lon,
            top_lat=raw_grid.top_lat,
            right_lon=raw_grid.left_lon + (raw_grid.width - 1) * raw_grid.lon_step,
            center_lon=raw_grid.left_lon + (raw_grid.width - 1) * raw_grid.lon_step / 2,
            center_lat=raw_grid.top_lat - (raw_grid.height - 1) * raw_grid.lat_step / 2,
            width=raw_grid.width,
            height=raw_grid.height,
            lon_step=raw_grid.lon_step,
            lat_step=raw_grid.lat_step,
        )
    if len(header) < 256:
        raise ValueError(f"Radar file is shorter than its 256-byte header: {radar_path}")
    if header[:3] != b"MOC":
        raise ValueError(f"Unsupported radar magic {header[:4]!r}")

    width = _read_u32(header, 0x94)
    height = _read_u32(header, 0x98)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid radar grid shape: {(height, width)}")
    left_lon = _read_i32(header, 0x80) / 1000.0
    top_lat = _read_i32(header, 0x84) / 1000.0
    right_lon = _read_i32(header, 0x88) / 1000.0
    lon_step = _read_u32(header, 0x9C) / 10000.0
    lat_step = _read_u32(header, 0xA0) / 10000.0
    if lon_step <= 0 and width > 1:
        lon_step = (right_lon - left_lon) / (width - 1)
    if lat_step <= 0:
        lat_step = lon_step
    if lon_step <= 0 or lat_step <= 0:
        raise ValueError("Radar grid resolution must be positive")
    return CinradGridHeader(
        path=radar_path,
        product=_decode_ascii(header[0x10:0x18]),
        product_name=_decode_ascii(header[0x18:0x58]),
        header_size=_read_u32(header, 0x58) or 256,
        compressed_size=_read_u32(header, 0x5C),
        data_bytes=_read_u32(header, 0xAC),
        timestamp=_header_time(header),
        left_lon=left_lon,
        top_lat=top_lat,
        right_lon=right_lon,
        center_lon=_read_i32(header, 0x8C) / 1000.0,
        center_lat=_read_i32(header, 0x90) / 1000.0,
        width=width,
        height=height,
        lon_step=lon_step,
        lat_step=lat_step,
    )


def cinrad_bbox_slices(
    header: CinradGridHeader,
    bbox: RadarBBox | Sequence[float],
) -> tuple[slice, slice]:
    """Convert a lon/lat bounding box into radar row/column slices."""

    if len(bbox) != 4:
        raise ValueError("bbox must be (lon_min, lat_min, lon_max, lat_max)")
    lon_min, lat_min, lon_max, lat_max = (float(value) for value in bbox)
    if lon_min > lon_max or lat_min > lat_max:
        raise ValueError(f"Invalid bbox: {bbox}")
    column_start = max(int(np.floor((lon_min - header.left_lon) / header.lon_step)), 0)
    column_stop = min(
        int(np.ceil((lon_max - header.left_lon) / header.lon_step)) + 1,
        header.width,
    )
    row_start = max(int(np.floor((header.top_lat - lat_max) / header.lat_step)), 0)
    row_stop = min(
        int(np.ceil((header.top_lat - lat_min) / header.lat_step)) + 1,
        header.height,
    )
    if row_start >= row_stop or column_start >= column_stop:
        raise ValueError(f"bbox {bbox} does not overlap the radar grid")
    return slice(row_start, row_stop), slice(column_start, column_stop)


def _crop_slices(crop: Sequence[int], shape: tuple[int, int]) -> tuple[slice, slice]:
    if len(crop) != 4:
        raise ValueError("crop must be (row_start, row_stop, col_start, col_stop)")
    row_start, row_stop, column_start, column_stop = (int(value) for value in crop)
    if (
        row_start < 0
        or column_start < 0
        or row_stop <= row_start
        or column_stop <= column_start
    ):
        raise ValueError(f"Invalid crop: {crop}")
    if row_stop > shape[0] or column_stop > shape[1]:
        raise ValueError(f"crop {crop} exceeds radar grid {shape}")
    return slice(row_start, row_stop), slice(column_start, column_stop)


def read_cinrad(
    path: str | Path,
    *,
    header: CinradGridHeader | None = None,
    raw_grid: RadarGridSpec | None = None,
    scale_factor: float = 10.0,
    missing_values: Sequence[int] = (-32768,),
    no_echo_values: Sequence[int] = (-1280,),
    fill_value: float | None = np.nan,
    bbox: RadarBBox | Sequence[float] | None = None,
    crop: Sequence[int] | None = None,
    stride: int = 1,
    name: str = "reflectivity",
) -> xr.DataArray:
    """Decode one composite-reflectivity product into an xarray field."""

    if scale_factor == 0 or not np.isfinite(scale_factor):
        raise ValueError("scale_factor must be finite and non-zero")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if bbox is not None and crop is not None:
        raise ValueError("bbox and crop are mutually exclusive")
    radar_path = Path(path).expanduser()
    metadata = header or read_cinrad_header(radar_path, raw_grid=raw_grid)
    with radar_path.open("rb") as stream:
        stream.seek(metadata.header_size)
        compressed = stream.read(metadata.compressed_size or -1)
    payload = bz2.decompress(compressed)
    expected = metadata.height * metadata.width * np.dtype("<i2").itemsize
    if len(payload) != expected:
        raise ValueError(f"Unexpected radar payload size: got {len(payload)}, expected {expected}")
    raw = np.frombuffer(payload, dtype="<i2").reshape(metadata.height, metadata.width)
    latitude = metadata.lat
    longitude = metadata.lon
    if bbox is not None:
        row_slice, column_slice = cinrad_bbox_slices(metadata, bbox)
    elif crop is not None:
        row_slice, column_slice = _crop_slices(crop, raw.shape)
    else:
        row_slice, column_slice = slice(None), slice(None)
    raw = raw[row_slice, column_slice][::stride, ::stride]
    latitude = latitude[row_slice][::stride]
    longitude = longitude[column_slice][::stride]

    missing_codes = tuple(missing_values)
    no_echo_codes = tuple(no_echo_values)
    missing = np.isin(raw, np.asarray(missing_codes, dtype=np.int16))
    no_echo = np.isin(raw, np.asarray(no_echo_codes, dtype=np.int16))
    special = missing | no_echo
    values = raw.astype(np.float32, copy=True)
    values[~special] /= float(scale_factor)
    if fill_value is None:
        if missing_codes:
            values[missing] = float(missing_codes[0])
        if no_echo_codes:
            values[no_echo] = float(no_echo_codes[0]) / float(scale_factor)
    else:
        values[special] = float(fill_value)
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
        name=name,
        attrs={
            "long_name": metadata.product_name or "Composite reflectivity",
            "units": "dBZ",
            "product": metadata.product,
            "source_path": str(radar_path),
        },
    )
def _time_bound(value: object, *, end: bool) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    text = str(value)
    if end and len(text) == 8 and text.isdigit():
        return timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return timestamp


def discover_cinrad_files(
    paths: str | Path | Iterable[str | Path],
    *,
    pattern: str = "*.bin",
    time_range: tuple[object, object] | None = None,
    raw_grid: RadarGridSpec | None = None,
) -> list[CinradRecord]:
    """Discover, timestamp and sort radar files, rejecting duplicates."""

    candidates = [paths] if isinstance(paths, (str, Path)) else list(paths)
    files: list[Path] = []
    for value in candidates:
        path = Path(value).expanduser()
        if path.is_dir():
            files.extend(item for item in path.rglob(pattern) if item.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No radar files matched {pattern!r}")
    start = _time_bound(time_range[0], end=False) if time_range else None
    end = _time_bound(time_range[1], end=True) if time_range else None
    if start is not None and end is not None and start > end:
        raise ValueError("time_range start cannot be after end")
    seen: dict[pd.Timestamp, Path] = {}
    records = []
    for path in files:
        header = read_cinrad_header(path, raw_grid=raw_grid)
        timestamp = _filename_time(path) or header.timestamp
        if timestamp is None:
            raise ValueError(f"Cannot infer radar timestamp: {path}")
        if start is not None and timestamp < start:
            continue
        if end is not None and timestamp > end:
            continue
        if timestamp in seen:
            raise ValueError(
                f"Duplicate radar timestamp {timestamp}: {seen[timestamp]} and {path}"
            )
        seen[timestamp] = path
        records.append(CinradRecord(timestamp, path, header))
    records.sort(key=lambda record: record.time)
    if not records:
        raise ValueError("No radar files remain after time filtering")
    return records
