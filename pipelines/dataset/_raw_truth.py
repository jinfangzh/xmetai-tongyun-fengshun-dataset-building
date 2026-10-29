"""Private raw ERA5 day extraction pipeline implementation."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from datakit import io as dataset_io
from pipelines.dataset_building.production.era5_c116 import (
    DERIVED,
    DYNAMIC_CHANNELS,
    PRESSURE_VARIABLES as ERA5_PRESSURE_VARIABLES,
    GROUPS,
    SOURCE_DIRECTORIES,
    SOURCE_VARIABLES,
    STATIC,
    TARGET_LAT,
    TARGET_LON,
)
from datakit.transforms import RegularLatLonRegridder
from datakit.validation import (
    expected_six_hourly_times,
    validate_era5_layout,
    validate_six_hourly_day,
)


def _netcdf4_dataset_class() -> Any:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "netCDF4 is required for extract_raw_truth_day; install xmetai-datakit[io]"
        ) from exc
    return Dataset


def _source_spec(channel: str) -> tuple[str, str, str, int | None]:
    match = re.fullmatch(r"([a-z]+)(\d+)", channel)
    pressure_names = set(ERA5_PRESSURE_VARIABLES) | {"w"}
    if match and match.group(1) in pressure_names:
        logical = match.group(1)
        return "pl", logical, logical, int(match.group(2))
    logical = channel
    group = GROUPS[logical]
    directory = SOURCE_DIRECTORIES.get(logical, logical)
    return group, directory, logical, None


def _monthly_file(root: Path, group: str, directory: str, day: date) -> Path:
    base = root / group / directory / str(day.year)
    if not base.is_dir():
        raise FileNotFoundError(base)
    candidates = sorted(base.glob(f"*_{day.year}{day.month}.nc"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one {day:%Y-%m} file in {base}, found {len(candidates)}"
        )
    return candidates[0]


def _coordinate_name(dataset: xr.Dataset, names: tuple[str, ...]) -> str:
    found = [name for name in names if name in dataset.coords]
    if len(found) != 1:
        raise ValueError(f"expected one coordinate from {names}, found {found}")
    return found[0]


def _source_variable(dataset: xr.Dataset, logical: str) -> str:
    candidates = SOURCE_VARIABLES.get(logical, (logical,))
    found = [name for name in candidates if name in dataset.data_vars]
    if len(found) != 1:
        raise ValueError(
            f"expected one variable for {logical} from {candidates}, found {found}"
        )
    return found[0]


def _open_selected(path: Path, logical: str, day: date) -> xr.Dataset:
    with dataset_io.open(path).read(engine="netcdf4", cache=False) as original:
        time_name = _coordinate_name(original, ("valid_time", "time"))
        lat_name = _coordinate_name(original, ("latitude", "lat"))
        lon_name = _coordinate_name(original, ("longitude", "lon"))
        variable_name = _source_variable(original, logical)
        start = np.datetime64(day.isoformat(), "ns")
        stop = np.datetime64((day + timedelta(days=1)).isoformat(), "ns")
        all_times = np.asarray(original[time_name].values).astype("datetime64[ns]")
        indices = np.flatnonzero((all_times >= start) & (all_times < stop))
        if indices.size == 0:
            raise ValueError(f"{path} contains no data for {day.isoformat()}")
        rename = {time_name: "time", lat_name: "lat", lon_name: "lon"}
        if "pressure_level" in original.coords:
            rename["pressure_level"] = "level"
        if variable_name != logical:
            rename[variable_name] = logical
        selected = original.isel({time_name: slice(int(indices[0]), int(indices[-1]) + 1)}).rename(rename)
        loaded = selected.load()
    return loaded


def _resolve_channels(channels: Sequence[str] | None) -> tuple[str, ...]:
    if channels is None:
        selected = tuple(DYNAMIC_CHANNELS)
    else:
        selected = tuple(str(item) for item in channels)
    if not selected:
        raise ValueError("channels cannot be empty")
    if len(set(selected)) != len(selected):
        raise ValueError("channels must be unique")
    allowed = set(DYNAMIC_CHANNELS)
    unknown = [name for name in selected if name not in allowed]
    if unknown:
        raise ValueError(f"unsupported channels: {unknown}")
    return selected


def _create_output(
    path: Path,
    source_root: Path,
    day: date,
    channels: Sequence[str],
) -> tuple[list[float], dict[str, float]]:
    Dataset = _netcdf4_dataset_class()
    channel_names = list(channels)
    if len(set(channel_names)) != len(channel_names):
        raise RuntimeError("channels must be unique")

    times = expected_six_hourly_times(day)
    lat = np.asarray(TARGET_LAT, dtype="f4")
    lon = np.asarray(TARGET_LON, dtype="f4")
    regridder = RegularLatLonRegridder(target_lat=lat, target_lon=lon)

    samples = [np.nan] * len(channel_names)
    channel_units = [""] * len(channel_names)
    source_names = [""] * len(channel_names)
    static_samples: dict[str, float] = {}

    grouped: dict[tuple[str, str, str], list[tuple[int, str, int | None]]] = {}
    for index, channel in enumerate(channel_names):
        if channel in DERIVED:
            continue
        group, directory, logical, level = _source_spec(channel)
        grouped.setdefault((group, directory, logical), []).append((index, channel, level))

    with Dataset(path, "w", format="NETCDF4") as output:
        output.createDimension("time", 4)
        output.createDimension("channel", len(channel_names))
        output.createDimension("lat", 721)
        output.createDimension("lon", 1440)

        time_var = output.createVariable("time", "i8", ("time",))
        time_var.units = "hours since 1979-01-01 00:00:00"
        time_var.calendar = "proleptic_gregorian"
        time_var[:] = (times - np.datetime64("1979-01-01T00", "h")).astype("timedelta64[h]").astype("i8")

        channel_var = output.createVariable("channel", str, ("channel",))
        channel_var.long_name = "weather variable channel name"
        channel_var.description = "Variable names aligned with data[:, channel, :, :]"
        channel_var[:] = np.asarray(channel_names, dtype=object)

        index_var = output.createVariable("channel_index", "i2", ("channel",))
        index_var.long_name = "numeric channel index (legacy)"
        index_var.description = "Legacy positional index retained for compatibility"
        index_var[:] = np.arange(len(channel_names), dtype="i2")

        lat_var = output.createVariable("lat", "f4", ("lat",))
        lat_var.units = "degrees_north"
        lat_var.axis = "Y"
        lat_var[:] = lat

        lon_var = output.createVariable("lon", "f4", ("lon",))
        lon_var.units = "degrees_east"
        lon_var.axis = "X"
        lon_var[:] = lon

        data_var = output.createVariable(
            "data",
            "f4",
            ("time", "channel", "lat", "lon"),
            fill_value=np.float32(np.nan),
            chunksizes=(1, 1, 721, 1440),
            zlib=True,
            complevel=4,
            shuffle=True,
        )
        data_var.long_name = "raw source values before unit conversion and normalization"

        for (group, directory, logical), entries in grouped.items():
            path_in = _monthly_file(source_root, group, directory, day)
            dataset = _open_selected(path_in, logical, day)
            validate_six_hourly_day(dataset, day)
            slab = dataset[logical]
            unit = str(slab.attrs.get("units", "unknown"))
            for index, channel, level in entries:
                field = slab.sel(level=level) if level is not None else slab
                values = np.stack([regridder.apply(field, i) for i in range(4)])
                data_var[:, index, :, :] = values
                samples[index] = float(values[0, 0, 0])
                channel_units[index] = unit
                source_names[index] = logical
                print(f"[raw] {index + 1:03d}/{len(channel_names)} {channel}", flush=True)
                del values

        for channel, u_name, v_name in (("ws10m", "u10m", "v10m"), ("ws100m", "u100m", "v100m")):
            if channel not in channel_names:
                continue
            index = channel_names.index(channel)
            components = []
            for logical in (u_name, v_name):
                group, directory, _, _ = _source_spec(logical)
                path_in = _monthly_file(source_root, group, directory, day)
                dataset = _open_selected(path_in, logical, day)
                validate_six_hourly_day(dataset, day)
                field = dataset[logical]
                components.append(np.stack([regridder.apply(field, i) for i in range(4)]))
            values = np.hypot(components[0], components[1]).astype("f4")
            data_var[:, index, :, :] = values
            samples[index] = float(values[0, 0, 0])
            channel_units[index] = "m s**-1"
            source_names[index] = f"derived:hypot({u_name},{v_name})"
            print(f"[raw] {index + 1:03d}/{len(channel_names)} {channel}", flush=True)

        static_units: dict[str, str] = {}
        for logical in STATIC:
            group, directory, _, _ = _source_spec(logical)
            path_in = _monthly_file(source_root, group, directory, day)
            dataset = _open_selected(path_in, logical, day)
            validate_six_hourly_day(dataset, day)
            field = dataset[logical]
            values = regridder.apply(field, 0)
            attributes = dict(field.attrs)

            variable = output.createVariable(
                logical,
                "f4",
                ("lat", "lon"),
                fill_value=np.float32(np.nan),
                chunksizes=(721, 1440),
                zlib=True,
                complevel=4,
                shuffle=True,
            )
            for key in ("units", "long_name", "standard_name"):
                if key in attributes:
                    variable.setncattr(key, attributes[key])
            variable.source_file = str(path_in)
            variable[:, :] = values
            static_samples[logical] = float(values[0, 0])
            static_units[logical] = str(attributes.get("units", "unknown"))
            print(f"[static] {logical}", flush=True)

        output.setncatts(
            {
                "dataset_id": f"era5.{day:%Y%m%d}.c{len(channel_names)}.p25.h6.raw_truth",
                "processing_stage": "raw_truth_not_unit_converted_not_normalized",
                "source_root": str(source_root),
                "time_coverage_start": f"{day.isoformat()}T00:00:00Z",
                "time_coverage_end": f"{day.isoformat()}T18:00:00Z",
                "geospatial_lat_range": "[-90.0, 90.0]",
                "geospatial_lon_range": "[0.0, 359.75]",
                "grid_note": "Values are on the p25 grid with latitude stored north-to-south from 90 to -90.",
                "channel_names_json": json.dumps(channel_names, ensure_ascii=False),
                "channel_units_json": json.dumps(channel_units, ensure_ascii=False),
                "channel_source_names_json": json.dumps(source_names, ensure_ascii=False),
                "static_units_json": json.dumps(static_units, ensure_ascii=False),
                "zarr_preprocessing_json": json.dumps(
                    {
                        "q*": "multiply raw kg/kg by 1000 before normalization",
                        "ssr,ssrd,fdir,ttr": "divide raw accumulated energy by 21600 before normalization",
                        "tp": "multiply raw metres by 1000, then log1p(max(mm, 0)); no z-score",
                        "ws10m": "hypot(raw u10m, raw v10m)",
                        "ws100m": "hypot(raw u100m, raw v100m)",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return samples, static_samples


def _verify(
    path: Path,
    day: date,
    channels: Sequence[str],
    samples: list[float],
    static_samples: dict[str, float],
) -> None:
    Dataset = _netcdf4_dataset_class()
    with Dataset(path, "r") as dataset:
        channels = [str(value) for value in dataset.variables["channel"][:].tolist()]
        data = dataset.variables["data"]
        lat = np.asarray(dataset.variables["lat"][:], dtype="f4")
        lon = np.asarray(dataset.variables["lon"][:], dtype="f4")

        validate_era5_layout(
            channels=channels,
            data_shape=tuple(data.shape),
            lat=lat,
            lon=lon,
            expected_channels=channels,
            expected_lat=TARGET_LAT,
            expected_lon=TARGET_LON,
            expected_time_steps=4,
        )

        for index, expected in enumerate(samples):
            actual = float(data[0, index, 0, 0])
            if not np.isclose(actual, expected, rtol=0, atol=0, equal_nan=True):
                raise RuntimeError(f"read-back failed for {channels[index]}")

        for name, expected in static_samples.items():
            actual = float(dataset.variables[name][0, 0])
            if not np.isclose(actual, expected, rtol=0, atol=0, equal_nan=True):
                raise RuntimeError(f"read-back failed for static field {name}")

        units = json.loads(dataset.getncattr("channel_units_json"))
        if units[channels.index("tp")].strip() != "m":
            raise RuntimeError(f"raw tp units must be m, got {units[channels.index('tp')]!r}")
        expected_id = f"era5.{day:%Y%m%d}.c{len(channels)}.p25.h6.raw_truth"
        if dataset.getncattr("dataset_id") != expected_id:
            raise RuntimeError("dataset_id verification failed")


def extract_raw_truth_day(
    source_root: str | Path,
    day: date,
    output_path: str | Path,
    *,
    channels: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Extract one UTC day of raw ERA5 into one channel-oriented NetCDF truth file."""

    source = Path(source_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    selected_channels = _resolve_channels(channels)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; use overwrite=True: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    print(f"source: {source}")
    print(f"date:   {day.isoformat()}")
    print(f"output: {output}")

    try:
        samples, static_samples = _create_output(temporary, source, day, selected_channels)
        _verify(temporary, day, selected_channels, samples, static_samples)
        os.replace(temporary, output)
        _verify(output, day, selected_channels, samples, static_samples)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"[DONE] {output}")
    return {
        "status": "passed",
        "output": {
            "path": str(output),
            "dataset_id": f"era5.{day:%Y%m%d}.c{len(selected_channels)}.p25.h6.raw_truth",
            "channel_count": len(selected_channels),
            "channels": list(selected_channels),
            "static_fields": list(STATIC),
        },
        "input": {
            "source_root": str(source),
            "day": day.isoformat(),
        },
    }
