"""Convert daily ERA5 pressure-level NetCDF files to channel-form NetCDF."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from datakit.io import read_netcdf, write_netcdf
from datakit.transforms import resize
from datakit.schema import LEVELS_13
from datakit.transforms import (
    level_to_channel,
    normalize_coordinates,
    select_singleton,
)


VARIABLES = {
    "z": "geopotential_stream-oper_daily-mean",
    "t": "temperature_0_daily-mean",
    "u": "u_component_of_wind_0_daily-mean",
    "v": "v_component_of_wind_0_daily-mean",
    "q": "specific_humidity_0_daily-mean",
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process daily ERA5 pressure-level NetCDF files"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--resolution", type=float, default=1.5)
    parser.add_argument("--engine", default="netcdf4")
    return parser.parse_args()


def _target_grid(resolution: float) -> tuple[np.ndarray, np.ndarray]:
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    lat = np.arange(90.0, -90.0 - resolution / 2, -resolution)
    lon = np.arange(0.0, 360.0, resolution)
    return lat, lon


def _process_variable(
    source: Path,
    short_name: str,
    resolution: float,
    engine: str,
) -> xr.Dataset:
    opened = read_netcdf(source, engine=engine)
    dataset = (
        opened.to_dataset(name=opened.name or short_name)
        if isinstance(opened, xr.DataArray)
        else opened
    )

    if short_name not in dataset:
        raise KeyError(
            f"{short_name!r} not found in {source}; "
            f"available: {sorted(dataset.data_vars)}"
        )

    prepared = normalize_coordinates(dataset, add_level=False)
    array = prepared[short_name]
    array = select_singleton(array, "time")
    array = select_singleton(array, "member")

    if "level" not in array.dims:
        raise ValueError(f"{source} has no pressure-level dimension")

    target_levels = list(reversed(LEVELS_13))
    available = set(array.level.values.tolist())
    missing = [level for level in target_levels if level not in available]
    if missing:
        raise ValueError(f"{source} is missing pressure levels: {missing}")

    array = array.sel(level=target_levels)

    target_lat, target_lon = _target_grid(resolution)
    array = resize(
        array,
        lat=target_lat,
        lon=target_lon,
    ).astype("float32")

    channels = level_to_channel(array, short_name)["data"]
    expected = [f"{short_name}{level}" for level in target_levels]
    return channels.sel(channel=expected).to_dataset(name="data")


def main() -> None:
    args = get_args()

    if args.start_date > args.end_date:
        raise ValueError("start-date must not be later than end-date")

    current = args.start_date
    while current <= args.end_date:
        day = current.strftime("%Y%m%d")

        for short_name, filename in VARIABLES.items():
            source = args.input_root / day / f"{filename}.nc"
            output = (
                args.output_root
                / short_name
                / str(current.year)
                / f"{day}.nc"
            )

            processed = _process_variable(
                source,
                short_name,
                args.resolution,
                args.engine,
            )
            write_netcdf(processed, output, engine=args.engine)
            print(f"Saved: {output}")

        current += timedelta(days=1)


if __name__ == "__main__":
    main()