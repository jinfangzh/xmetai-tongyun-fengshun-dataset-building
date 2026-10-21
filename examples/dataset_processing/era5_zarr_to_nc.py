"""Export a Zarr time slice to NetCDF using datakit public APIs.

Workflow:
1. extract_zarr_to_netcdf: select time range + channels, denormalize with mean/std.
2. Optional split: write one daily NetCDF per channel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.dataset import (
    extract_zarr_to_netcdf,
    split_netcdf_by_time_and_channel,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zarr -> NetCDF via datakit")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        required=True,
        help="Input Zarr path",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help=(
            "Optional directory containing mean/std (nc, npy, or zarr root arrays). "
            "When omitted, statistics are read from the input Zarr dataset first."
        ),
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        required=True,
        help="Output NetCDF file path",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date, e.g. 1950-01-01",
    )
    parser.add_argument(
        "--time-interval",
        type=int,
        default=1,
        help="Number of days to export (half-open interval)",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="",
        help="Comma-separated channels to export",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="data",
        help="Variable name in Zarr dataset",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="netcdf4",
        choices=("netcdf4", "h5netcdf", "scipy"),
        help="NetCDF engine",
    )
    parser.add_argument(
        "--split-daily-by-channel",
        action="store_true",
        help="Additionally split extracted NetCDF to <out_root>/<channel>/<year>/<yyyymmdd>.nc",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="Root directory for split files; required when --split-daily-by-channel is set",
    )
    return parser.parse_args()


def _parse_channels(raw: str) -> list[str] | None:
    channels = [item.strip() for item in raw.split(",") if item.strip()]
    return channels or None


def main() -> None:
    args = get_args()

    channels = _parse_channels(args.channels)

    output = extract_zarr_to_netcdf(
        zarr_path=args.zarr_path,
        save_path=args.save_path,
        start_date=args.start_date,
        time_interval=args.time_interval,
        channels=channels,
        variable=args.variable,
        stats_path=args.stats_path,
        engine=args.engine,
    )
    print(f"Saved merged NetCDF: {output}")

    if args.split_daily_by_channel:
        if args.split_root is None:
            raise ValueError("--split-root is required when --split-daily-by-channel is enabled")
        split_paths = split_netcdf_by_time_and_channel(
            Path(output),
            args.split_root,
            engine=args.engine,
        )
        for path in split_paths:
            print(f"Saved split: {path}")


if __name__ == "__main__":
    main()
