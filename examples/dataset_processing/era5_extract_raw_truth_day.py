"""Extract one raw ERA5 UTC day to NetCDF with optional channel subsets.

This example uses the reusable dataset pipeline API:
``pipelines.dataset.extract_raw_truth_day``.

It requires an existing ERA5 raw monthly archive laid out as:
``<root>/<group>/<variable>/<year>/*_<yyyym>.nc``.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from pipelines.dataset import extract_raw_truth_day
from pipelines.dataset_building.definitions.era5_c116 import DYNAMIC_CHANNELS


def parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def parse_channels(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("channels cannot be empty")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="ERA5 raw root directory")
    parser.add_argument("--date", type=parse_day, required=True, help="UTC day, e.g. 2025-01-01")
    parser.add_argument("--output", type=Path, required=True, help="output NetCDF path")
    parser.add_argument(
        "--channels",
        type=parse_channels,
        help="comma-separated channels, e.g. z500,t2m,tp,ws10m",
    )
    parser.add_argument(
        "--subset",
        choices=("demo-6", "core-70", "all-116"),
        default="demo-6",
        help="quick presets when --channels is omitted",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def subset_channels(name: str) -> list[str] | None:
    if name == "all-116":
        return None
    if name == "demo-6":
        return ["z500", "t850", "u850", "v850", "t2m", "tp"]
    if name == "core-70":
        return list(DYNAMIC_CHANNELS[:70])
    raise ValueError(f"Unsupported subset preset: {name}")


def main() -> None:
    args = parse_args()
    selected = args.channels if args.channels is not None else subset_channels(args.subset)
    report = extract_raw_truth_day(
        args.source,
        args.date,
        args.output,
        channels=selected,
        overwrite=args.overwrite,
    )
    print("status:", report["status"])
    print("output:", report["output"]["path"])
    print("dataset_id:", report["output"]["dataset_id"])
    print("channel_count:", report["output"]["channel_count"])


if __name__ == "__main__":
    main()
