"""Convert daily channel NetCDF files with unit preprocessing only.

This example keeps the workflow at the datakit API boundary:
1) read daily NetCDF files;
2) apply preprocess_channel_units for channel rename + q/radiation/tp unit handling;
3) optionally select output channels;
4) write converted daily NetCDF files.

No normalization is applied in this stage.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import xarray as xr

from datakit import io as dataset_io
from datakit.transforms import (
    preprocess_channel_units,
)


def _parse_day_from_stem(path: Path) -> date:
    stem = path.stem.strip()
    if "-" in stem:
        return date.fromisoformat(stem)
    return datetime.strptime(stem, "%Y%m%d").date()


def _parse_cli_day(text: str | None) -> date | None:
    if text is None:
        return None
    value = text.strip()
    if not value:
        return None
    if "-" in value:
        return date.fromisoformat(value)
    return datetime.strptime(value, "%Y%m%d").date()


def _in_range(path: Path, start: date | None, end: date | None) -> bool:
    current = _parse_day_from_stem(path)
    if start is not None and current < start:
        return False
    if end is not None and current > end:
        return False
    return True


def _select_data_array(opened: xr.Dataset | xr.DataArray, variable: str) -> xr.DataArray:
    if isinstance(opened, xr.DataArray):
        return opened
    if variable in opened.data_vars:
        return opened[variable]
    if len(opened.data_vars) == 1:
        return opened[next(iter(opened.data_vars))]
    raise KeyError(f"Variable {variable!r} not found; available: {sorted(opened.data_vars)}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily NetCDF unit preprocessing example")
    parser.add_argument("--source", type=Path, required=True, help="Directory of daily *.nc files")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write converted *.nc files")
    parser.add_argument("--variable", type=str, default="data", help="Variable name to process")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument(
        "--levels",
        type=str,
        default="",
        help="Optional comma-separated channels to keep and order",
    )
    parser.add_argument(
        "--accumulation-hours",
        type=float,
        default=6.0,
        help="Accumulation window hours used for radiation unit conversion.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    source = args.source.expanduser()
    output = args.output.expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    start_day = _parse_cli_day(args.start_date)
    end_day = _parse_cli_day(args.end_date)
    if start_day is not None and end_day is not None and start_day > end_day:
        raise ValueError("start-date cannot be greater than end-date")

    levels = [item.strip() for item in args.levels.split(",") if item.strip()]
    files = sorted(path for path in source.glob("*.nc") if _in_range(path, start_day, end_day))
    if not files:
        raise FileNotFoundError(f"No daily NetCDF files found in {source}")

    output.mkdir(parents=True, exist_ok=True)
    print(f"发现 {len(files)} 个输入文件")

    converted = 0
    for index, path in enumerate(files, start=1):
        out_name = f"{_parse_day_from_stem(path):%Y%m%d}.unit_converted.nc"
        destination = output / out_name
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {destination}")

        opened = dataset_io.open(path).read(load=True)
        try:
            data = _select_data_array(opened, args.variable).astype("float32")
            data = preprocess_channel_units(
                data,
                channel_dim="channel",
                accumulation_hours=float(args.accumulation_hours),
            ).astype("float32")

            if levels:
                available = [str(name) for name in data.channel.values.tolist()]
                missing = [name for name in levels if name not in available]
                if missing:
                    raise KeyError(f"{path} missing requested levels: {missing}")
                data = data.sel(channel=levels)

            ds = data.to_dataset(name=args.variable)
            encoding = {
                args.variable: {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32"}
            }
            dataset_io.open(destination).write(ds, mode="w", engine="netcdf4", format="NETCDF4", encoding=encoding)[0]
            converted += 1
            print(f"[{index:03d}/{len(files):03d}] {path.name} -> {destination.name}")
        finally:
            if hasattr(opened, "close"):
                opened.close()

    print("=" * 60)
    print(f"转换完成: {converted} 个文件")
    print(f"输出目录: {output}")
    print(f"accumulation_hours: {float(args.accumulation_hours)}")


if __name__ == "__main__":
    main()
