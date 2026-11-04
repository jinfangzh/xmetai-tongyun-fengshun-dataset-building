from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

import xarray as xr

from datakit import io as dataset_io
from datakit.transforms import normalize_xarray, preprocess_channel_units
from pipelines.dataset import convert_to_zarr

EXAMPLE_DATA_ROOT = Path("/workspace/data/Datakit_database/example/era5_nc_to_zarr")
DEFAULT_SOURCE = EXAMPLE_DATA_ROOT / "nc"
DEFAULT_DESTINATION = Path("/workspace/data/Datakit_database/output/era5_nc_to_zarr/output.zarr")
DEFAULT_MEAN = EXAMPLE_DATA_ROOT / "stats/mean.nc"
DEFAULT_STD = EXAMPLE_DATA_ROOT / "stats/std.nc"
DEFAULT_MASK = EXAMPLE_DATA_ROOT / "stats/mask.nc"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="6-hourly NetCDF -> normalized Zarr")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--mean-nc", type=Path, default=DEFAULT_MEAN)
    parser.add_argument("--std-nc", type=Path, default=DEFAULT_STD)
    parser.add_argument("--mask-nc", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--accumulation-hours", type=float, default=6.0)
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--channels", type=str, default="", help="Comma-separated channel names")
    parser.add_argument("--dataset-id", type=str, default="era5.20250101.c116.p25.h6.v1")
    parser.add_argument("--schema-version", type=str, default="2.0")
    parser.add_argument("--content-version", type=str, default="v1")
    parser.add_argument("--data-revision", type=str, default="20260929")
    return parser.parse_args()


def _parse_cli_date(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    return date.fromisoformat(text) if "-" in text else date.fromisoformat(f"{text[:4]}-{text[4:6]}-{text[6:8]}")


def _load_stat_array(path: Path, variable: str) -> xr.DataArray:
    opened = dataset_io.open(path).read(load=True)
    try:
        value = opened if isinstance(opened, xr.DataArray) else opened[variable]
        if "channel" not in value.dims:
            raise ValueError(f"{path} must contain a 'channel' dimension")
        return value.load().astype("float32")
    finally:
        if hasattr(opened, "close"):
            opened.close()


def _load_any_array(path: Path, variable: str) -> xr.DataArray:
    opened = dataset_io.open(path).read(load=True)
    try:
        return (opened if isinstance(opened, xr.DataArray) else opened[variable]).load()
    finally:
        if hasattr(opened, "close"):
            opened.close()


def main() -> None:
    args = get_args()
    source, destination = args.source.expanduser(), args.destination.expanduser()
    mean_nc, std_nc, mask_nc = args.mean_nc.expanduser(), args.std_nc.expanduser(), args.mask_nc.expanduser()
    start_day, end_day = _parse_cli_date(args.start_date), _parse_cli_date(args.end_date)
    accumulation_hours = float(args.accumulation_hours)
    nc_files = sorted(source.glob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(f"没有找到 NetCDF 文件: {source}")

    print("=" * 60)
    print(f"发现 {len(nc_files)} 个 NetCDF 文件")
    for file in nc_files[:5]:
        print(file)
    print("=" * 60)

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    mean, std, mask = _load_stat_array(mean_nc, "mean"), _load_stat_array(std_nc, "std"), _load_any_array(mask_nc, "mask")

    print("\n开始读取与预处理  NetCDF")
    datasets = []
    requested_channels = [item.strip() for item in args.channels.split(",") if item.strip()]

    for file in nc_files:
        ds = dataset_io.open(file).read(load=True)

        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset(name=ds.name or "data")
        if "data" not in ds:
            ds.close()
            raise KeyError(f"{file} 中未找到变量 'data'")
        if "time" not in ds.dims:
            ds.close()
            raise ValueError(f"{file} 必须包含 'time' dimension")

        if start_day is not None:
            ds = ds.sel(time=ds.time.dt.date >= start_day)
        if end_day is not None:
            ds = ds.sel(time=ds.time.dt.date <= end_day)
        if ds.sizes["time"] == 0:
            ds.close()
            continue

        print(f"处理: {file} -> {ds.sizes['time']} 个 time steps")

        da = preprocess_channel_units(ds["data"].astype("float32"), channel_dim="channel", accumulation_hours=accumulation_hours).astype("float32")

        if requested_channels:
            missing = [name for name in requested_channels if name not in da.channel.values.tolist()]
            if missing:
                ds.close()
                raise KeyError(f"{file} 缺少指定 channels: {missing}")
            da = da.sel(channel=requested_channels)

        missing_mean = [name for name in da.channel.values.tolist() if name not in mean.channel.values]
        missing_std = [name for name in da.channel.values.tolist() if name not in std.channel.values]
        if missing_mean or missing_std:
            ds.close()
            raise ValueError(f"{file} 的 channel 与统计量不匹配: missing_mean={missing_mean}, missing_std={missing_std}")

        mean_sel, std_sel = mean.sel(channel=da.channel), std.sel(channel=da.channel)
        da = normalize_xarray(da, mean_sel, std_sel + 1e-12, channel_dim="channel", log_channels=(), log_clip=(0.0, 1000.0)).astype("float32")
        datasets.append(da.to_dataset(name="data"))
        ds.close()

    if not datasets:
        raise FileNotFoundError("指定日期范围内没有找到 NetCDF 数据")

    print("\n开始合并 NetCDF")
    dataset = xr.concat(datasets, dim="time").sortby("time")
    dataset["data"] = dataset["data"].astype("float16")
    mean, std = mean.sel(channel=dataset.channel), std.sel(channel=dataset.channel)

    print(dataset)
    print("\n开始转换 NetCDF -> Zarr")

    convert_to_zarr(
        dataset,
        destination,
        chunks={"time": 1},
        sidecars={"mean": mean, "std": std, "mask": mask},
        dataset_attrs={"dataset_id": args.dataset_id, "schema_version": args.schema_version, "content_version": args.content_version, "data_revision": args.data_revision},
        add_geospatial_ranges=True,
    )

    print("\n转换完成:")
    print(destination)
    print("\n读取 Zarr")

    opened = dataset_io.open(destination).read()
    try:
        print(opened)
    finally:
        opened.close()
        for ds in datasets:
            ds.close()


if __name__ == "__main__":
    main()