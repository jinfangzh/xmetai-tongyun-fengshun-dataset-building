from __future__ import annotations

import argparse
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import xarray as xr

from datakit import io as dataset_io
from datakit.transforms import normalize_xarray
from pipelines.dataset import convert_to_zarr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# EXAMPLE_DATA_ROOT = Path("/workspace/data/Datakit_database/example/oras5_nc_to_zarr")
DEFAULT_SOURCE = Path("/workspace/data/Datakit_database/example/oras5_nc_to_zarr_88/nc")
DEFAULT_DESTINATION = Path("/workspace/data/Datakit_database/output/oras5_nc_to_zarr")
DEFAULT_MEAN = Path("/workspace/data/Datakit_database/example/oras5_nc_to_zarr_88/stats/mean.nc")
DEFAULT_STD = Path("/workspace/data/Datakit_database/example/oras5_nc_to_zarr_88/stats/std.nc")
DEFAULT_MASK = Path("/workspace/data/Datakit_database/example/oras5_nc_to_zarr_88/stats/mask.nc")


VARIABLES_LIST = [
    "so50", "so266", "so514", "so809", "so1177", "so1652", "so2275", "so3087", "so4118", "so5385", "so6902", "so8692", "so10803", "so13307", "so16316", "so19979", "so22114", "so27135", "so33386", "so41179", "so50863",
    "to50", "to266", "to514", "to809", "to1177", "to1652", "to2275", "to3087", "to4118", "to5385", "to6902", "to8692", "to10803", "to13307", "to16316", "to19979", "to22114", "to27135", "to33386", "to41179", "to50863",
    "uo50", "uo266", "uo514", "uo809", "uo1177", "uo1652", "uo2275", "uo3087", "uo4118", "uo5385", "uo6902", "uo8692", "uo10803", "uo13307", "uo16316", "uo19979", "uo22114", "uo27135", "uo33386", "uo41179", "uo50863",
    "vo50", "vo266", "vo514", "vo809", "vo1177", "vo1652", "vo2275", "vo3087", "vo4118", "vo5385", "vo6902", "vo8692", "vo10803", "vo13307", "vo16316", "vo19979", "vo22114", "vo27135", "vo33386", "vo41179", "vo50863",
    "sit", "ssh", "sic", "mlt",
]

assert len(VARIABLES_LIST) == 88


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORAS5 NetCDF -> normalized Zarr")
    parser.add_argument("--source", "--input-root", dest="source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", "--output-path", dest="destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--mean-nc", type=Path, default=DEFAULT_MEAN)
    parser.add_argument("--std-nc", type=Path, default=DEFAULT_STD)
    parser.add_argument("--mask-nc", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--channels", type=str, default="", help="Comma-separated channel names")
    parser.add_argument("--dataset-id", type=str, default="oras5.20250101.c88.v1")
    parser.add_argument("--schema-version", type=str, default="1.0")
    parser.add_argument("--content-version", type=str, default="v1")
    parser.add_argument("--data-revision", type=str, default="20261008")
    parser.add_argument("--strict", action="store_true", help="Raise error if any daily source file is missing")
    return parser.parse_args()


def _parse_cli_date(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    return date.fromisoformat(text) if "-" in text else date.fromisoformat(f"{text[:4]}-{text[4:6]}-{text[6:8]}")


def _generate_days(start: date, end: date) -> list[str]:
    days: list[str] = []
    current = start
    while current <= end:
        days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return days


def _resolve_folder(channel_name: str) -> str:
    prefix = channel_name[:2]
    if prefix in {"so", "to", "uo", "vo"}:
        return prefix
    return channel_name


def _build_nc_path(root: Path, day: str, folder: str) -> Path:
    return root / folder / day[:4] / f"{day}.nc"


def _extract_day_from_name(path: Path) -> date | None:
    stem = path.stem
    token = stem.split("_")[0]
    if len(token) == 8 and token.isdigit():
        return date.fromisoformat(f"{token[:4]}-{token[4:6]}-{token[6:8]}")
    return None


def _list_daily_files(input_root: Path) -> list[Path]:
    patterns = ["*_oras5.nc", "*.nc", "*/*.nc", "*/*_oras5.nc"]
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        for p in sorted(input_root.glob(pattern)):
            if p.is_file() and p not in seen:
                seen.add(p)
                files.append(p)
    return files


def _find_flat_daily_file(input_root: Path, day: str) -> Path | None:
    candidates = [
        input_root / f"{day}_oras5.nc",
        input_root / f"{day}.nc",
        input_root / day[:4] / f"{day}_oras5.nc",
        input_root / day[:4] / f"{day}.nc",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _infer_date_range(input_root: Path, channels: list[str]) -> tuple[date, date]:
    found: list[date] = []
    for file in _list_daily_files(input_root):
        parsed = _extract_day_from_name(file)
        if parsed is not None:
            found.append(parsed)

    if not found:
        if channels:
            folder = _resolve_folder(channels[0])
            base = input_root / folder
            raise FileNotFoundError(f"No daily NetCDF files found under: {input_root}; legacy folder check: {base}")
        raise FileNotFoundError(f"No daily NetCDF files found under: {input_root}")
    return found[0], found[-1]


def _infer_available_channels(input_root: Path, mean: xr.DataArray, std: xr.DataArray) -> list[str]:
    files = _list_daily_files(input_root)
    if not files:
        return VARIABLES_LIST

    opened = dataset_io.open(files[0]).read(load=True)
    try:
        ds = _as_dataset(opened)
        if "data" not in ds or "channel" not in ds["data"].dims:
            return VARIABLES_LIST
        data_channels = [str(item) for item in ds["data"].channel.values.tolist()]
    finally:
        if hasattr(opened, "close"):
            opened.close()

    mean_channels = {str(item) for item in mean.channel.values.tolist()}
    std_channels = {str(item) for item in std.channel.values.tolist()}
    matched = [name for name in data_channels if name in mean_channels and name in std_channels]
    return matched or VARIABLES_LIST


def _as_dataset(obj: xr.Dataset | xr.DataArray, variable: str = "data") -> xr.Dataset:
    if isinstance(obj, xr.DataArray):
        return obj.to_dataset(name=obj.name or variable)
    return obj


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


def _read_day_dataset(input_root: Path, day: str, channels: list[str], strict: bool) -> xr.Dataset | None:
    flat_file = _find_flat_daily_file(input_root, day)
    if flat_file is not None:
        opened = dataset_io.open(flat_file).read(load=True)
        try:
            ds = _as_dataset(opened)
            if "data" not in ds:
                raise KeyError(f"{flat_file} missing variable 'data'")
            da = ds["data"]
            if "channel" not in da.dims:
                raise ValueError(f"{flat_file} data must contain 'channel' dimension")
            missing = [name for name in channels if name not in da.channel.values.tolist()]
            if missing:
                raise KeyError(f"{flat_file} missing channel(s): {missing}")
            da = da.sel(channel=channels)
            if "time" in da.dims:
                if da.sizes["time"] == 0:
                    if strict:
                        raise ValueError(f"{flat_file} has empty 'time' dimension")
                    return None
            else:
                da = da.expand_dims(time=[pd.Timestamp(day)])

            return da.astype("float32").to_dataset(name="data")
        finally:
            if hasattr(opened, "close"):
                opened.close()

    opened_by_folder: dict[str, xr.Dataset] = {}
    arrays: list[xr.DataArray] = []

    try:
        for channel in channels:
            folder = _resolve_folder(channel)
            if folder not in opened_by_folder:
                nc_path = _build_nc_path(input_root, day, folder)
                if not nc_path.exists():
                    if strict:
                        raise FileNotFoundError(f"Missing source file: {nc_path}")
                    return None
                opened_by_folder[folder] = _as_dataset(dataset_io.open(nc_path).read(load=True))

            ds = opened_by_folder[folder]
            if "data" not in ds:
                raise KeyError(f"{folder}/{day}.nc missing variable 'data'")

            da = ds["data"]
            if "channel" in da.dims:
                if channel not in da.channel.values.tolist():
                    raise KeyError(f"{folder}/{day}.nc missing channel: {channel}")
                da = da.sel(channel=channel)

            if "time" in da.dims:
                if da.sizes["time"] == 0:
                    if strict:
                        raise ValueError(f"{folder}/{day}.nc has empty 'time' dimension")
                    return None
                da = da.isel(time=0)

            da = da.squeeze().astype("float32")
            da = da.expand_dims(dim={"channel": [channel]})
            arrays.append(da)

        daily = xr.concat(arrays, dim="channel")
        daily = daily.expand_dims(time=[pd.Timestamp(day)])
        return daily.to_dataset(name="data")
    finally:
        for ds in opened_by_folder.values():
            ds.close()


def main() -> None:
    args = get_args()

    input_root = args.source.expanduser()
    output_path = args.destination.expanduser()
    mean_nc = args.mean_nc.expanduser()
    std_nc = args.std_nc.expanduser()
    mask_nc = args.mask_nc.expanduser()

    start_date = _parse_cli_date(args.start_date)
    end_date = _parse_cli_date(args.end_date)

    requested_channels = [item.strip() for item in args.channels.split(",") if item.strip()]
    mean = _load_stat_array(mean_nc, "mean")
    std = _load_stat_array(std_nc, "std")
    mask = _load_any_array(mask_nc, "mask")

    channels = requested_channels if requested_channels else _infer_available_channels(input_root, mean, std)

    missing_mean = [name for name in channels if name not in mean.channel.values]
    missing_std = [name for name in channels if name not in std.channel.values]
    if missing_mean or missing_std:
        raise ValueError(
            "channel-stat mismatch: "
            f"missing_mean={missing_mean}, missing_std={missing_std}. "
            "Tip: pass --channels explicitly or regenerate stats with matching channel names."
        )

    if start_date is None or end_date is None:
        inferred_start, inferred_end = _infer_date_range(input_root, channels)
        if start_date is None:
            start_date = inferred_start
        if end_date is None:
            end_date = inferred_end

    if start_date > end_date:
        raise ValueError(f"start-date must be <= end-date, got {start_date} > {end_date}")

    days = _generate_days(start_date, end_date)
    print(f"计划处理 {len(days)} 天, channels={len(channels)}")

    datasets: list[xr.Dataset] = []
    for day in days:
        day_dataset = _read_day_dataset(input_root, day, channels, strict=args.strict)
        if day_dataset is not None:
            datasets.append(day_dataset)

    if not datasets:
        raise FileNotFoundError("指定日期范围内没有找到可处理的 ORAS5 NetCDF 数据")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = xr.concat(datasets, dim="time").sortby("time")
    dataset = dataset.sel(channel=channels)

    mean_sel = mean.sel(channel=dataset.channel)
    std_sel = std.sel(channel=dataset.channel)

    dataset["data"] = normalize_xarray(
        dataset["data"].astype("float32"),
        mean_sel,
        std_sel + 1e-12,
        channel_dim="channel",
        log_channels=(),
        log_clip=(0.0, 1000.0),
    ).astype("float16")

    print(dataset)
    print("开始写入 ORAS5 Zarr")

    convert_to_zarr(
        dataset,
        output_path,
        chunks={"time": 1},
        sidecars={"mean": mean_sel.astype("float32"), "std": std_sel.astype("float32"), "mask": mask},
        dataset_attrs={
            "dataset_id": args.dataset_id,
            "schema_version": args.schema_version,
            "content_version": args.content_version,
            "data_revision": args.data_revision,
        },
        add_geospatial_ranges=True,
    )

    print("写入完成:")
    print(output_path)

    opened = dataset_io.open(output_path).read()
    try:
        print(opened)
    finally:
        opened.close()
        for ds in datasets:
            ds.close()


if __name__ == "__main__":
    main()
