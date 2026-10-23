from __future__ import annotations

import argparse
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from datakit import io as dataset_io

INPUT_ROOT = Path( "/home/zjf/xmetai-datakit/examples/dataset_downloader/"
    "copernicusmarine_download/copernicusmarine_data/oras5_daily"); 
OUTPUT_ROOT = Path("/home/zjf/xmetai-datakit-data/oras5/oras5_process"); 
START_DATE = date(1993, 1, 1); 
END_DATE = date(1993, 1, 1)
DEFAULT_GRID_RES = 0.25
NEW_LAT = np.arange(90, -90.1, -DEFAULT_GRID_RES); NEW_LON = np.arange(0, 360, DEFAULT_GRID_RES)
VARIABLES_2D = {"sit": "sithick_oras", "sic": "siconc_oras", "ssh": "zos_oras", "mlt": "mlotst_oras"}
VARIABLES_3D = {"so": "so_oras", "to": "thetao_oras", "uo": "uo_oras", "vo": "vo_oras"}
DEFAULT_LAYER_DEFINITIONS = [
    (0.5058, "50"), (1.5559, "155"), (2.6677, "266"), (3.8563, "385"),
    (5.1404, "514"), (6.5430, "654"), (8.0925, "809"), (9.8228, "982"),
    (11.7737, "1177"), (13.9910, "1399"), (16.5253, "1652"), (19.4298, "1942"),
    (22.7576, "2275"), (26.5583, "2655"), (30.8746, "3087"), (35.7402, "3574"),
    (41.1800, "4118"), (47.2119, "4721"), (53.8506, "5385"), (61.1128, "6111"),
    (69.0217, "6902"), (77.6112, "7761"), (86.9294, "8692"), (97.0413, "9704"),
    (108.0303, "10803"), (120.0000, "12000"), (133.0758, "13307"), (147.4062, "14740"),
    (163.1645, "16316"), (180.5499, "18054"), (199.7900, "19979"), (221.1412, "22114"),
    (244.8906, "24489"), (271.3564, "27135"), (300.8875, "30088"), (333.8628, "33386"),
    (370.6885, "37068"), (411.7939, "41179"), (457.6256, "45762"), (508.6399, "50863"),
]


def _build_target_layers(layer_defs: list[tuple[float, str]]) -> dict[float, tuple[str, int]]:
    return {depth: (suffix, idx) for idx, (depth, suffix) in enumerate(layer_defs)}


TARGET_LAYERS = _build_target_layers(DEFAULT_LAYER_DEFINITIONS)


def _parse_date(text: str) -> date:
    return date.fromisoformat(text) if "-" in text else date.fromisoformat(f"{text[:4]}-{text[4:6]}-{text[6:8]}")


def _build_target_grid(grid_res: float) -> tuple[np.ndarray, np.ndarray]:
    if grid_res <= 0:
        raise ValueError(f"grid-res must be positive, got {grid_res}")
    lat = np.arange(90.0, -90.0 - 1e-6, -grid_res)
    lon = np.arange(0.0, 360.0, grid_res)
    return lat, lon


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process ORAS5 daily NetCDF to unified channel NetCDF")
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--start-date", type=str, default=START_DATE.isoformat(), help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=END_DATE.isoformat(), help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--grid-res", type=float, default=DEFAULT_GRID_RES, help="Target lat/lon resolution in degrees, e.g. 0.25 or 1.5")
    return parser.parse_args()

def normalize_coords(data):
    if "latitude" in data.dims or "latitude" in data.coords:
        data = data.rename({"latitude": "lat"})
    if "longitude" in data.dims or "longitude" in data.coords:
        data = data.rename({"longitude": "lon"})
    if "lat" not in data.dims or "lon" not in data.dims:
        raise ValueError(f"Invalid coordinates: {data.dims}")
    data = data.assign_coords(lon=(data.lon.values + 360) % 360).sortby("lon")
    data = data.reindex(lat=NEW_LAT, lon=NEW_LON, method=None)
    if data.lat.values[0] < data.lat.values[-1]:
        data = data.isel(lat=slice(None, None, -1))
    return data.astype(np.float32)


def process_2d(ds, target, source):
    if source not in ds:
        raise KeyError(f"{source} not found")
    data = ds[source]
    if "time" in data.dims:
        if data.sizes["time"] != 1:
            raise ValueError(f"{source}: expected 1 time step")
        data = data.isel(time=0)
    data = normalize_coords(data.squeeze(drop=True)).rename("data")
    data.attrs.pop("valid_min", None); data.attrs.pop("valid_max", None)
    data = data.expand_dims(channel=[target])

    return data.transpose("channel", "lat", "lon")


def process_3d(ds, target, source):
    if source not in ds:
        raise KeyError(f"{source} not found")
    if "depth" not in ds:
        raise KeyError("depth not found")
    data_source = ds[source]
    if "time" in data_source.dims and data_source.sizes["time"] != 1:
        raise ValueError(f"{source}: expected 1 time step")
    depths = ds.depth.values
    result = []
    prefix = target
    for target_depth, (depth_suffix, depth_idx) in TARGET_LAYERS.items():
        if depth_idx >= len(depths):
            raise IndexError(f"{source}: depth index {depth_idx} out of range")
        actual_depth = float(depths[depth_idx])
        if abs(actual_depth - target_depth) > 0.1:
            warnings.warn(f"{source}: target={target_depth}, actual={actual_depth}, index={depth_idx}")
        data = data_source
        if "time" in data.dims:
            data = data.isel(time=0)
        data = normalize_coords(data.isel(depth=depth_idx).squeeze(drop=True)).rename("data")
        if "depth" in data.coords:
            data = data.drop_vars("depth")
        data.attrs.pop("valid_min", None); data.attrs.pop("valid_max", None)
        data = data.expand_dims(channel=[f"{prefix}{depth_suffix}"])
        data = data.transpose("channel", "lat", "lon")
        result.append(data)
    return result


def process_day(day):
    input_file = INPUT_ROOT / str(day.year) / f"{day:%Y%m%d}_oras5.nc"
    output_file = OUTPUT_ROOT / str(day.year) / f"{day:%Y%m%d}_oras5.nc"

    if not input_file.exists():
        print(f"[SKIP] input not found: {input_file}")
        return
    if output_file.exists():
        print(f"[SKIP] output exists: {output_file}")
        return

    print(f"[PROCESS] {day}")

    with dataset_io.open(input_file).read(engine="netcdf4") as ds:
        channels = []
        for target, source in VARIABLES_2D.items():
            print(f"  2D: {target} <- {source}")
            channels.append(process_2d(ds, target, source))
        for target, source in VARIABLES_3D.items():
            print(f"  3D: {target} <- {source}")
            channels.extend(process_3d(ds, target, source))

        output = xr.concat(channels, dim="channel").transpose("channel", "lat", "lon").to_dataset()
        output.attrs = {"dataset": "ORAS5", "date": day.isoformat(), "channel_count": len(channels), "processing": "0.25 degree grid, no normalization"}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    for name in output.data_vars:
        output[name].encoding = {}
    output.to_netcdf(output_file, engine="netcdf4")
    print(f"[DONE] {output_file} | channels={len(channels)}")


def main():
    global INPUT_ROOT, OUTPUT_ROOT, START_DATE, END_DATE, NEW_LAT, NEW_LON

    args = get_args()
    INPUT_ROOT = args.input_root.expanduser()
    OUTPUT_ROOT = args.output_root.expanduser()
    START_DATE = _parse_date(args.start_date)
    END_DATE = _parse_date(args.end_date)
    NEW_LAT, NEW_LON = _build_target_grid(float(args.grid_res))

    day = START_DATE
    while day <= END_DATE:
        try:
            process_day(day)
        except Exception as e:
            print(f"[ERROR] {day}: {type(e).__name__}: {e}")
        day += timedelta(days=1)


if __name__ == "__main__":
    main()