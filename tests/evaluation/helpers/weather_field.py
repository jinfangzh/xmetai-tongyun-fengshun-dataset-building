"""Synthetic weather-field inputs for tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr

# Small global grid used by the default synthetic end-to-end test.  The grid is
# deliberately just large enough for spherical bands up to degree 3.
SYNTHETIC_FIELD_LAT = np.linspace(-90.0, 90.0, 9, dtype="f8")
SYNTHETIC_FIELD_LON = np.arange(-180.0, 180.0, 22.5, dtype="f8")
SYNTHETIC_FIELD_LEADS = (6, 12)


def _synthetic_fields(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return smooth z500/t2m fields on an arbitrary longitude convention."""
    lat2d = np.deg2rad(np.asarray(lat, dtype="f8"))[:, None]
    lon2d = np.deg2rad(np.mod(np.asarray(lon, dtype="f8"), 360.0))[None, :]
    z500 = 50_000.0 + 120.0 * np.sin(lat2d) + 30.0 * np.cos(2.0 * lon2d)
    t2m = 275.0 + 8.0 * np.cos(lat2d) + 2.0 * np.sin(lon2d)
    return z500, t2m


def write_synthetic_fuxi_phys(root, init_time: datetime) -> Path:
    """Write two native FuXi-phys files (6 h and 12 h) for the chain test."""
    root = Path(root)
    date_dir = root / init_time.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    # Forecast latitude is descending and longitude uses 0..360.  ERA5 below
    # uses the opposite latitude order and -180..180, so the production matcher
    # must perform the same normalization needed by real inputs.
    lat = SYNTHETIC_FIELD_LAT[::-1]
    lon = np.mod(SYNTHETIC_FIELD_LON, 360.0)
    order = np.argsort(lon)
    lon = lon[order]
    z500, t2m = _synthetic_fields(lat, lon)
    for index, (z_offset, t_offset) in enumerate(((2.0, 1.0), (4.0, 2.0)), start=1):
        dataset = xr.Dataset(
            {
                "Z500": (("lat", "lon"), z500 + z_offset),
                "T2M": (("lat", "lon"), t2m + t_offset),
            },
            coords={"lat": lat, "lon": lon},
        )
        dataset["Z500"].attrs["units"] = "m^2/s^2"
        dataset["T2M"].attrs["units"] = "K"
        dataset.to_netcdf(date_dir / f"{index:03d}.nc")
    return root


def write_synthetic_fuxi_ens_phys(
    root, init_time: datetime, member_offsets: Mapping[str, float]
) -> Path:
    """Write ensemble FuXi-phys files: ``root/YYYYMMDD/member_XXX/001.nc…``.

    ``member_offsets`` maps member dir names (``member_000`` …) to a constant
    z500 offset.  Both leads carry the same offset; the observation side varies
    per lead instead (see ``write_synthetic_era5_zarr``), so the two time slices
    stay distinguishable and a lead mis-pairing is caught by the RMSE values.
    """
    root = Path(root)
    date_dir = root / init_time.strftime("%Y%m%d")
    lat = SYNTHETIC_FIELD_LAT[::-1]
    lon = np.mod(SYNTHETIC_FIELD_LON, 360.0)
    order = np.argsort(lon)
    lon = lon[order]
    z500, _ = _synthetic_fields(lat, lon)
    for member, offset in member_offsets.items():
        member_dir = date_dir / member
        member_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, len(SYNTHETIC_FIELD_LEADS) + 1):
            dataset = xr.Dataset(
                {"Z500": (("lat", "lon"), z500 + float(offset))},
                coords={"lat": lat, "lon": lon},
            )
            dataset["Z500"].attrs["units"] = "m^2/s^2"
            dataset.to_netcdf(member_dir / f"{index:03d}.nc")
    return root


def _write_zarr_store(path: Path, channel: str, values: np.ndarray) -> Path:
    """Write the minimal native store shape consumed by Era5ZarrSource.

    Zarr 2 exposes ``Group.create_dataset`` while Zarr 3 uses
    ``Group.create_array``.  The production reader supports both generations,
    so this test fixture must not force contributors onto only one of them.
    """
    import zarr

    path = Path(path)
    group = zarr.open_group(str(path), mode="w")

    def create(name, *, data, **kwargs):
        method = getattr(group, "create_array", None)
        if method is None:
            method = group.create_dataset
        return method(name, data=data, **kwargs)

    create(
        "data",
        data=np.asarray(values, dtype="f4")[:, None, :, :],
        chunks=(1, 1, values.shape[-2], values.shape[-1]),
        overwrite=True,
    )
    time = create("time", data=np.asarray(SYNTHETIC_FIELD_LEADS, dtype="f8"), overwrite=True)
    time.attrs["units"] = "hours since 2025-01-02 00:00:00"
    create("lat", data=SYNTHETIC_FIELD_LAT, overwrite=True)
    create("lon", data=SYNTHETIC_FIELD_LON, overwrite=True)
    (path / "channel_names.json").write_text(json.dumps([channel]), encoding="utf-8")
    return path


def write_synthetic_era5_zarr(root) -> Mapping[str, Path]:
    """Write separate pressure-level and surface stores for two valid times.

    The 12 h slice carries a +10 offset so the two valid times differ: if the
    reader or matcher mis-pairs a lead to the wrong time slice, the 12 h RMSE
    comes out wrong (6 / 8 instead of 2 / 1) and the test fails instead of
    silently passing on two identical fields.
    """
    root = Path(root)
    z500, t2m = _synthetic_fields(SYNTHETIC_FIELD_LAT, SYNTHETIC_FIELD_LON)
    z500_values = np.stack([z500, z500 + 10.0], axis=0)
    t2m_values = np.stack([t2m, t2m + 10.0], axis=0)
    return {
        "pl": _write_zarr_store(root / "era5_pl.zarr", "z_500", z500_values),
        "sfc": _write_zarr_store(root / "era5_sfc.zarr", "t2m", t2m_values),
    }


def write_synthetic_fuxi_phys_multi(root, init_times) -> Path:
    """Write deterministic forecast for several init days (for the execution test)."""
    root = Path(root)
    for init in init_times:
        write_synthetic_fuxi_phys(root, init)
    return root


def write_synthetic_era5_zarr_multi(root, init_times) -> Mapping[str, Path]:
    """Write a multi-day ERA5 zarr: one slice per (init, lead) valid time.

    Same field convention as ``write_synthetic_era5_zarr`` (the 12 h slice carries
    the +10 offset), just spread over several init days so the execution layer has
    multiple chunks to run.
    """
    root = Path(root)
    z500, t2m = _synthetic_fields(SYNTHETIC_FIELD_LAT, SYNTHETIC_FIELD_LON)
    times, z500_slices, t2m_slices = [], [], []
    for init in init_times:
        for lead in SYNTHETIC_FIELD_LEADS:
            times.append((init - datetime(2025, 1, 1)).total_seconds() / 3600.0 + lead)
            offset = 10.0 if lead == 12 else 0.0
            z500_slices.append(z500 + offset)
            t2m_slices.append(t2m + offset)
    return {
        "pl": _write_zarr_store_multi(root / "era5_pl.zarr", "z_500", np.stack(z500_slices), times),
        "sfc": _write_zarr_store_multi(root / "era5_sfc.zarr", "t2m", np.stack(t2m_slices), times),
    }


def _write_zarr_store_multi(path, channel, values, times) -> Path:
    """Like ``_write_zarr_store`` but with an explicit multi-day time axis."""
    import zarr

    path = Path(path)
    group = zarr.open_group(str(path), mode="w")

    def create(name, *, data, **kwargs):
        method = getattr(group, "create_array", None)
        if method is None:
            method = group.create_dataset
        return method(name, data=data, **kwargs)

    create(
        "data",
        data=np.asarray(values, dtype="f4")[:, None, :, :],
        chunks=(1, 1, values.shape[-2], values.shape[-1]),
        overwrite=True,
    )
    time = create("time", data=np.asarray(times, dtype="f8"), overwrite=True)
    time.attrs["units"] = "hours since 2025-01-01 00:00:00"
    create("lat", data=SYNTHETIC_FIELD_LAT, overwrite=True)
    create("lon", data=SYNTHETIC_FIELD_LON, overwrite=True)
    (path / "channel_names.json").write_text(json.dumps([channel]), encoding="utf-8")
    return path


def write_synthetic_daily_climatology(path) -> Path:
    """Write a tiny 365-day, 6-hourly climatology used with 15-day smoothing."""
    path = Path(path)
    times = np.arange(
        np.datetime64("1990-01-01T00:00:00"),
        np.datetime64("1991-01-01T00:00:00"),
        np.timedelta64(6, "h"),
    )
    shape = (times.size, SYNTHETIC_FIELD_LAT.size, SYNTHETIC_FIELD_LON.size)
    dataset = xr.Dataset(
        {
            "z500": (("time", "lat", "lon"), np.zeros(shape, dtype="f4")),
            "t2m": (("time", "lat", "lon"), np.zeros(shape, dtype="f4")),
        },
        coords={
            "time": times,
            "lat": SYNTHETIC_FIELD_LAT,
            "lon": SYNTHETIC_FIELD_LON,
        },
    )
    dataset.to_netcdf(path)
    return path
