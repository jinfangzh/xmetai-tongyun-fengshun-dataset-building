"""Known-value v2 sources, including independently misaligned sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import zarr

if TYPE_CHECKING:
    from legacy_zarr_refresh import RefreshProfile


def write_legacy_store(path: Path, *, aligned: bool = False) -> np.ndarray:
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    values = np.arange(16, dtype="f4").reshape(2, 2, 2, 2)
    channels = ["ssrd", "q1000"]
    lat = [-1.0, 1.0]
    mean = [4.0, 2.0]
    std = [2.0, 0.5]
    mask = np.asarray([[0.0, 0.8], [1.0, 0.2]], dtype="f4")
    if aligned:
        values = values[:, [1, 0], ::-1, :]
        channels = ["q1000", "ssrd"]
        lat = [1.0, -1.0]
        mean = [2.0, 4.0]
        std = [0.5, 2.0]
        mask = mask[::-1]
    group.create_array("data", data=values, chunks=(1, 2, 2, 2))
    group.create_array("channel", data=np.asarray(channels, dtype=str), chunks=(2,))
    group.create_array("lat", data=np.asarray(lat), chunks=(2,))
    group.create_array("lon", data=np.asarray([0.0, 180.0]), chunks=(2,))
    time = group.create_array("time", data=np.asarray([0, 6], dtype="i8"), chunks=(2,))
    time.attrs["units"] = "hours since 2024-01-01 00:00:00"
    np.save(path / "mean.npy", np.asarray(mean, dtype="f4"))
    np.save(path / "std.npy", np.asarray(std, dtype="f4"))
    np.save(path / "mask.npy", mask)
    return values


def profile(source: Path) -> RefreshProfile:
    from legacy_zarr_refresh import RefreshProfile

    return RefreshProfile(
        key="synthetic",
        input=source,
        output_name="synthetic.refreshed.zarr",
        cadence_hours=6,
        radiation_seconds=21600,
        statistics_divide={"q1000": 2.0},
        statistics_multiply={"ssrd": 3.0},
        mask={"kind": "npy", "path": "mask.npy", "threshold": 0.5},
        expected_channels=2,
        expected_grid=(2, 2),
    )


def write_const(path: Path) -> None:
    with xr.Dataset(
        {
            "data": (
                ("channel", "lat", "lon"),
                np.asarray([[[1, 2], [3, 4]], [[0, 1], [1, 1]]], dtype="f4"),
            )
        },
        coords={
            "channel": np.asarray(["z", "lsm"]),
            "lat": np.asarray([-1.0, 1.0]),
            "lon": np.asarray([180.0, 0.0]),
        },
    ) as dataset:
        dataset.to_netcdf(path)


def write_auxiliaries(root: Path) -> None:
    root.mkdir()
    write_const(root / "const.nc")
    with xr.Dataset(
        {"weight": (("lat",), np.asarray([10.0, 20.0], dtype="f4"))},
        coords={"lat": np.asarray([-1.0, 1.0])},
    ) as dataset:
        dataset.to_netcdf(root / "weight.nc")
    with xr.Dataset(
        {"lsm": (("lat", "lon"), np.asarray([[0, 1], [1, 1]], dtype="u1"))},
        coords={
            "lat": np.asarray([-1.0, 1.0]),
            "lon": np.asarray([180.0, 0.0]),
        },
    ) as dataset:
        dataset.to_netcdf(root / "mask.nc")


def snapshot_files(path: Path) -> dict[str, bytes]:
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }
