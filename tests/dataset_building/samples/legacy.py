"""Known-value v2 sources, including independently misaligned sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import yaml
import zarr

from datakit.schema.datasets import load_schema

if TYPE_CHECKING:
    from legacy_zarr_refresh import RefreshProfile


def write_legacy_store(path: Path, *, aligned: bool = False, zarr_format: int = 2) -> np.ndarray:
    group = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
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
    group.create_array("lon", data=np.asarray([0.0, 2.0]), chunks=(2,))
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
        target_schema=Path(__file__).with_name("legacy.yaml"),
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
            "lon": np.asarray([2.0, 0.0]),
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
            "lon": np.asarray([2.0, 0.0]),
        },
    ) as dataset:
        dataset.to_netcdf(root / "mask.nc")


def snapshot_files(path: Path) -> dict[str, bytes]:
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }


def write_product_sample(root: Path, standard: Path):
    """Keep real product semantics in an explicitly miniature test declaration.

    This is a production-layout regression, not evidence of production coverage.
    Values and source permutations are independent of the refresh implementation.
    """
    from legacy_zarr_refresh import RefreshProfile

    document = yaml.safe_load(standard.read_text(encoding="utf-8"))
    meta = document["meta_info"]
    resolution = meta["grid"]["space_resolution"]
    ascending = meta["grid"]["latitude"]["direction"] == "south-to-north"
    latitude = 90 - np.arange(3, dtype="f4") * resolution
    if ascending:
        latitude = latitude[::-1].copy()
    longitude = np.arange(4, dtype="f4") * resolution
    cadence = meta["time_coverage"]["time_resolution"]["value"]
    times = np.datetime64("2024-01-01T00", "ns") + np.arange(3) * np.timedelta64(cadence, "h")
    output = root / "output" / "miniature.zarr"
    meta.update(
        name="Miniature product-layout regression",
        dataset_id="miniature." + meta["dataset_id"],
        dataset_path=str(output),
        auxiliary_directory=str(output),
        identity_mapping_evidence="Independent miniature fixture directory mapping",
    )
    meta["time_coverage"].update(first=str(times[0]), last=str(times[-1]), time_count=3)
    meta["grid"]["latitude"].update(first=float(latitude[0]), last=float(latitude[-1]), count=3)
    meta["grid"]["longitude"].update(first=0.0, last=float(longitude[-1]), count=4)
    for item in meta["auxiliaries"]["files"].values():
        if item["format"] == "npy":
            item["grid_provenance"] = "Independently generated miniature fixture axes"
    schema_path = root / "miniature.yaml"
    schema_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    schema, _ = load_schema(schema_path)
    names = tuple(meta["channels"])
    count = len(names)
    # Distinct channel/spatial signatures expose coordinated data/statistic mistakes.
    values = (
        np.arange(3, dtype="f4")[:, None, None, None] / 4
        + np.arange(count, dtype="f4")[None, :, None, None] / 8
        + np.arange(12, dtype="f4").reshape(1, 1, 3, 4) / 32
    ).astype("f2")
    mean = np.arange(count, dtype="f4") + 10
    std = np.arange(count, dtype="f4") / 8 + 1
    for channel in schema.product.channels:
        if channel.variable in schema.product.effective_statistics:
            index = names.index(channel.name)
            mean[index], std[index] = schema.product.effective_statistics[channel.variable]
    land = (np.arange(12).reshape(3, 4) % 3 == 0).astype("u1")
    permutation = np.arange(count - 1, -1, -1)
    source = root / "legacy.zarr"
    group = zarr.open_group(str(source), mode="w", zarr_format=2)
    group.create_array(
        "data", data=values[:, permutation, ::-1, :].astype("f4"), chunks=(1, count, 3, 4)
    )
    group.create_array("level", data=np.asarray(names)[permutation])
    group.create_array("lat", data=latitude[::-1])
    group.create_array("lon", data=longitude)
    time = group.create_array("time", data=np.arange(3, dtype="i8") * cadence)
    time.attrs["units"] = "hours since 2024-01-01 00:00:00"
    np.save(source / "mean.npy", mean[permutation])
    np.save(source / "std.npy", std[permutation])
    np.save(source / "land.npy", land[::-1])
    for auxiliary in schema.variant.auxiliaries:
        if auxiliary.role == "const":
            constants = np.ones((len(auxiliary.channels), 3, 4), dtype="f4")
            if "lsm" in auxiliary.channels:
                constants[auxiliary.channels.index("lsm")] = land
            with xr.Dataset(
                {"legacy_constants": (("channel", "lat", "lon"), constants[::-1, ::-1])},
                coords={
                    "channel": np.array(auxiliary.channels)[::-1],
                    "lat": latitude[::-1],
                    "lon": longitude,
                },
            ) as dataset:
                dataset.to_netcdf(source / "const.nc")
        elif auxiliary.role == "weight":
            if auxiliary.format == "netcdf":
                np.save(source / "weight.npy", np.array([3, 2, 1], dtype="f4").reshape(1, 3, 1))
                continue
            # Exercise both NC-to-NPY and NPY-to-NC, retaining the actual grid.
            with xr.Dataset(
                {"latitude_weight": (("lat",), np.array([1.0, 2.0, 3.0], dtype="f4")[::-1])},
                coords={"lat": latitude[::-1]},
            ) as dataset:
                dataset.to_netcdf(source / "weight.nc")
    windows = {
        step["accumulation_window_seconds"]
        for channel in schema.product.channels
        for step in channel.preprocessing
        if step["operation"] == "divide"
    }
    profile = RefreshProfile(
        key=standard.stem,
        input=source,
        output_name=output.name,
        cadence_hours=cadence,
        target_schema=schema_path,
        radiation_seconds=next(iter(windows)) if windows else None,
        mask={"kind": "npy", "path": "land.npy"},
    )
    return (
        profile,
        schema,
        {
            "data": values,
            "mean": mean,
            "std": std,
            "mask": land,
            "lat": latitude,
            "lon": longitude,
            "time": times,
        },
    )
