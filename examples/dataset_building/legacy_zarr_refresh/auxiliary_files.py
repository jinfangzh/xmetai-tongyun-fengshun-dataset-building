"""Migrate optional files associated with a refreshed legacy Zarr store."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr

from datakit import io as dataset_io


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npy(path: Path, values: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"destination exists (use --overwrite): {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_netcdf(path: Path, dataset: xr.Dataset, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"destination exists (use --overwrite): {path}")
    temporary = path.with_name(path.stem + ".tmp.nc")
    temporary.unlink(missing_ok=True)
    dataset.to_netcdf(temporary)
    os.replace(temporary, path)


def read_zarr_coordinate(store: Path, name: str) -> np.ndarray:
    return np.asarray(zarr.open_array(str(store / name), mode="r")[:])


def index_for_coordinate(
    source: np.ndarray, target: np.ndarray, *, longitude: bool = False
) -> np.ndarray:
    source = np.asarray(source, dtype="f8").reshape(-1)
    target = np.asarray(target, dtype="f8").reshape(-1)
    if source.size != target.size:
        raise ValueError(f"coordinate size differs: {source.size} != {target.size}")
    source_key = np.mod(source, 360.0) if longitude else source
    target_key = np.mod(target, 360.0) if longitude else target
    if np.allclose(source_key, target_key, rtol=0, atol=1e-6):
        return np.arange(source.size, dtype="i8")
    if np.allclose(source_key[::-1], target_key, rtol=0, atol=1e-6):
        return np.arange(source.size - 1, -1, -1, dtype="i8")
    lookup = {round(float(value), 7): index for index, value in enumerate(source_key)}
    try:
        indices = np.asarray(
            [lookup[round(float(value), 7)] for value in target_key], dtype="i8"
        )
    except KeyError as exc:
        raise ValueError(f"target coordinate value missing from source: {exc}") from exc
    if not np.allclose(source_key[indices], target_key, rtol=0, atol=1e-6):
        raise ValueError("coordinate mapping failed")
    return indices


def const_data_variable(dataset: xr.Dataset) -> str:
    matches = [
        name
        for name, variable in dataset.data_vars.items()
        if {"channel", "lat", "lon"}.issubset(variable.dims)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one const field, found {matches}")
    return matches[0]


def build_const(source_file: Path, target_store: Path) -> tuple[xr.Dataset, dict[str, Any]]:
    target_lat = read_zarr_coordinate(target_store, "lat").astype("f8")
    target_lon = read_zarr_coordinate(target_store, "lon").astype("f8")
    source = dataset_io.open(source_file).read(decode_cf=False)
    try:
        variable_name = const_data_variable(source)
        field = source[variable_name].transpose("channel", "lat", "lon")
        source_channels = [str(value) for value in source["channel"].values.tolist()]
        lat_index = index_for_coordinate(source["lat"].values, target_lat)
        lon_index = index_for_coordinate(
            source["lon"].values, target_lon, longitude=True
        )
        values = np.asarray(field.values, dtype="f4")
        values = values[:, lat_index, :][:, :, lon_index]
        result = xr.Dataset(
            data_vars={variable_name: (("channel", "lat", "lon"), values)},
            coords={
                "channel": np.asarray(source_channels, dtype=str),
                "lat": target_lat,
                "lon": target_lon,
            },
            attrs=dict(source.attrs),
        )
        result[variable_name].attrs.update(dict(field.attrs))
        result["channel"].attrs.update(dict(source["channel"].attrs))
        result["lat"].attrs.update(dict(source["lat"].attrs))
        result["lon"].attrs.update(dict(source["lon"].attrs))
        details = {
            "source_variable": variable_name,
            "source_channels": source_channels,
            "target_channels": source_channels,
            "latitude_reordered": not np.array_equal(
                lat_index, np.arange(lat_index.size)
            ),
            "longitude_reordered": not np.array_equal(
                lon_index, np.arange(lon_index.size)
            ),
            "generated_channels": [],
            "channel_policy": "preserved from legacy const.nc",
            "shape": list(values.shape),
        }
        return result, details
    finally:
        source.close()

def build_weight(source_file: Path, target_store: Path) -> tuple[np.ndarray, dict[str, Any]]:
    target_lat = read_zarr_coordinate(target_store, "lat").astype("f8")
    source = dataset_io.open(source_file).read(decode_cf=False)
    try:
        candidates = [v for v in source.data_vars.values() if v.dims == ("lat",)]
        if len(candidates) != 1:
            raise ValueError("expected one weight variable with dimensions (lat,)")
        field = candidates[0]
        lat_index = index_for_coordinate(source["lat"].values, target_lat)
        values = np.asarray(field.values, dtype="f4")[lat_index].reshape(1, -1, 1)
        return values, {
            "source_variable": field.name,
            "latitude_reordered": not np.array_equal(
                lat_index, np.arange(lat_index.size)
            ),
            "shape": list(values.shape),
        }
    finally:
        source.close()


def build_weight_npy(
    source_file: Path, source_root: Path, target_store: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    source_lat, _ = source_spatial_coordinates(source_root)
    target_lat = read_zarr_coordinate(target_store, "lat").astype("f8")
    raw = np.asarray(np.load(source_file, allow_pickle=False), dtype="f4")
    if raw.shape == (source_lat.size,):
        vector = raw
    elif raw.shape == (1, source_lat.size, 1):
        vector = raw[0, :, 0]
    else:
        raise ValueError(
            f"unsupported weight.npy shape {raw.shape}; expected "
            f"{(source_lat.size,)} or {(1, source_lat.size, 1)}"
        )
    lat_index = index_for_coordinate(source_lat, target_lat)
    values = vector[lat_index].reshape(1, -1, 1).astype("f4")
    return values, {
        "source_format": "npy",
        "latitude_reordered": not np.array_equal(
            lat_index, np.arange(lat_index.size)
        ),
        "shape": list(values.shape),
    }


def source_spatial_coordinates(source_root: Path) -> tuple[np.ndarray, np.ndarray]:
    const_file = source_root / "const.nc"
    if const_file.is_file():
        dataset = dataset_io.open(const_file).read(decode_cf=False)
        try:
            if "lat" in dataset.coords and "lon" in dataset.coords:
                return (
                    np.asarray(dataset["lat"].values),
                    np.asarray(dataset["lon"].values),
                )
        finally:
            dataset.close()
    return (
        np.asarray(zarr.open_array(str(source_root / "lat"), mode="r")[:]),
        np.asarray(zarr.open_array(str(source_root / "lon"), mode="r")[:]),
    )


def build_mask(source_file: Path, source_root: Path, target_store: Path) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(np.load(source_file, allow_pickle=False)).squeeze()
    source_lat, source_lon = source_spatial_coordinates(source_root)
    target_lat = read_zarr_coordinate(target_store, "lat")
    target_lon = read_zarr_coordinate(target_store, "lon")
    if values.shape != (source_lat.size, source_lon.size):
        raise ValueError(
            f"mask shape {values.shape} != source grid {(source_lat.size, source_lon.size)}"
        )
    lat_index = index_for_coordinate(source_lat, target_lat)
    lon_index = index_for_coordinate(source_lon, target_lon, longitude=True)
    values = values[lat_index, :][:, lon_index].astype(bool)
    return values, {
        "latitude_reordered": not np.array_equal(
            lat_index, np.arange(lat_index.size)
        ),
        "longitude_reordered": not np.array_equal(
            lon_index, np.arange(lon_index.size)
        ),
        "shape": list(values.shape),
    }


def build_mask_netcdf(
    source_file: Path,
    target_store: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align one two-dimensional NetCDF mask to the refreshed grid."""

    target_lat = read_zarr_coordinate(target_store, "lat")
    target_lon = read_zarr_coordinate(target_store, "lon")
    dataset = dataset_io.open(source_file).read(decode_cf=False)
    try:
        candidates = [
            variable
            for variable in dataset.data_vars.values()
            if {"lat", "lon"}.issubset(variable.dims)
        ]
        if len(candidates) != 1:
            names = [str(variable.name) for variable in candidates]
            raise ValueError(
                f"expected one mask variable with lat/lon dimensions, found {names}"
            )
        field = candidates[0].transpose("lat", "lon")
        lat_index = index_for_coordinate(dataset["lat"].values, target_lat)
        lon_index = index_for_coordinate(
            dataset["lon"].values,
            target_lon,
            longitude=True,
        )
        values = np.asarray(field.values)[lat_index, :][:, lon_index].astype(bool)
        return values, {
            "source_format": "netcdf",
            "source_variable": str(field.name),
            "latitude_reordered": not np.array_equal(
                lat_index, np.arange(lat_index.size)
            ),
            "longitude_reordered": not np.array_equal(
                lon_index, np.arange(lon_index.size)
            ),
            "shape": list(values.shape),
        }
    finally:
        dataset.close()

def build_land_mask_from_const(
    source_file: Path, target_store: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    target_lat = read_zarr_coordinate(target_store, "lat")
    target_lon = read_zarr_coordinate(target_store, "lon")
    source = dataset_io.open(source_file).read(decode_cf=False)
    try:
        variable_name = const_data_variable(source)
        field = source[variable_name].transpose("channel", "lat", "lon")
        channels = [str(value) for value in source["channel"].values.tolist()]
        if "lsm" not in channels:
            raise ValueError(f"const.nc has no lsm channel: {source_file}")
        lat_index = index_for_coordinate(source["lat"].values, target_lat)
        lon_index = index_for_coordinate(
            source["lon"].values, target_lon, longitude=True
        )
        values = np.asarray(field.values[channels.index("lsm")])
        land = values[lat_index, :][:, lon_index] > 0.5
        return land, {
            "source": str(source_file),
            "derivation": "land_mask = const.nc/lsm > 0.5; sea_mask = 1 - land_mask",
            "latitude_reordered": not np.array_equal(
                lat_index, np.arange(lat_index.size)
            ),
            "longitude_reordered": not np.array_equal(
                lon_index, np.arange(lon_index.size)
            ),
            "shape": [2, int(land.shape[0]), int(land.shape[1])],
        }
    finally:
        source.close()


def write_embedded_masks(target_store: Path, land: np.ndarray) -> None:
    array = zarr.open_array(str(target_store / "mask"), mode="r+")
    expected = (2, *land.shape)
    if tuple(array.shape) != expected:
        raise ValueError(f"embedded mask shape {array.shape} != {expected}")
    replacement = np.stack((land, ~land)).astype("u1")
    original = np.asarray(array[:]).copy()
    try:
        array[:] = replacement
        if not np.array_equal(np.asarray(array[:]), replacement):
            raise IOError("embedded mask verification failed after write")
    except Exception:
        array[:] = original
        raise


def choose_output(output_root: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = output_root / name
        if candidate.is_dir():
            return candidate
    return None


def verify_target_store(path: Path) -> None:
    if path.name.startswith(".") or path.name.endswith(".staging"):
        raise ValueError(f"refusing to modify staging directory: {path}")
    for name in (".zgroup", "data", "lat", "lon", "channel"):
        if not (path / name).exists():
            raise ValueError(f"not a completed refreshed store; missing {name}: {path}")


def migrate_auxiliary_files(
    key: str,
    spec: dict[str, Any],
    output_root: Path,
    *,
    dry_run: bool,
    overwrite: bool,
    strict: bool,
) -> dict[str, Any]:
    source = Path(spec["source"])
    target = choose_output(output_root, spec["outputs"])
    if target is None:
        message = f"no completed output found for {key}: {spec['outputs']}"
        if strict:
            raise FileNotFoundError(message)
        return {"dataset": key, "status": "skipped", "reason": message}
    verify_target_store(target)
    report: dict[str, Any] = {
        "dataset": key,
        "source": str(source),
        "target": str(target),
        "dry_run": dry_run,
        "artifacts": {},
    }

    const_source = source / "const.nc"
    if const_source.is_file():
        dataset, details = build_const(const_source, target)
        destination = target / "const.nc"
        if not dry_run:
            atomic_netcdf(destination, dataset, overwrite)
            details["sha256"] = sha256(destination)
        dataset.close()
        details.update({"source": str(const_source), "destination": str(destination)})
        report["artifacts"]["const.nc"] = details
    else:
        report["artifacts"]["const.nc"] = {"status": "source_missing_skipped"}

    weight_nc_source = source / "weight.nc"
    weight_npy_source = source / "weight.npy"
    if weight_nc_source.is_file():
        weight_source = weight_nc_source
        values, details = build_weight(weight_source, target)
        details["source_format"] = "netcdf"
        destination = target / "weight.npy"
        if not dry_run:
            atomic_npy(destination, values, overwrite)
            details["sha256"] = sha256(destination)
        details.update({"source": str(weight_source), "destination": str(destination)})
        report["artifacts"]["weight.npy"] = details
    elif weight_npy_source.is_file():
        weight_source = weight_npy_source
        values, details = build_weight_npy(weight_source, source, target)
        destination = target / "weight.npy"
        if not dry_run:
            atomic_npy(destination, values, overwrite)
            details["sha256"] = sha256(destination)
        details.update({"source": str(weight_source), "destination": str(destination)})
        report["artifacts"]["weight.npy"] = details
    else:
        report["artifacts"]["weight.npy"] = {"status": "source_missing_skipped"}

    mask_npy_source = source / "mask.npy"
    mask_nc_source = source / "mask.nc"
    mask_source: Path | None = None
    if mask_npy_source.is_file():
        mask_source = mask_npy_source
        values, details = build_mask(mask_source, source, target)
        details["source_format"] = "numpy"
    elif mask_nc_source.is_file():
        mask_source = mask_nc_source
        values, details = build_mask_netcdf(mask_source, target)
    else:
        values = None
        details = {"status": "source_missing_skipped"}

    if mask_source is not None and values is not None:
        destination = target / "mask.npy"
        if not dry_run:
            atomic_npy(destination, values, overwrite)
            details["sha256"] = sha256(destination)
        details["semantics"] = (
            "preserved from the legacy mask source; not assumed to be land_mask"
        )
        details.update({"source": str(mask_source), "destination": str(destination)})
        report["artifacts"]["mask.npy"] = details

        refresh_embedded_mask = bool(spec.get("refresh_embedded_mask", False))
        if (
            refresh_embedded_mask
            and const_source.is_file()
            and (target / "mask").is_dir()
        ):
            land, embedded_details = build_land_mask_from_const(
                const_source, target
            )
            if not dry_run:
                write_embedded_masks(target, land)
                embedded_details["verified"] = True
            report["artifacts"]["embedded_mask"] = embedded_details
        else:
            report["artifacts"]["embedded_mask"] = {
                "status": "preserved",
                "reason": (
                    "embedded mask replacement requires explicit "
                    "refresh_embedded_mask=true"
                ),
            }
    else:
        report["artifacts"]["mask.npy"] = details

    report["status"] = "planned" if dry_run else "ok"
    if not dry_run:
        manifest = target / "auxiliary_refresh_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    **report,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return report
