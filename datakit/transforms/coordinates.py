from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np
import xarray as xr

from datakit.schema import DIMENSION_ALIASES

XarrayObject: TypeAlias = xr.Dataset | xr.DataArray


def _pick_coord_name(obj: XarrayObject, candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in obj.coords:
            return name
    return None


def _coord_range(obj: XarrayObject, candidates: Sequence[str]) -> list[float] | None:
    coord_name = _pick_coord_name(obj, candidates)
    if coord_name is None:
        return None
    values = np.asarray(obj.coords[coord_name].values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return None
    return [float(valid.min()), float(valid.max())]


def enrich_dataset_attrs(
    obj: XarrayObject,
    *,
    dataset_attrs: Mapping[str, object] | None = None,
    add_geospatial_ranges: bool = False,
    lat_coord_names: Sequence[str] = ("lat", "latitude"),
    lon_coord_names: Sequence[str] = ("lon", "longitude"),
) -> XarrayObject:
    """Inject common metadata attrs and optional geospatial coordinate ranges."""

    if not dataset_attrs and not add_geospatial_ranges:
        return obj

    enriched = obj.copy(deep=False)
    if dataset_attrs:
        enriched.attrs.update(dict(dataset_attrs))

    if add_geospatial_ranges:
        lat_range = _coord_range(enriched, lat_coord_names)
        lon_range = _coord_range(enriched, lon_coord_names)
        if lat_range is not None:
            enriched.attrs["geospatial_lat_range"] = lat_range
        if lon_range is not None:
            enriched.attrs["geospatial_lon_range"] = lon_range

    return enriched


def select_singleton(obj: XarrayObject, dimension: str) -> XarrayObject:
    """Select and remove a dimension only when its size is exactly one.

    A scalar coordinate with the same name is removed as well. Objects that do
    not contain the requested dimension or coordinate are returned unchanged.
    """

    if dimension in obj.dims:
        size = obj.sizes[dimension]
        if size != 1:
            raise ValueError(f"Expected one {dimension}, got {size}")
        return obj.isel({dimension: 0}, drop=True)
    if dimension in obj.coords:
        return obj.drop_vars(dimension)
    return obj


def standardize_dimensions(obj: XarrayObject) -> XarrayObject:
    """Rename common meteorological dimensions and coordinates to canonical names."""

    present = set(obj.dims) | set(obj.coords)
    mapping: dict[str, str] = {}
    targets: dict[str, str] = {}
    for old, new in DIMENSION_ALIASES.items():
        if old not in present or old == new:
            continue
        if new in present:
            raise ValueError(
                f"Cannot standardize {old!r} to {new!r}: both names are present"
            )
        if new in targets:
            raise ValueError(
                f"Cannot standardize both {targets[new]!r} and {old!r} to {new!r}"
            )
        mapping[old] = new
        targets[new] = old
    return obj.rename(mapping) if mapping else obj


def normalize_coordinates(
    obj: XarrayObject,
    *,
    north_to_south: bool = True,
    longitude_range: str = "0_360",
    add_level: bool = True,
) -> XarrayObject:
    """Normalize latitude order, longitude convention, and optional channel dimension."""

    result = standardize_dimensions(obj)
    if "lat" in result.coords:
        latitude = np.asarray(result["lat"].values)
        if latitude.ndim != 1:
            raise ValueError("Canonical 'lat' coordinates must be one-dimensional")
        if np.unique(latitude).size != latitude.size:
            raise ValueError("Latitude coordinates contain duplicates")
        if latitude.size > 1:
            result = result.sortby("lat", ascending=not north_to_south)

    if "lon" in result.coords:
        lon = result["lon"]
        longitude = np.asarray(lon.values)
        if longitude.ndim != 1:
            raise ValueError("Canonical 'lon' coordinates must be one-dimensional")
        if longitude_range == "0_360":
            normalized = longitude % 360
        elif longitude_range == "-180_180":
            normalized = ((longitude + 180) % 360) - 180
        else:
            raise ValueError("longitude_range must be '0_360' or '-180_180'")
        if np.unique(normalized).size != normalized.size:
            raise ValueError(
                "Longitude normalization creates duplicate coordinates; "
                "remove the duplicated cyclic endpoint first"
            )
        result = result.assign_coords(lon=normalized).sortby("lon")

    if add_level and "level" not in result.dims and "channel" not in result.dims:
        axis = 1 if "time" in result.dims else 0
        result = result.expand_dims({"level": [1]}, axis=axis)
    return result


def nearest_indices(
    obj: XarrayObject,
    coordinates: Sequence[float],
    *,
    dim: str = "depth",
) -> list[int]:
    source = np.asarray(obj[dim].values, dtype=float)
    if source.ndim != 1 or source.size == 0:
        raise ValueError(f"Coordinate '{dim}' must be a non-empty 1-D array")
    return [int(np.abs(source - coordinate).argmin()) for coordinate in coordinates]
