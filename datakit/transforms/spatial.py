"""In-memory grid selection, alignment, and spatial interpolation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import xarray as xr


def _missing_interpolation_dependency() -> ImportError:
    return ImportError(
        "Spatial interpolation requires scipy; install xmetai-datakit[interpolation]"
    )


def resize(
    obj: xr.Dataset | xr.DataArray,
    *,
    lat: Sequence[float] | None = None,
    lon: Sequence[float] | None = None,
    resolution: float = 0.25,
    extrapolate: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Interpolate onto explicit coordinates or a regular target resolution."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    source_lat = np.asarray(obj["lat"].values)
    source_lon = np.asarray(obj["lon"].values)
    if source_lat.size < 2 or source_lon.size < 2:
        raise ValueError("lat and lon must each contain at least two points")
    if lat is None:
        delta = np.sign(source_lat[1] - source_lat[0]) * resolution
        lat = np.arange(source_lat[0], source_lat[-1] + delta / 2, delta)
    if lon is None:
        delta = np.sign(source_lon[1] - source_lon[0]) * resolution
        stop = source_lon[-1] + delta / 2
        lon = np.arange(source_lon[0], stop, delta)
    kwargs = {"fill_value": "extrapolate"} if extrapolate else None
    try:
        return obj.interp(lat=np.asarray(lat), lon=np.asarray(lon), kwargs=kwargs)
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        raise _missing_interpolation_dependency() from exc


def interpolate_grid(
    obj: xr.Dataset | xr.DataArray,
    *,
    point_scale: float | Sequence[float] | None = None,
    target_shape: Sequence[int] | None = None,
) -> xr.Dataset | xr.DataArray:
    """Interpolate to a grid with an explicit shape or point-count scale.

    ``point_scale=2`` doubles the point count along both spatial dimensions;
    a two-item sequence may be used for different latitude/longitude scales.
    ``target_shape=(nlat, nlon)`` is the least ambiguous form.
    """

    lat = np.asarray(obj["lat"].values)
    lon = np.asarray(obj["lon"].values)
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
        raise ValueError("lat and lon must be one-dimensional with at least two points")

    supplied = sum(argument is not None for argument in (point_scale, target_shape))
    if supplied != 1:
        raise ValueError("Provide exactly one of point_scale or target_shape")
    if target_shape is not None:
        shape = np.asarray(target_shape)
        if (
            shape.shape != (2,)
            or not np.issubdtype(shape.dtype, np.integer)
            or np.any(shape < 1)
        ):
            raise ValueError("target_shape must contain two positive integers")
        nlat, nlon = (int(shape[0]), int(shape[1]))
    else:
        try:
            scales = np.asarray(point_scale, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "point_scale must be positive scalar or two-item sequence"
            ) from exc
        if scales.ndim == 0:
            scales = np.repeat(scales, 2)
        if scales.shape != (2,) or not np.all(np.isfinite(scales)):
            raise ValueError("point_scale must be positive scalar or two-item sequence")
        if np.any(scales <= 0):
            raise ValueError("point_scale must be positive scalar or two-item sequence")
        nlat = max(1, int(np.ceil(lat.size * scales[0])))
        nlon = max(1, int(np.ceil(lon.size * scales[1])))

    new_lat = np.linspace(
        lat[0],
        lat[-1],
        nlat,
    )
    new_lon = np.linspace(
        lon[0],
        lon[-1],
        nlon,
    )
    try:
        return obj.interp(lat=new_lat, lon=new_lon, method="linear")
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        raise _missing_interpolation_dependency() from exc


def crop(
    obj: xr.Dataset | xr.DataArray,
    *,
    lat: Sequence[float],
    lon: Sequence[float],
    tolerance: float = 0.01,
) -> xr.Dataset | xr.DataArray:
    """Select target grid points without replacing their source coordinates."""

    target_lat = np.asarray(lat)
    target_lon = np.asarray(lon)
    if target_lat.ndim != 1 or target_lon.ndim != 1:
        raise ValueError("Target lat/lon coordinates must be one-dimensional")
    if target_lat.size == 0 or target_lon.size == 0:
        raise ValueError("Target lat/lon coordinates cannot be empty")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative number")

    def coordinate_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        if source.ndim != 1:
            raise ValueError("Source lat/lon coordinates must be one-dimensional")
        indices = []
        for value in target:
            matches = np.flatnonzero(
                np.isclose(source, value, atol=tolerance, rtol=0)
            )
            if matches.size == 0:
                raise ValueError(f"Target coordinate {value!r} not found in source data")
            if matches.size > 1:
                raise ValueError(
                    f"Target coordinate {value!r} ambiguously matches source data"
                )
            indices.append(int(matches[0]))
        if len(set(indices)) != len(indices):
            raise ValueError("Target coordinates must map to unique source points")
        return np.asarray(indices, dtype=np.int64)

    source_lat = np.asarray(obj["lat"].values)
    source_lon = np.asarray(obj["lon"].values)
    lat_indices = coordinate_indices(
        source_lat,
        target_lat,
    )
    lon_indices = coordinate_indices(
        source_lon,
        target_lon,
    )
    result = obj.isel(lat=lat_indices, lon=lon_indices)
    if not np.allclose(result.lat.values, target_lat, atol=tolerance, rtol=0):
        raise ValueError("Selected latitude coordinates do not match the target grid")
    if not np.allclose(result.lon.values, target_lon, atol=tolerance, rtol=0):
        raise ValueError("Selected longitude coordinates do not match the target grid")
    return result


# Grid alignment and cached regular-grid interpolation.


def _weights(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, ...]:
    upper = np.searchsorted(source, target, side="right")
    upper = np.clip(upper, 1, len(source) - 1)
    lower = upper - 1
    weight_upper = ((target - source[lower]) / (source[upper] - source[lower])).astype("f4")
    return lower, upper, 1.0 - weight_upper, weight_upper


@dataclass
class RegularLatLonRegridder:
    """Separable linear interpolation onto one regular latitude/longitude grid."""

    target_lat: np.ndarray
    target_lon: np.ndarray
    _cache: dict[tuple[bytes, bytes], tuple[np.ndarray, ...]] = field(default_factory=dict)

    def apply(self, data: xr.DataArray, time_index: int | None = None) -> np.ndarray:
        if time_index is not None:
            data = data.isel(time=time_index)
        field = data.squeeze(drop=True).transpose("lat", "lon")

        source_lat = np.asarray(field.lat.values, dtype="f8")
        source_lon = np.mod(np.asarray(field.lon.values, dtype="f8"), 360.0)

        target_lat = np.asarray(self.target_lat, dtype="f8")
        target_lon = np.asarray(self.target_lon, dtype="f8")

        if (
            np.array_equal(source_lat.astype("f4"), target_lat.astype("f4"))
            and np.array_equal(source_lon.astype("f4"), target_lon.astype("f4"))
        ):
            return np.asarray(field.values, dtype="f4")

        lat_order = np.argsort(source_lat)
        lon_order = np.argsort(source_lon)
        source_lat = source_lat[lat_order]
        source_lon = source_lon[lon_order]
        values = np.asarray(field.values, dtype="f4")[lat_order][:, lon_order]

        source_lon = np.concatenate((source_lon, source_lon[:1] + 360.0))
        values = np.concatenate((values, values[:, :1]), axis=1)

        key = (source_lat.tobytes(), source_lon.tobytes())
        weights = self._cache.get(key)
        if weights is None:
            weights = _weights(source_lat, target_lat) + _weights(source_lon, target_lon)
            self._cache[key] = weights

        lat0, lat1, lat_w0, lat_w1, lon0, lon1, lon_w0, lon_w1 = weights
        along_lat = values[lat0] * lat_w0[:, None] + values[lat1] * lat_w1[:, None]
        return (along_lat[:, lon0] * lon_w0[None, :] + along_lat[:, lon1] * lon_w1[None, :]).astype(
            "f4", copy=False
        )


def align_lon_to(ds: xr.Dataset, target_lon: Any) -> xr.Dataset:
    """把 ``ds`` 的经度折算到 ``target_lon`` 所在的那一圈。

    不同数据源的经度原点会差 180°（如 ERA5 zarr 用 -180..180，预报文件用 0..360）。
    xarray 的 ``interp`` 在目标坐标超出源坐标范围时**直接返回 NaN**，不报错，
    所以不折算就会静默配对出一整片空场。经度是周期坐标，折算等价且不丢信息。

    折算后必须 ``sortby``：``[-180..180)`` 取模到 ``[0..360)`` 会变成降序，
    而 ``spectrum`` / ``zonal_spectrum`` 的 rFFT 依赖经度升序。
    """
    if "lon" not in ds.coords:
        return ds
    lon = np.asarray(ds["lon"].values)
    target = np.asarray(target_lon)
    if lon.size == 0 or target.size == 0:
        return ds
    lon_min = float(lon.min())
    target_min = float(target.min())
    if lon_min < 0.0 <= target_min:
        return ds.assign_coords(lon=lon % 360.0).sortby("lon")
    if lon_min >= 0.0 > target_min:
        return ds.assign_coords(lon=((lon + 180.0) % 360.0) - 180.0).sortby("lon")
    return ds


def align_to_grid(
    ds: xr.Dataset, target_lat: Any, target_lon: Any, atol: float = 1e-6
) -> Optional[xr.Dataset]:
    """把 ``ds`` 按**索引**搬到 ``target_lat`` / ``target_lon`` 这套网格上。

    只认「同一套网格，可能上下颠倒（lat 升序 vs 降序）或差半圈（经度原点不同）」
    这一种情况：lat 要么逐点相等、要么逐点反转；lon 要么相等、要么存在一个环形
    位移 ``k`` 能把源经度对齐到目标经度。对不上返回 ``None``，由调用方决定是否
    退回插值。

    这里一个插值都不做，是参考实现（``vfc/regr_ens.py:84-118``）的做法：两边本
    就是同一套全球网格时，``a[:, ::-1, :]`` 和 ``np.roll`` 是精确搬运——既没有
    双线性权重误差，也不会把 (时效 × 要素) 展平成 GB 级的中间数组。
    """
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return None
    lat = np.asarray(ds["lat"].values, dtype="f8")
    lon = np.asarray(ds["lon"].values, dtype="f8")
    want_lat = np.asarray(target_lat, dtype="f8")
    want_lon = np.asarray(target_lon, dtype="f8")

    if lat.shape != want_lat.shape or lon.shape != want_lon.shape or lon.size == 0:
        return None

    if np.allclose(lat, want_lat, rtol=0.0, atol=atol):
        flip = False
    elif np.allclose(lat, want_lat[::-1], rtol=0.0, atol=atol):
        flip = True
    else:
        return None

    # 位移量先用首元素定位候选，再整体比对确认：均匀网格上是 O(n)，
    # 逐个 k 去试是 O(n²)（1440² 也还好，但没必要）。
    roll = next(
        (
            int(k)
            for k in np.flatnonzero(np.isclose(lon, want_lon[0], rtol=0.0, atol=atol))
            if np.allclose(np.roll(lon, int(k)), want_lon, rtol=0.0, atol=atol)
        ),
        None,
    )
    if roll is None:
        return None

    if flip:
        ds = ds.isel(lat=slice(None, None, -1))
    if roll:
        ds = ds.roll(lon=roll, roll_coords=True)
    # 坐标按目标逐点写回：下游掩码/权重是按标签对齐的，浮点尾数级别的差异
    # 留在坐标里会让本可命中的配对悄悄落空。
    return ds.assign_coords(lat=want_lat, lon=want_lon)


# Interpolation from a grid to station coordinates.


_METHODS = {"bilinear": "linear", "nearest": "nearest"}


class GridToStationInterpolator:
    """网格场插值到站点。

    输入：网格数据 (lat, lon) 或 (..., lat, lon)
    输出：站点数据 (..., station)
    """

    def __init__(self, method: Literal["bilinear", "nearest"] = "bilinear"):
        """
        Args:
            method: 插值方法（bilinear / nearest）
        """
        if method not in _METHODS:
            raise ValueError(f"Unknown interpolation method: {method}")
        self.method = method

    def transform(
        self,
        grid_data: xr.DataArray,
        station_lats: np.ndarray,
        station_lons: np.ndarray,
    ) -> xr.DataArray:
        """插值网格场到站点。

        Args:
            grid_data: 网格数据，必须包含 'lat' 和 'lon' 维度
            station_lats: 站点纬度数组 (n_stations,)
            station_lons: 站点经度数组 (n_stations,)

        Returns:
            站点数据，shape = (..., n_stations)；网格范围外的站点为 NaN
        """
        if "lat" not in grid_data.dims or "lon" not in grid_data.dims:
            raise ValueError("grid_data must have 'lat' and 'lon' dimensions")

        grid_lats = np.asarray(grid_data.coords["lat"].values, dtype="f8")
        grid_lons = np.asarray(grid_data.coords["lon"].values, dtype="f8")
        if not (np.all(np.diff(grid_lats) > 0) or np.all(np.diff(grid_lats) < 0)):
            raise ValueError("Grid latitudes must be monotonic")
        if not (np.all(np.diff(grid_lons) > 0) or np.all(np.diff(grid_lons) < 0)):
            raise ValueError("Grid longitudes must be monotonic")

        lats = np.asarray(station_lats, dtype="f8")
        lons = np.asarray(station_lons, dtype="f8").copy()
        if grid_lons.min() >= 0 and grid_lons.max() <= 360:
            lons[lons < 0] += 360.0
        elif grid_lons.min() >= -180 and grid_lons.max() <= 180:
            lons[lons > 180] -= 360.0

        dim_order = [d for d in grid_data.dims if d not in ("lat", "lon")] + ["lat", "lon"]
        interpolated = grid_data.transpose(*dim_order).interp(
            lat=xr.DataArray(lats, dims="station"),
            lon=xr.DataArray(lons, dims="station"),
            method=_METHODS[self.method],
        )
        interpolated = interpolated.assign_coords(
            station=np.arange(lats.size),
            station_lat=("station", lats),
            station_lon=("station", lons),
        )
        interpolated.attrs = dict(grid_data.attrs)
        return interpolated
