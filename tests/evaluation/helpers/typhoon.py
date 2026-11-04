# -*- coding: utf-8 -*-
"""台风测试辅助工具：合成数据与文件生成。

供单元测试和集成测试共用。

坐标约定与主仓库一致：经纬度对一律 ``(lat, lon)``；init 用 UTC，
BABJ 报文用北京时（init 的北京时 = init + 8h）。
"""

from __future__ import annotations

import datetime as _dt
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from datakit.evaluation.metrics.track_error import along_cross_track, great_circle_km

#: 同经度上 1° 纬度的大圆距离（r=6371 km），用来手算预期路径误差
KM_PER_DEG = math.pi * 6371.0 / 180.0

#: 合成网格：0.25°，所有设计位置都落在格点上
GRID_LAT = np.round(np.arange(15.0, 27.0 + 1e-9, 0.25), 6)
GRID_LON = np.round(np.arange(115.0, 127.0 + 1e-9, 0.25), 6)

#: 合成场参数：中心 95000 Pa = 950 hPa；风场恒定（vmax 就是它）
MSL_BASE_PA = 101000.0
MSL_MIN_PA = 95000.0
MSL_SIGMA_DEG = 2.0
WIND_MS = 30.0


# --------------------------------------------------------------------------
# 合成输入工厂
# --------------------------------------------------------------------------


def gaussian_msl(
    center_lat: float,
    center_lon: float,
    *,
    lat: np.ndarray = GRID_LAT,
    lon: np.ndarray = GRID_LON,
    base_pa: float = MSL_BASE_PA,
    min_pa: float = MSL_MIN_PA,
    sigma_deg: float = MSL_SIGMA_DEG,
) -> np.ndarray:
    """以 ``(center_lat, center_lon)`` 为极小值的高斯海平面气压场（Pa）。"""
    d2 = (lat[:, None] - center_lat) ** 2 + (lon[None, :] - center_lon) ** 2
    return base_pa - (base_pa - min_pa) * np.exp(-d2 / (2.0 * sigma_deg**2))


def _write_field_file(path: Path, lat, lon, center, *, u_ms: float, v_ms: float) -> None:
    dataset = xr.Dataset(
        {
            "msl": (("lat", "lon"), gaussian_msl(center[0], center[1], lat=lat, lon=lon)),
            "u10m": (("lat", "lon"), np.full((lat.size, lon.size), float(u_ms))),
            "v10m": (("lat", "lon"), np.full((lat.size, lon.size), float(v_ms))),
        },
        coords={"lat": lat, "lon": lon},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)


def write_fuxi_single(
    root,
    init_utc: datetime,
    centers: Sequence[Tuple[float, float]],
    *,
    lat=GRID_LAT,
    lon=GRID_LON,
    u_ms: float = WIND_MS,
    v_ms: float = 0.0,
) -> None:
    """写确定性 FuXi 目录：``root/YYYYMMDD/001.nc…``。

    ``centers`` 按 lead 升序给每个时效的台风中心；第 k 个文件（001.nc）对应
    时效 6k h（布局 ``lead_from="index"``，文件顺序即时效顺序）。
    """
    date_dir = Path(root) / init_utc.strftime("%Y%m%d")
    for index, center in enumerate(centers, start=1):
        _write_field_file(date_dir / f"{index:03d}.nc", lat, lon, center, u_ms=u_ms, v_ms=v_ms)


def write_fuxi_ens(
    root,
    init_utc: datetime,
    member_centers: Mapping[str, Sequence[Tuple[float, float]]],
    *,
    lat=GRID_LAT,
    lon=GRID_LON,
    u_ms: float = WIND_MS,
    v_ms: float = 0.0,
) -> None:
    """写集合 FuXi 目录：``root/YYYYMMDD/member_XXX/001.nc…``。

    ``member_centers`` 形如 ``{"member_000": [(lat, lon), ...], "member_001": [...]}``。
    """
    date_dir = Path(root) / init_utc.strftime("%Y%m%d")
    for member, centers in member_centers.items():
        for index, center in enumerate(centers, start=1):
            _write_field_file(
                date_dir / member / f"{index:03d}.nc", lat, lon, center, u_ms=u_ms, v_ms=v_ms
            )


def write_babj(
    path,
    tcname: str,
    tcid: str,
    rows: Sequence[Tuple[datetime, float, float, float, float, float]],
) -> None:
    """写 BABJ diamond7 报文（GBK）。

    ``rows``: ``(北京时 datetime, 时效 h, lon, lat, pmin_hpa, vmax_ms)``；
    时效 000 的行才是分析实况（读取端只认这些行，其余行用来测过滤）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(("diamond 7 %s台风路径\n" % tcname).encode("gbk"))
        stream.write(("%s      %s   20            %d\n" % (tcname, tcid, len(rows))).encode("gbk"))
        stream.write(b"\n")
        for moment, tau, lon, lat, pmin, vmax in rows:
            stream.write(
                (
                    "%4d %3d %3d %3d %3d %11.1f %11.1f %6d %5d"
                    "     0.0       0.0       0.0       0.0\n"
                    % (
                        moment.year,
                        moment.month,
                        moment.day,
                        moment.hour,
                        int(tau),
                        lon,
                        lat,
                        int(pmin),
                        int(vmax),
                    )
                ).encode("ascii")
            )


# Frozen legacy comparison helper; production uses paired batches and TrackError.
def match_errors(
    track: Dict[str, List[float]],
    leads_hours: Sequence[float],
    init_utc: _dt.datetime,
    babj: Dict[_dt.datetime, Dict[str, float]],
    tz_shift: float = 8.0,
) -> List[Dict[str, float]]:
    """诊断路径对 babj 实况逐时效配对求误差。

    track: diagnose_track 的输出；babj: read_babj_analyses 的字典
    （北京时）。验证时刻(北京时) = init_utc + 时效 + tz_shift。
    无实况记录的时效各项为 NaN（路径/强度误差保持 NaN，不外推）。

    首个**配对成功**的时效 ``at_km``/``ct_km`` 为空——没有前一个实况位置就
    无从定向；实况有缺口时，缺的是那个时效，不是固定 ``lead_h=6``。
    """
    prev_babj: Optional[Tuple[float, float]] = None
    rows: List[Dict[str, float]] = []
    for k, tau in enumerate(leads_hours):
        v_bjt = init_utc + _dt.timedelta(hours=float(tau) + tz_shift)
        ob = babj.get(v_bjt)
        f_lat, f_lon = track["lat"][k], track["lon"][k]
        f_p, f_v = track["pmin_hpa"][k], track["vmax"][k]
        if ob is None:
            rows.append(
                {
                    "lead_h": float(tau),
                    "valid_bjt": str(v_bjt),
                    "fcst_lat": f_lat,
                    "fcst_lon": f_lon,
                    "fcst_pmin_hpa": f_p,
                    "fcst_vmax_ms": f_v,
                    "obs_lat": np.nan,
                    "obs_lon": np.nan,
                    "obs_pmin_hpa": np.nan,
                    "obs_vmax_ms": np.nan,
                    "track_err_km": np.nan,
                    "at_km": np.nan,
                    "ct_km": np.nan,
                    "wind_err_ms": np.nan,
                    "pmin_err_hpa": np.nan,
                }
            )
            continue
        o_lat, o_lon = ob["lat"], ob["lon"]
        err = float(great_circle_km(f_lat, f_lon, o_lat, o_lon))
        if prev_babj is None:  # 首个配对时次
            at = ct = float("nan")
        else:
            at, ct = along_cross_track(prev_babj[0], prev_babj[1], o_lat, o_lon, f_lat, f_lon)
        rows.append(
            {
                "lead_h": float(tau),
                "valid_bjt": str(v_bjt),
                "fcst_lat": f_lat,
                "fcst_lon": f_lon,
                "fcst_pmin_hpa": f_p,
                "fcst_vmax_ms": f_v,
                "obs_lat": o_lat,
                "obs_lon": o_lon,
                "obs_pmin_hpa": ob["pmin_hpa"],
                "obs_vmax_ms": ob["vmax_ms"],
                "track_err_km": err,
                "at_km": at,
                "ct_km": ct,
                "wind_err_ms": f_v - ob["vmax_ms"] if np.isfinite(f_v) else float("nan"),
                "pmin_err_hpa": f_p - ob["pmin_hpa"],
            }
        )
        prev_babj = (o_lat, o_lon)
    return rows
