# -*- coding: utf-8 -*-
"""降水测试辅助工具：合成资料与文件生成。

供单元测试和集成测试共用。

文件格式（GBK，两行文件头 + 空行 + 数据行）::

    diamond  3 2025年8月20日0时1小时降水(逐时)
    25 8 20 0 1000 0 0 0 0 1 77017

    50005    112.0000     21.0000    5.0    0.0
    站号 经度 纬度 海拔(m) 降水(mm)

命名支持读取端认的两种布局：分层 ``root/YYYYMMDD/HH.000``（``station_file``）
与扁平 ``root/YYYYMMDDHH.000``（``flat_station_file``）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import xarray as xr

#: 数据行列宽（与参考实现的 diamond 3 站点列一致）
_ROW_FORMAT = "%5d %11.4f %11.4f %7.1f %7.1f"

#: 一行数据 = (站号, 经度, 纬度, 海拔 m, 降水 mm)
StationRow = Tuple[int, float, float, float, float]


def station_file(root, moment: datetime) -> Path:
    """分层布局路径：``root/YYYYMMDD/HH.000``。"""
    return Path(root) / moment.strftime("%Y%m%d") / (moment.strftime("%H") + ".000")


def flat_station_file(root, moment: datetime) -> Path:
    """扁平布局路径：``root/YYYYMMDDHH.000``。"""
    return Path(root) / (moment.strftime("%Y%m%d%H") + ".000")


def write_diamond(path, moment: datetime, rows: Sequence[StationRow]) -> None:
    """写一个时刻的 Diamond 3 站点降水文件（GBK）。

    Args:
        path: 目标文件路径，通常由 ``station_file`` / ``flat_station_file`` 给出。
        moment: 该文件的观测时刻。
        rows: ``(站号, 经度, 纬度, 海拔 m, 降水 mm)``；写入顺序即文件顺序
            （读取端会按站号重排，用来测\"文件顺序与站号顺序不一致\"）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "diamond  3 %d年%d月%d日%d时1小时降水(逐时)"
        % (moment.year, moment.month, moment.day, moment.hour),
        "%02d %d %d %d 1000 0 0 0 0 1 77017"
        % (moment.year % 100, moment.month, moment.day, moment.hour),
        "",
    ]
    lines.extend(_ROW_FORMAT % row for row in rows)
    path.write_bytes(("\n".join(lines) + "\n").encode("gbk"))


def write_fuxi_tp(root, init_utc: datetime, *, lat, lon, values: np.ndarray) -> Path:
    """写四个 FuXi tp 预报文件（时效 6/12/18/24，``root/YYYYMMDD/001.nc…``）。

    四个文件用同一张 ``values`` 场（``(lat.size, lon.size)``，mm），于是 24h 累积
    = 4 × 单时效值——测试把站点摆在格点上，就能手算插值后的 24h 总量。
    """
    root = Path(root)
    date_dir = root / init_utc.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 5):
        dataset = xr.Dataset(
            {"tp": (("lat", "lon"), np.asarray(values, dtype="f8"))},
            coords={"lat": np.asarray(lat, dtype="f8"), "lon": np.asarray(lon, dtype="f8")},
        )
        dataset["tp"].attrs["units"] = "mm"
        dataset.to_netcdf(date_dir / f"{index:03d}.nc")
    return root


def write_fuxi_ens_tp(root, init_utc: datetime, *, lat, lon, member_values) -> Path:
    """写集合 FuXi tp 预报：``root/YYYYMMDD/member_XXX/001..004.nc``。

    ``member_values``：``{"member_000": [[...]], "member_001": [[...]]}``，每个成员的
    tp 场 ``(lat.size, lon.size)``；四个时效共用同一张场，24h 累积 = 4 × 单时效值。
    """
    root = Path(root)
    date_dir = root / init_utc.strftime("%Y%m%d")
    for member, values in member_values.items():
        member_dir = date_dir / member
        member_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, 5):
            dataset = xr.Dataset(
                {"tp": (("lat", "lon"), np.asarray(values, dtype="f8"))},
                coords={
                    "lat": np.asarray(lat, dtype="f8"),
                    "lon": np.asarray(lon, dtype="f8"),
                },
            )
            dataset["tp"].attrs["units"] = "mm"
            dataset.to_netcdf(member_dir / f"{index:03d}.nc")
    return root


def write_ref_probability(root, when: datetime, station_rows) -> Path:
    """写 ``ref/MMDDHH.000``：每行 ``站号 p≥0.1 p≥4 p≥13 p≥25``（4 列概率）。"""
    path = Path(root) / f"{when.strftime('%m%d%H')}.000"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "%d %.3f %.3f %.3f %.3f" % (sid, p0, p4, p13, p25)
        for (sid, p0, p4, p13, p25) in station_rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
