"""Select BABJ analyses and assemble evaluation storm tracks."""

import datetime as _dt
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from datakit.evaluation.contracts import (
    DataIndex,
    DataKind,
    DataRequest,
)
from datakit.evaluation.errors import DecodeError, DiscoveryError, TyphoonDataError
from datakit.io import get_description, open_dataset, source_config

from ._bundle import build_bundle


def read_babj_analyses(path) -> tuple[str, str, dict[_dt.datetime, dict[str, float]]]:
    """读 babj diamond7，只取时效 000 的实况记录。

    返回 ``(tcname, tcid, {北京时datetime: dict(lon, lat, pmin_hpa, vmax_ms)})``。

    列序是 **lon 在前、lat 在后**（第 6、7 列），别按经纬顺序读反。
    """
    try:
        with open_dataset(path, source_config("observation_babj")).read(
            all_files=True
        ) as dataset:
            tcname = str(dataset.tcname.values[0])
            tcid = str(dataset.storm.values[0])
            names = get_description(source_config("observation_babj"))["format"][
                "columns"
            ]
            rows = np.column_stack([dataset[name].values for name in names])
    except (OSError, ValueError, IndexError) as exc:
        raise DecodeError(f"解析 babj 报文 {path} 失败: {exc}") from exc
    analyses: dict[_dt.datetime, dict[str, float]] = {}
    for row in rows:
        if round(row[4]) != 0:
            continue
        moment = _dt.datetime(*(int(value) for value in row[:4]))  # noqa: DTZ001 -- preserve source wall time
        analyses[moment] = {
            "lon": float(row[5]),
            "lat": float(row[6]),
            "pmin_hpa": float(row[7]),
            "vmax_ms": float(row[8]),
        }
    if not analyses:
        raise TyphoonDataError(f"{path} 中没有时效 000 的实况记录")
    return tcname, tcid, analyses


class BabjCatalog:
    """BABJ 报文目录：``babj*.dat``（也接受直接指到一个报文文件）。"""

    def __init__(self, babj_dir: Path):
        """
        Args:
            babj_dir: 报文目录，或单个 ``babj<编号>.dat`` 文件
        """
        self.babj_dir = Path(babj_dir)

    def discover(self, request: DataRequest) -> DataIndex:
        """列出目录下全部报文。不按单个时刻筛选，保留完整轨迹。"""
        if not self.babj_dir.exists():
            raise DiscoveryError(
                f"BABJ 路径不存在: {self.babj_dir}", source_id=request.source_id
            )
        if self.babj_dir.is_file():
            return DataIndex(
                source_id=request.source_id, available=[self.babj_dir], ambiguous=[]
            )
        try:
            records = open_dataset(
                self.babj_dir, source_config("observation_babj")
            ).discover(all_files=True)
            files = [record.path for record in records]
        except FileNotFoundError:
            files = []
        if not files:
            raise DiscoveryError(
                f"{self.babj_dir} 下没有 babj*.dat", source_id=request.source_id
            )
        return DataIndex(source_id=request.source_id, available=files, ambiguous=[])

    def file_time(self, path) -> datetime:
        """文件里最早的实况时刻（供按时间选文件的调用方兜底）。"""
        _, _, analyses = read_babj_analyses(path)
        return min(analyses)


class BabjSource:
    """Load BABJ records, prepare storm tracks and build evaluation input."""

    def __init__(self, source_id="babj", version="1.0.0"):
        self.source_id = source_id
        self.version = version

    def read(self, request, index):
        storms = []
        for path in index.available:
            tcname, tcid, analyses = read_babj_analyses(path)
            storms.append({"tcid": str(tcid), "tcname": tcname, "analyses": analyses})
        if not storms:
            raise DiscoveryError("没有可读的 BABJ 报文", source_id=request.source_id)
        dataset = assemble_storm_tracks(storms)
        return build_bundle(
            request,
            dataset,
            [str(path) for path in index.available],
            self.source_id,
            self.version,
            kind=DataKind.STATION_OBSERVATION,
            grid_type="storm_track",
            timezone="Asia/Shanghai",
        )


def assemble_storm_tracks(storms):
    """Preserve source wall times and missing positions across storm tracks."""
    if not storms:
        raise ValueError("No decoded storm analyses")
    times = sorted({t for storm in storms for t in storm["analyses"]})
    time_index = np.array(times, dtype="datetime64[ns]")
    tcid_index = np.array([storm["tcid"] for storm in storms], dtype=object)
    tcname_index = np.array([storm["tcname"] for storm in storms], dtype=object)

    shape = (len(times), len(storms))
    lat = np.full(shape, np.nan, dtype="f8")
    lon = np.full(shape, np.nan, dtype="f8")
    pmin = np.full(shape, np.nan, dtype="f8")
    vmax = np.full(shape, np.nan, dtype="f8")
    slot = {t: i for i, t in enumerate(times)}
    for j, storm in enumerate(storms):
        for moment, record in storm["analyses"].items():
            i = slot[moment]  # 该时刻必有槽位（times 是全并集）
            lat[i, j] = record["lat"]
            lon[i, j] = record["lon"]
            pmin[i, j] = record["pmin_hpa"]
            vmax[i, j] = record["vmax_ms"]

    ds = xr.Dataset(
        {
            "lat": (("time", "storm"), lat),
            "lon": (("time", "storm"), lon),
            "pmin": (("time", "storm"), pmin),
            "vmax": (("time", "storm"), vmax),
        },
        coords={
            "time": time_index,
            "storm": tcid_index,
            "tcname": ("storm", tcname_index),
        },
    )
    ds["lat"].attrs.update(units="degrees_north", long_name="Storm centre latitude")
    ds["lon"].attrs.update(units="degrees_east", long_name="Storm centre longitude")
    ds["pmin"].attrs.update(units="hPa", long_name="Minimum sea level pressure")
    ds["vmax"].attrs.update(units="m/s", long_name="Maximum sustained wind")

    return ds
