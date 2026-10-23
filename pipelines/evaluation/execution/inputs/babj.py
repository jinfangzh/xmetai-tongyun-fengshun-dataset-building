"""BABJ file discovery and evaluation input orchestration."""

import datetime as _dt
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from datakit.evaluation.contracts import (
    DataIndex,
    DataKind,
    DataRequest,
)
from datakit.evaluation.errors import DecodeError, DiscoveryError, TyphoonDataError


def read_babj_analyses(path) -> Tuple[str, str, Dict[_dt.datetime, Dict[str, float]]]:
    """读 babj diamond7，只取时效 000 的实况记录。

    返回 ``(tcname, tcid, {北京时datetime: dict(lon, lat, pmin_hpa, vmax_ms)})``。

    列序是 **lon 在前、lat 在后**（第 6、7 列），别按经纬顺序读反。
    """
    try:
        with open(path, "rb") as f:
            f.readline()
            head = f.readline().decode("gbk", "ignore").split()
        tcname, tcid = head[0], head[1]
        rows = np.loadtxt(path, skiprows=3, encoding="gbk", ndmin=2)
    except (OSError, ValueError, IndexError) as e:
        raise DecodeError("解析 babj 报文 %s 失败: %s" % (path, e)) from e
    analyses: Dict[_dt.datetime, Dict[str, float]] = {}
    for r in rows:
        if int(round(r[4])) != 0:  # 只要实况行
            continue
        t = _dt.datetime(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
        analyses[t] = {
            "lon": float(r[5]),
            "lat": float(r[6]),
            "pmin_hpa": float(r[7]),
            "vmax_ms": float(r[8]),
        }
    if not analyses:
        raise TyphoonDataError("%s 中没有时效 000 的实况记录" % path)
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
            records = io.open(self.babj_dir, "babj").discover(all_files=True)
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


from datakit import io
from datakit.io.backends.babj import assemble_storm_tracks
from pipelines.evaluation.execution.sources import build_bundle


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
