# -*- coding: utf-8 -*-
"""BSS 外部气候概率参考数据源（ref/MMDDHH.000）。

对标 ``ref/tiqnqi/xmetai_model_verification/vfc/station_categorical.py::RefProb``：

* 逐 6h 气候概率文件，每行一个站：``站号 p≥0.1 p≥4 p≥13 p≥25``（4 列概率，[0,1]）；
* 文件名为**北京时**，HH 为 6h 窗口的**结束时刻**；
* 按验证时刻北京时查表；缺 ref 的站对应行置 NaN（由调用方按 0 兜底）。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from datakit import io
from datakit.evaluation.errors import ConfigError


class RefProbabilitySource:
    """BSS 的逐 6h 气候概率参考（按验证时刻北京时查表）。

    不是格点/时间序列数据，因此不走 Catalog/DataBundle 协议；暴露
    ``probabilities(when, station_ids)`` 供站点协议直接查表。
    """

    def __init__(
        self,
        root_dir,
        source_id: str = "ref_probability",
        description="station_probability",
    ):
        self.source_id = source_id
        self.root_dir = str(root_dir)
        source = io.open(root_dir, description)
        self._thresholds = list(source.description["thresholds"])
        self._cache: Dict[str, Dict[int, np.ndarray]] = {}
        try:
            records = source.discover(all_files=True)
        except FileNotFoundError as exc:
            raise ConfigError(f"ref 目录 {self.root_dir} 下无 .000 文件") from exc
        self._index = {
            record.fields["cycle_time"]: str(record.path) for record in records
        }

    @property
    def thresholds(self):
        return list(self._thresholds)

    def _load(self, key: str) -> Dict[int, np.ndarray]:
        if key not in self._cache:
            a = np.loadtxt(self._index[key], ndmin=2)
            if a.shape[1] != len(self._thresholds) + 1:
                raise ConfigError(f"ref {key} 的列数与描述中的阈值不一致")
            self._cache[key] = {int(sid): a[i, 1:] for i, sid in enumerate(a[:, 0])}
        return self._cache[key]

    def probabilities(self, when, station_ids) -> np.ndarray:
        """when: 北京时 datetime；station_ids: 站号数组 → (n, 4) 概率矩阵。

        缺 ref 的站对应行置 NaN（4 列全 NaN），由调用方决定跳过。
        """
        key = when.strftime("%m%d%H")
        if key not in self._index:
            raise ConfigError(f"ref 缺少时次 {key}（{when}）")
        table = self._load(key)
        out = np.full((len(station_ids), len(self._thresholds)), np.nan, dtype="f8")
        for i, sid in enumerate(station_ids):
            row = table.get(int(sid))
            if row is not None:
                out[i] = row
        return out
