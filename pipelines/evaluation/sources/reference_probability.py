"""Look up station climatological probabilities for evaluation BSS."""

import numpy as np

from datakit.evaluation.errors import ConfigError
from datakit.io import open_dataset, source_config


class RefProbabilitySource:
    """BSS 的逐 6h 气候概率参考（按验证时刻北京时查表）。

    不是格点/时间序列数据，因此不走 Catalog/DataBundle 协议；暴露
    ``probabilities(when, station_ids)`` 供站点协议直接查表。
    """

    def __init__(
        self,
        root_dir,
        source_id: str = "ref_probability",
        layout=None,
    ):
        self.source_id = source_id
        self.root_dir = str(root_dir)
        source = open_dataset(
            root_dir, source_config("climatology_station_probability"), layout=layout
        )
        self._layout = source.layout
        self._thresholds = list(source.layout["thresholds"])
        self._cache: dict[str, dict[int, np.ndarray]] = {}
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

    def _load(self, key: str) -> dict[int, np.ndarray]:
        if key not in self._cache:
            try:
                with open_dataset(
                    self._index[key],
                    source_config("climatology_station_probability"),
                    layout=self._layout,
                ).read(all_files=True) as dataset:
                    values = (
                        dataset.probability.isel(cycle_time=0)
                        .transpose("station", "threshold")
                        .values
                    )
                    self._cache[key] = {
                        int(sid): values[i].copy()
                        for i, sid in enumerate(dataset.station.values)
                    }
            except ValueError as exc:
                raise ConfigError(f"ref {key} 的概率表读取失败: {exc}") from exc
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
