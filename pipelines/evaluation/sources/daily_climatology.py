"""Sample, smooth and package daily climatology for evaluation."""

import calendar
import logging
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from datakit import io
from datakit.evaluation.contracts import DataIndex, DataRequest
from datakit.evaluation.errors import DecodeError, DiscoveryError
from datakit.io.backends.files import NETCDF_READ_LOCK
from datakit.schema.mapping import (
    DatasetMapping,
    VarSpec,
    apply_dataset_mapping,
    load_mapping,
)
from datakit.schema.semantics import DataKind

from ._bundle import build_bundle

log = logging.getLogger(__name__)

DEFAULT_SMOOTH_DAYS = 15

#: time:units 的单位 -> 小时换算系数
_HOURS_PER_UNIT = {
    "hour": 1.0,
    "hours": 1.0,
    "hr": 1.0,
    "h": 1.0,
    "day": 24.0,
    "days": 24.0,
    "d": 24.0,
    "minute": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "min": 1.0 / 60.0,
    "second": 1.0 / 3600.0,
    "seconds": 1.0 / 3600.0,
    "s": 1.0 / 3600.0,
}


def _hours_per_unit(units: Any, path: Path) -> float:
    """从 ``time:units`` 里取出单位对应的小时系数（忽略起算时刻）。"""
    text = str(units or "").strip().lower()
    unit = text.partition(" since ")[0].strip()
    factor = _HOURS_PER_UNIT.get(unit)
    if factor is None:
        raise ValueError(
            f"{path} 的 time 轴单位无法识别（需要 days / hours / minutes / seconds，"
            f"可带 ' since <时刻>'，实际是 {units!r}）"
        )
    return factor


def _day_of_year(moment: datetime) -> int:
    """365 天口径的 0-based 日序：闰年 d>=59 减 1，把 2/29 并到 2/28。"""
    day = moment.timetuple().tm_yday - 1
    if calendar.isleap(moment.year) and day >= 59:
        day -= 1
    return day


def _day_count(moment: datetime) -> int:
    """366 天口径的 0-based 日序：平年 d>=59 加 1，跳过 2/29 那一格。"""
    day = moment.timetuple().tm_yday - 1
    if not calendar.isleap(moment.year) and day >= 59:
        day += 1
    return day


def _day_fraction(moment: datetime) -> float:
    """当天已过的比例（时/分/秒），起报时次因此参与日序插值。"""
    return (moment.hour * 3600 + moment.minute * 60 + moment.second) / 86400.0


class _DailyClimatologySampler:
    """将已打开的日序气候态转换成 ``(valid_time, lat, lon)`` 的参考场。

    按传入的有效时刻取场，日序索引 + 线性插值，
    取场前先对年内日序做 ``smooth_days`` 天的环形滑动平均（年内首尾相接）。

    ``smooth_days`` 是气象参数，不是 IO 参数：它只决定"气候态数值被抹平多少"，
    与 ``execution.loads`` 的 ``window:W``（窗块滚动缓存）毫无关系。取 1 = 不平滑
    （逐日原值，噪声大），越大越平滑（季节循环也被抹），0 或负数等同不平滑。
    """

    def __init__(
        self,
        smooth_days: int = DEFAULT_SMOOTH_DAYS,
        mapping=None,
        chunk_points: int = 200_000_000,
    ):
        #: 环形平滑天数（取场前对年内日序做中心对齐的滑动平均）
        self.smooth_days = int(smooth_days)
        self.mapping = (
            mapping
            if mapping is not None
            else DatasetMapping(packed_variable="data", packed_dimension="level")
        )
        #: 单次读取的点数预算。取到 200M（f4 约 800MB）是为了让常见的几十步窗口
        #: **一次读完整个纬度范围**——纬度分块每多一块，就多一遍对同一批时间块的
        #: 重复解压。这里的默认值只在时间窗口大到装不下时才会真的切块。
        self.chunk_points = int(chunk_points)

    # -- 时间轴 -------------------------------------------------------------

    def _time_hours(self, dataset: xr.Dataset, path: Path) -> np.ndarray:
        """时间轴 -> 各时刻对应的小时数（0 = 年内第 0 天 00:00）。"""
        if "time" not in dataset.coords:
            raise ValueError(f"{path} 没有 time 坐标，无法做日序索引")
        coord = dataset["time"]
        values = np.asarray(coord.values)
        if np.issubdtype(values.dtype, np.datetime64):
            return self._hours_from_datetime(values, path)
        factor = _hours_per_unit(coord.attrs.get("units"), path)
        return np.asarray(values, dtype="f8") * factor

    @staticmethod
    def _hours_from_datetime(values: np.ndarray, path: Path) -> np.ndarray:
        """datetime64 时间轴 -> 距首年 1/1 00:00 的小时数。

        合成年气候态把年内日序写成某一年（如 1990）的连续时刻，日序信息完整编码
        在时刻里，锚到该年 1/1 就与数值轴同口径。

        允许跨年：366 天口径从平年 1/1 起算时，第 366 天必然落到次年 1/1
        （如 ``era5_clim_phys_14.nc`` 的 1990-01-01 … 1991-01-01 18:00），
        那是正常的，不是错误。
        """
        moments = values.astype("datetime64[h]").astype("int64")
        first_year = int(values[0].astype("datetime64[Y]").astype("int64")) + 1970
        start = np.datetime64(f"{first_year:04d}-01-01", "h").astype("int64")
        hours = (moments - start).astype("f8")
        if hours[0] != 0.0:
            raise ValueError(
                f"{path} 的 time 从 {values[0]} 起算，不是所在年的 1/1；"
                "日序气候态必须从年初开始，否则日序会对错格"
            )
        return hours

    def _steps_per_day(self, hours: np.ndarray, path: Path) -> int:
        if hours.size < 2:
            raise ValueError(f"{path} 的时间轴至少要有 2 个时次才能判断步长")
        steps = np.diff(hours)
        if not np.allclose(steps, steps[0], atol=1e-6):
            raise ValueError(f"{path} 的时间轴步长不均匀，无法做日序索引")
        step = float(steps[0])
        if step <= 0:
            raise ValueError(f"{path} 的时间轴不是递增的")
        per_day = 24.0 / step
        if abs(per_day - round(per_day)) > 1e-6 or round(per_day) < 1:
            raise ValueError(
                f"{path} 的时间轴步长 {step}h 不能整除 24 小时（每天 {per_day} 步）"
            )
        return round(per_day)

    def _year_length(self, hours: np.ndarray, steps_per_day: int, path: Path) -> int:
        total = hours.size / steps_per_day
        days = round(total)
        if abs(total - days) > 1e-6 or days not in (365, 366):
            raise ValueError(
                f"{path} 的时间轴折合 {total} 天，只支持 365 天（闰年 2/29 并到 2/28）"
                f"或 366 天（含 2/29）两种约定"
            )
        return days

    # -- 变量场 -------------------------------------------------------------

    def _field(self, dataset: xr.Dataset, name: str) -> xr.DataArray | None:
        """取一个变量的 ``(time, lat, lon)`` 惰性视图（不立即载入）。"""
        if name not in dataset.data_vars:
            return None
        field = dataset[name]

        drop = [
            dim
            for dim in field.dims
            if dim not in ("time", "lat", "lon") and field.sizes[dim] == 1
        ]
        if drop:
            field = field.squeeze(dim=drop, drop=True)
        if not {"time", "lat", "lon"}.issubset(field.dims):
            raise ValueError(
                f"变量 {name} 的维度是 {field.dims}，无法规范成 (time, lat, lon)"
            )
        return field.transpose("time", "lat", "lon")

    # -- 平滑 ---------------------------------------------------------------

    def _smooth(self, slab: np.ndarray, half: int) -> np.ndarray:
        """沿时间轴做环形滑动平均（年内首尾相接）。"""
        if half <= 0:
            return slab
        ksize = 2 * half + 1
        padded = np.concatenate([slab[-half:], slab, slab[:half]], axis=0)
        cumulative = np.cumsum(padded, axis=0, dtype="f8")
        cumulative = np.concatenate(
            [np.zeros((1,) + cumulative.shape[1:], dtype="f8"), cumulative], axis=0
        )
        return ((cumulative[ksize:] - cumulative[:-ksize]) / ksize).astype("f4")

    # -- 转换接口 --------------------------------------------------------

    def apply(self, dataset, variables, times, source_name="climatology"):
        """Sample and smooth an already opened dataset, reading only needed slices."""
        from tqdm import tqdm

        path = Path(source_name)  # diagnostic label only; no path access
        times = sorted(times)
        if not times:
            raise ValueError("日序气候态需要显式给出要取的有效时刻")
        structural = replace(
            self.mapping,
            variables={
                name: replace(spec, scale=1.0, offset=0.0)
                for name, spec in self.mapping.variables.items()
            },
        )
        dataset = apply_dataset_mapping(dataset, structural, variables)
        if dataset is None:
            raise ValueError(f"气候态 {path} 没有请求的变量 {variables}")
        hours = self._time_hours(dataset, path)
        steps_per_day = self._steps_per_day(hours, path)
        year_days = self._year_length(hours, steps_per_day, path)
        total_steps = hours.size
        # smooth_days 天 -> 步数 -> 半径 half：实际窗宽是 2*half+1 步，
        # 中心对齐（half 步前 ~ half 步后），跨年首尾相接。
        # 例：6h 步长（steps_per_day=4）、smooth_days=15 -> 60 步 -> half=30
        #     -> 每个日序取前后各 30 步（7.5 天）共 61 步的算术平均。
        half = int((self.smooth_days * steps_per_day) // 2)
        half = max(0, min(half, (total_steps - 1) // 2))

        available = sorted(str(name) for name in dataset.data_vars)
        wanted_names = [str(name) for name in variables]
        resolved = [name for name in wanted_names if name in available]
        missing = [name for name in wanted_names if name not in available]
        if missing:
            log.warning("气候态里没有这些变量，已跳过: %s", ", ".join(missing))
        if not resolved:
            raise ValueError(
                f"气候态 {path} 里没有请求的变量 {wanted_names}；文件里有: {available}"
            )

        # 每个有效时刻 -> (i0, i1, 权重)，i0/i1 是年内的整数步号
        picks = [
            self._pick(moment, steps_per_day, total_steps, year_days)
            for moment in times
        ]
        wanted_steps = sorted({step for pick in picks for step in (pick[0], pick[1])})
        log.info(
            "气候态 %s: %d 个有效时刻，涉及 %d 个年内步号（全年 %d 步），"
            "平滑窗 ±%d 步，逐要素读取 %d 个",
            path.name,
            len(times),
            len(wanted_steps),
            total_steps,
            half,
            len(resolved),
        )

        lat = np.asarray(dataset["lat"].values, dtype="f8")
        lon = np.asarray(dataset["lon"].values, dtype="f8")
        fields: dict[str, xr.DataArray] = {}
        units: dict[str, str] = {}

        for order, name in enumerate(
            tqdm(
                resolved,
                desc=f"气候态 {path.name}",
                unit="要素",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ),
            start=1,
        ):
            field = self._field(dataset, name)
            if field is None:
                continue
            started = time.perf_counter()
            values = self._gather(field, wanted_steps, picks, half, len(lat), len(lon))
            log.info(
                "气候态要素 %d/%d: %s (%.1fs)",
                order,
                len(resolved),
                name,
                time.perf_counter() - started,
            )
            spec = self.mapping.spec_for(name)
            scale = spec.scale if spec else 1.0
            if scale != 1.0:
                values = values * scale
            if spec and spec.offset != 0.0:
                values = values + spec.offset
            unit = spec.unit if spec else "unknown"
            fields[name] = xr.DataArray(
                values,
                dims=("valid_time", "lat", "lon"),
                coords={"valid_time": times, "lat": lat, "lon": lon},
                name=name,
                attrs={"units": unit},
            )
            units[name] = unit

        if not fields:
            raise ValueError(f"气候态 {path} 没有取到任何请求的变量")
        return xr.Dataset(fields)

    def _pick(
        self, moment: datetime, steps_per_day: int, total_steps: int, year_days: int
    ) -> tuple[int, int, float]:
        """有效时刻 -> (前一步号, 后一步号, 后一步权重)，首尾环形相接。"""
        day = _day_of_year(moment) if year_days == 365 else _day_count(moment)
        position = (day + _day_fraction(moment)) * steps_per_day
        floor = int(np.floor(position))
        weight = position - floor
        lower = floor % total_steps
        upper = (lower + 1) % total_steps
        return lower, upper, float(weight)

    def _gather(
        self,
        field: xr.DataArray,
        wanted_steps: Sequence[int],
        picks: Sequence[tuple[int, int, float]],
        half: int,
        lat_size: int,
        lon_size: int,
    ) -> np.ndarray:
        """算出所有有效时刻的场。

        **只读 ``wanted_steps`` 前后各 ``half`` 步，不是整条时间轴。** 一年 1464
        步里真正用到的通常只有几十步；而 NetCDF/HDF5 基本都沿时间切块，读整条
        时间轴 = 把整份文件解压一遍。乘以纬度分块数（早先 20M 点的分块预算是
        9 行/块，721 行就是 81 块）就是几十次全量解压——几十 TB 的重复 IO，
        在日志上就是"卡死不动"。

        参考实现（``vfc/climo.py`` 的 ``read_slice`` / ``_rolling_raw``）同样是
        只读窗口内的行区间。

        ``half`` 步的余量是给平滑窗留的：``rows`` 按环形展开，取场的位置都在
        块内部，离块边至少 ``half`` 步，所以 ``_smooth`` 在块边界上的环形假设
        不会影响任何一个取到的点。
        """
        total = int(field.sizes["time"])
        wanted = np.asarray(wanted_steps, dtype=int)

        # 需要读的原始行：选中行各向两侧扩 half 步，按环形区间 [low, high] 展开。
        # 拆成至多两段**连续递增**行号，而不是一个取模后的整数数组：非单调索引
        # 会让后端退化成"读 min..max 再挑"，越年界那次照样把整年读一遍。
        low = int(wanted.min()) - half
        high = int(wanted.max()) + half
        if high - low + 1 >= total:
            segments = [(0, total - 1)]
        else:
            begin = low % total
            span = high - low + 1
            if begin + span <= total:
                segments = [(begin, begin + span - 1)]
            else:
                segments = [(begin, total - 1), (0, (begin + span - 1) % total)]
        rows = np.concatenate([np.arange(a, b + 1, dtype=int) for a, b in segments])
        spot = {int(step): order for order, step in enumerate(rows)}

        lower = np.asarray([spot[int(pick[0])] for pick in picks], dtype=int)
        upper = np.asarray([spot[int(pick[1])] for pick in picks], dtype=int)
        weights = np.asarray([pick[2] for pick in picks], dtype="f4")[:, None, None]

        out = np.empty((len(picks), lat_size, lon_size), dtype="f4")
        rows_per_chunk = max(1, self.chunk_points // max(1, rows.size * lon_size))
        for start in range(0, lat_size, rows_per_chunk):
            stop = min(start + rows_per_chunk, lat_size)
            pieces = [
                np.asarray(
                    field.isel(time=slice(a, b + 1), lat=slice(start, stop)).values,
                    dtype="f4",
                )
                for a, b in segments
            ]
            slab = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
            smoothed = self._smooth(slab, half)
            out[:, start:stop, :] = (1.0 - weights) * smoothed[
                lower
            ] + weights * smoothed[upper]
        return out


class DailyClimatologyCatalog:
    """日序气候态的"发现"：这份数据就是一个文件，所以只做路径解析与存在性校验。

    ``root_dir`` 可以是那个 .nc 文件本身，也可以是不含其它 .nc 的目录。
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    def discover(self, request: DataRequest) -> DataIndex:
        target = self.root_dir
        if target.is_file():
            return DataIndex(
                source_id=request.source_id, available=[{"path": str(target)}]
            )
        if not target.exists():
            raise DiscoveryError(
                f"气候态路径不存在: {target}", source_id=request.source_id
            )
        matches = sorted(target.glob("*.nc"))
        if not matches:
            raise DiscoveryError(
                f"{target} 里没有 .nc 文件", source_id=request.source_id
            )
        if len(matches) > 1:
            raise DiscoveryError(
                f"{target} 里有 {len(matches)} 个 .nc 文件，无法确定用哪个: "
                + ", ".join(item.name for item in matches[:5]),
                source_id=request.source_id,
            )
        return DataIndex(
            source_id=request.source_id,
            available=[{"path": str(matches[0])}],
        )


class DailyClimatologySource:
    def __init__(
        self,
        source_id="daily_climatology",
        version="1.0.0",
        smooth_days=DEFAULT_SMOOTH_DAYS,
        scales=None,
        units=None,
        chunk_points=200_000_000,
        mapping=None,
    ):
        self.source_id = source_id
        self.version = version
        selected = (
            load_mapping(mapping)
            if mapping is not None
            else DatasetMapping(packed_variable="data", packed_dimension="level")
        )
        variables = dict(selected.variables)
        for name in set(scales or {}) | set(units or {}):
            spec = variables.get(name, VarSpec(source=name))
            variables[name] = replace(
                spec,
                scale=float((scales or {}).get(name, spec.scale)),
                unit=(units or {}).get(name, spec.unit),
            )
        selected = replace(selected, variables=variables)
        self.sampler = _DailyClimatologySampler(
            smooth_days=smooth_days, mapping=selected, chunk_points=chunk_points
        )

    def read(self, request, index):
        if not request.init_times:
            raise DecodeError(
                "日序气候态需要显式给出要取的有效时刻（request.init_times）"
            )
        path = Path(index.available[0]["path"])
        try:
            with NETCDF_READ_LOCK, io.open(path).read() as dataset:
                payload = self.sampler.apply(
                    dataset,
                    request.variables,
                    request.init_times,
                    source_name=str(path),
                )
        except ValueError as exc:
            raise DecodeError(str(exc)) from exc
        return build_bundle(
            request,
            payload,
            [str(path)],
            "daily_climatology",
            self.version,
            kind=DataKind.REFERENCE,
        )
