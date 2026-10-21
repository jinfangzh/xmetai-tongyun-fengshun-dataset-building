# -*- coding: utf-8 -*-
"""Discover gridded source files and decode them into standard xarray datasets."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import xarray as xr

from datakit.io.readers.errors import SourceDecodeError, SourceDiscoveryError
from datakit.io.readers.hdf5 import hdf5_guard
from datakit.io.readers.layouts import GriddedLayout

COORDINATE_ALIASES = {"latitude": "lat", "longitude": "lon"}

@dataclass(frozen=True)
class GriddedRecord:
    """一条"文件 <-> 时刻/时效/分组"的记录。"""

    path: Path
    group: Optional[str] = None
    member: Optional[Any] = None
    init_time: Optional[datetime] = None
    lead_time: Optional[float] = None

    @property
    def valid_time(self) -> Optional[datetime]:
        if self.init_time is None or self.lead_time is None:
            return self.init_time
        return self.init_time + timedelta(hours=float(self.lead_time))


class GriddedFileCatalog:
    """按布局声明扫描文件的通用目录。"""

    def __init__(self, root_dir: Path, layout: GriddedLayout):
        self.root_dir = Path(root_dir)
        self.layout = layout

    def discover_records(self, request: Any) -> List[GriddedRecord]:
        if not self.root_dir.exists():
            raise SourceDiscoveryError(
                f"Root directory does not exist: {self.root_dir}",
                source_id=request.source_id,
            )
        if not request.init_times:
            raise SourceDiscoveryError(
                f"{self.layout.name} catalog requires init_times in request",
                source_id=request.source_id,
            )

        needed_groups = self.layout.groups_for(request.variables)
        records: List[GriddedRecord] = []
        for moment in request.init_times:
            date_dir = Path(self.layout.root_template.format(root=self.root_dir, init=moment))
            if not date_dir.exists():
                continue
            records.extend(self._scan_dir(date_dir, moment, needed_groups, request))

        if not records:
            raise SourceDiscoveryError(
                f"No {self.layout.name} files found in {self.root_dir} "
                f"for requested times and variables",
                source_id=request.source_id,
            )
        return records

    def _scan_dir(
        self,
        date_dir: Path,
        moment: datetime,
        needed_groups: set,
        request: Any,
    ) -> List[GriddedRecord]:
        records: List[GriddedRecord] = []
        for member, directory in self._iter_member_dirs(date_dir):
            matched: List[GriddedRecord] = []
            for path in sorted(item for item in directory.iterdir() if item.is_file()):
                parsed = self._parse_name(path, moment)
                if parsed is None:
                    continue
                file_time, lead, group = parsed
                if file_time != moment:
                    continue
                if needed_groups and group not in needed_groups:
                    continue
                matched.append(
                    GriddedRecord(
                        path=path,
                        group=group,
                        member=member,
                        init_time=file_time,
                        lead_time=lead,
                    )
                )

            if self.layout.lead_from == "index":
                matched = [
                    replace(record, lead_time=(position + 1) * self.layout.step_hours)
                    for position, record in enumerate(matched)
                ]

            for record in matched:
                if (
                    record.lead_time is not None
                    and request.lead_times
                    and record.lead_time not in request.lead_times
                ):
                    continue
                records.append(record)
        return records

    def _iter_member_dirs(self, date_dir: Path) -> Iterator[Tuple[Any, Path]]:
        """产出 (成员标识, 目录)；没有成员层时产出 (None, 起报目录)。"""
        if not self.layout.member_glob:
            yield None, date_dir
            return
        directories = [
            path for path in date_dir.glob(self.layout.member_glob) if path.is_dir()
        ]
        for path in sorted(directories, key=self._member_sort_key):
            yield self._member_value(path), path

    @staticmethod
    def _member_value(path: Path) -> Any:
        """成员目录 -> 成员标识；默认取目录名里的数字。"""
        match = re.search(r"(\d+)", path.name)
        return int(match.group(1)) if match else path.name

    def _member_sort_key(self, path: Path) -> Tuple[int, Any]:
        value = self._member_value(path)
        if isinstance(value, int):
            return (0, value)
        return (1, str(value))

    def _parse_name(
        self, path: Path, moment: datetime
    ) -> Optional[Tuple[datetime, Optional[float], Optional[str]]]:
        """解析文件名；没有 init 命名组时，时刻取所在目录代表的时刻。"""
        for pattern in self.layout.patterns:
            match = re.fullmatch(pattern.regex, path.name)
            if match is None:
                continue
            groups = match.groupdict()
            init_raw = groups.get("init")
            file_time = (
                datetime.strptime(init_raw, self.layout.init_format) if init_raw else moment
            )
            group = pattern.group or groups.get("group")
            lead: Optional[float] = None
            if self.layout.lead_from == "filename" and self.layout.time_kind == "forecast":
                lead_raw = groups.get("lead")
                if lead_raw is None:
                    continue
                lead = float(lead_raw)
            return file_time, lead, group
        return None

class GriddedDatasetReader:
    """Decode and normalize source records without evaluation result contracts."""

    def __init__(self, source_id: str, layout: GriddedLayout):
        self.source_id = source_id
        self.layout = layout
        self.step_hours = layout.step_hours

    def read_records(
        self, records: Sequence[GriddedRecord], variables: Sequence[str]
    ) -> Tuple[xr.Dataset, List[Tuple[GriddedRecord, xr.Dataset]]]:
        return self._read_records(records, variables)

    def _read_records(
        self, records: Sequence[GriddedRecord], variables: Sequence[str]
    ) -> Tuple[xr.Dataset, List[Tuple[GriddedRecord, xr.Dataset]]]:
        self._warn_on_member_mismatch(records)
        frames: List[Tuple[GriddedRecord, xr.Dataset]] = []
        for record in records:
            dataset = self._read_record(record, variables)
            if dataset is not None:
                frames.append((record, dataset))
        if not frames:
            raise SourceDecodeError(
                f"None of requested variables {list(variables)} found in "
                f"{len(records)} files",
                source_id=self.source_id,
            )
        return self._combine(frames), frames

    def _warn_on_member_mismatch(self, records: Sequence[GriddedRecord]) -> None:
        """成员数在不同起报之间不一致时给出警告（避免静默少样本）。"""
        if not self.layout.member_glob:
            return
        counts: Dict[Optional[datetime], set] = {}
        for record in records:
            counts.setdefault(record.init_time, set()).add(record.member)
        sizes = {moment: len(members) for moment, members in counts.items()}
        if len(set(sizes.values())) > 1:
            warnings.warn(
                f"各起报的集合成员数不一致: "
                f"{ {str(k): v for k, v in sorted(sizes.items())} }；"
                f"缺失成员会以缺测参与平均"
            )

    def _read_record(
        self, record: GriddedRecord, variables: Sequence[str]
    ) -> Optional[xr.Dataset]:
        # HDF5 后端不是线程安全的，threads 形态下并发读会段错误（见 base.hdf5_guard），
        # 所以文件访问排队；锁只包这一层，归一化/换算在锁外照常并发。
        with hdf5_guard(self.layout.engine, self.layout.engine_fallback):
            return self._read_record_unlocked(record, variables)

    def _read_record_unlocked(
        self, record: GriddedRecord, variables: Sequence[str]
    ) -> Optional[xr.Dataset]:
        if self.layout.per_variable_open:
            datasets: List[xr.Dataset] = []
            for variable in variables:
                filters = self.layout.grib_filters.get(str(variable))
                if filters is None:
                    continue
                try:
                    with self._open(record.path, filters) as source:
                        dataset = source.load()
                except SourceDecodeError:
                    raise
                except Exception as exc:
                    warnings.warn(f"Failed to read {variable} from {record.path}: {exc}")
                    continue
                normalized = self._normalize(dataset, [str(variable)])
                if normalized is not None:
                    datasets.append(normalized)
            if not datasets:
                return None
            return datasets[0] if len(datasets) == 1 else xr.merge(datasets)

        try:
            with self._open(record.path, None) as source:
                # 先在**惰性**数据集上做改名/挑选/换算，再 load——顺序反过来
                # （先 load 再挑）会把文件里的变量**全部**读进内存，哪怕只请求了
                # 其中几个。集合预报一个文件十来个要素，51 成员 × 4 时效 = 204 个
                # 文件，差一个要素就是几个 GB 的峰值。
                dataset = self._normalize(source, variables)
                if dataset is not None:
                    dataset = dataset.load()
        except SourceDecodeError:
            raise
        except Exception as exc:
            raise SourceDecodeError(
                f"Failed to read {record.path}: {exc}",
                source_id=self.source_id,
                path=str(record.path),
                cause=exc,
            ) from exc
        return dataset

    def _open(self, path: Path, filter_by_keys: Optional[Dict[str, Any]]):
        if self.layout.engine == "cfgrib":
            return xr.open_dataset(
                path, engine="cfgrib", backend_kwargs={"filter_by_keys": filter_by_keys}
            )
        try:
            return xr.open_dataset(path)
        except OSError as exc:
            # 非 ASCII 路径下 netCDF4 的 C 层会报 ENOENT（文件其实存在），
            # 回退到 h5netcdf（h5py 走 UTF-8 路径）再试一次。
            fallback = self.layout.engine_fallback
            if not fallback:
                raise
            try:
                dataset = xr.open_dataset(path, engine=fallback)
            except Exception:
                raise exc
            warnings.warn(
                f"默认后端打开失败（{exc}），已回退到 engine={fallback}: {path}"
            )
            return dataset

    def _normalize(self, dataset: xr.Dataset, variables: Sequence[str]) -> Optional[xr.Dataset]:
        """坐标别名、变量改名、单位换算、维度规范化。"""
        dataset = self._standardize_coordinates(dataset)

        renames = {}
        for file_variable in dataset.data_vars:
            standard_name = self.layout.standard_name_for(str(file_variable))
            if standard_name is None:
                spec = self.layout.spec_for(str(file_variable).lower())
                if spec is not None and spec.source is None:
                    continue  # 单变量文件，稍后按请求改名
                standard_name = str(file_variable).lower()
            if standard_name != file_variable:
                renames[file_variable] = standard_name
        if renames:
            dataset = dataset.rename(renames)

        available = [str(name) for name in variables if name in dataset.data_vars]
        if not available:
            unresolved = [
                str(name)
                for name in variables
                if (self.layout.spec_for(str(name)) is not None)
                and self.layout.spec_for(str(name)).source is None
            ]
            if len(dataset.data_vars) == 1 and unresolved:
                dataset = dataset.rename({list(dataset.data_vars)[0]: unresolved[0]})
                available = [unresolved[0]]
        if not available:
            return None
        dataset = dataset[available]

        for name in available:
            spec = self.layout.spec_for(name)
            if spec is None:
                continue
            if spec.scale != 1.0 or spec.offset != 0.0:
                dataset[name] = dataset[name] * spec.scale + spec.offset
                dataset[name].attrs["units"] = spec.unit
                if spec.source_unit:
                    dataset[name].attrs["original_units"] = spec.source_unit

        if self.layout.squeeze_dims:
            singleton = [
                dim
                for dim in self.layout.squeeze_dims
                if dim in dataset.dims and dataset.sizes[dim] == 1
            ]
            if singleton:
                dataset = dataset.squeeze(dim=singleton, drop=True)

        if self.layout.transpose_dims:
            for name in dataset.data_vars:
                order = [dim for dim in self.layout.transpose_dims if dim in dataset[name].dims]
                if order and len(order) == dataset[name].ndim:
                    dataset[name] = dataset[name].transpose(*order)

        if (
            self.layout.member_dim
            and self.layout.ensure_member_dim
            and self.layout.member_dim not in dataset.dims
        ):
            dataset = dataset.expand_dims(self.layout.member_dim)

        return dataset

    def _standardize_coordinates(self, dataset: xr.Dataset) -> xr.Dataset:
        renames = {
            alias: standard
            for alias, standard in COORDINATE_ALIASES.items()
            if alias in dataset.coords or alias in dataset.dims
        }
        return dataset.rename(renames) if renames else dataset

    def _combine(self, frames: Sequence[Tuple[GriddedRecord, xr.Dataset]]) -> xr.Dataset:
        """按时间语义拼接：预报拼 init_time x lead_time，观测拼 valid_time。"""
        if self.layout.time_kind == "forecast":
            by_init: Dict[datetime, Dict[Optional[float], Dict[Any, List[xr.Dataset]]]] = {}
            for record, dataset in frames:
                by_init.setdefault(record.init_time, {}).setdefault(
                    record.lead_time, {}
                ).setdefault(record.member, []).append(dataset)

            init_datasets = []
            for init_time in sorted(by_init):
                lead_datasets = []
                for lead_time in sorted(by_init[init_time], key=lambda value: (value is None, value)):
                    merged = self._merge_members(by_init[init_time][lead_time])
                    lead_datasets.append(merged.expand_dims(lead_time=[lead_time]))
                # coords/compat 双双写死：xarray 的弃用预告成对出现，只写一个它会
                # 接着问下一个（详见 matcher._add_valid_time 处的说明）。
                # 两个值都是当前的默认值，写死 = 行为不变。
                combined = xr.concat(
                    lead_datasets, dim="lead_time", coords="different", compat="equals"
                )
                init_datasets.append(combined.expand_dims(init_time=[init_time]))
            return xr.concat(
                init_datasets, dim="init_time", coords="different", compat="equals"
            )

        by_time: Dict[datetime, List[xr.Dataset]] = {}
        for record, dataset in frames:
            by_time.setdefault(record.valid_time, []).append(dataset)
        time_datasets = []
        for moment in sorted(by_time):
            parts = by_time[moment]
            merged = parts[0] if len(parts) == 1 else xr.merge(parts)
            time_datasets.append(merged.expand_dims(valid_time=[moment]))
        return xr.concat(
            time_datasets, dim="valid_time", coords="different", compat="equals"
        )

    def _merge_members(self, by_member: Dict[Any, List[xr.Dataset]]) -> xr.Dataset:
        """同一 (init, lead) 的多个成员拼成 member 维；同一成员的多个文件合并。"""
        members = sorted(by_member, key=lambda value: (value is None, value))
        merged_parts = []
        for member in members:
            parts = by_member[member]
            merged_parts.append(parts[0] if len(parts) == 1 else xr.merge(parts))
        if len(merged_parts) == 1:
            return merged_parts[0]

        name = self.layout.member_coord
        if name in merged_parts[0].dims:
            # 成员已经在文件内的维度里（如 Fengqing 内置 member 维）
            return merged_parts[0] if len(merged_parts) == 1 else xr.merge(merged_parts)
        stacked = xr.concat(
            merged_parts, dim=name, coords="different", compat="equals"
        )
        return stacked.assign_coords({name: list(members)})

