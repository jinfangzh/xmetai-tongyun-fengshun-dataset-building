"""Shared evaluation dataset contract, independent of concrete dataset examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from datakit.io import get_layout
from datakit.schema.mapping import VarSpec, load_mapping
from datakit.schema.semantics import DataKind


@dataclass(frozen=True)
class DatasetProfile:
    """一个数据源的变量映射和评估语义约定；文件组织与变量约定由各数据集模块集中声明。"""

    name: str
    io_layout: Mapping[str, Any] = field(default_factory=dict)
    variables: dict[str, VarSpec] = field(default_factory=dict)
    coordinate_aliases: dict[str, str] = field(
        default_factory=lambda: {"latitude": "lat", "longitude": "lon"}
    )
    packed_variable: str | None = None
    packed_dimension: str | None = None
    kind: DataKind = DataKind.GRIDDED_FORECAST
    engine: str = "netcdf"
    #: 默认后端打开失败时的回退后端（Windows 非 ASCII 路径下 netCDF4 会报 ENOENT）
    engine_fallback: str | None = "h5netcdf"
    time_kind: str = "forecast"
    step_hours: float = 6.0
    member_dim: str | None = None
    ensure_member_dim: bool = False
    groups: dict[str, str] = field(default_factory=dict)
    grib_filters: dict[str, dict[str, Any]] = field(default_factory=dict)
    squeeze_dims: tuple[str, ...] = ()
    transpose_dims: tuple[str, ...] = ()
    case_insensitive: bool = False

    @property
    def per_variable_open(self) -> bool:
        """是否需要按变量分别打开文件（cfgrib 需要按 filter_by_keys 过滤）。"""
        return bool(self.grib_filters)

    def groups_for(self, variables: Sequence[str]) -> set:
        """请求变量涉及的文件分组。"""
        return {
            self.groups[name]
            for name in (str(variable).lower() for variable in variables)
            if name in self.groups
        }

    def spec_for(self, standard_name: str) -> VarSpec | None:
        return self.variables.get(str(standard_name))

    def standard_name_for(self, file_variable: str) -> str | None:
        """源变量名 -> 标准名；未声明时返回 None（由调用方按小写兜底）。"""
        for standard_name, spec in self.variables.items():
            if spec.source is None:
                continue
            if spec.source == file_variable:
                return standard_name
            if (
                self.case_insensitive
                and spec.source.lower() == str(file_variable).lower()
            ):
                return standard_name
        return None

    def with_step_hours(self, step_hours: float) -> DatasetProfile:
        return replace(self, step_hours=float(step_hours))


def build_profile(description):
    """Adapt the declaration to the existing evaluation contract."""
    mapping = load_mapping(description.get("mapping", {}))
    contract = deepcopy(description.get("contract", {}))
    if "hours_per_index" in description.get("lead_time", {}):
        contract.setdefault("step_hours", description["lead_time"]["hours_per_index"])
    name = contract.pop("name", description["name"])
    if "kind" in contract:
        contract["kind"] = DataKind(contract["kind"])
    return DatasetProfile(
        name=name,
        io_layout=get_layout(description),
        variables=mapping.variables,
        coordinate_aliases=mapping.coordinate_aliases,
        packed_variable=mapping.packed_variable,
        packed_dimension=mapping.packed_dimension,
        squeeze_dims=tuple(mapping.squeeze_dims),
        transpose_dims=tuple(mapping.transpose_dims),
        member_dim=mapping.member_dim,
        ensure_member_dim=mapping.ensure_member_dim,
        case_insensitive=mapping.case_insensitive,
        **contract,
    )
