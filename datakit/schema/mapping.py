"""Declarative source-variable mappings; numerical operations live in transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from numbers import Real

import numpy as np
import xarray as xr

from .semantics import TemporalKind


@dataclass(frozen=True)
class VarSpec:
    """标准变量的解码声明。

    Attributes:
        source: 文件里的变量名；``None`` 表示"文件里唯一的变量"（如 cfgrib 按过滤器取出的单变量）。
        unit: 换算后的标准单位。
        scale: 数值换算系数，标准值 = 源值 * scale + offset。
        offset: 数值换算偏移。
        source_unit: 换算前的单位，写入 ``attrs["original_units"]``。
        temporal_kind: 该变量的时间语义。
    """

    source: str | None = None
    unit: str = "unknown"
    scale: float = 1.0
    offset: float = 0.0
    source_unit: str = ""
    temporal_kind: TemporalKind = TemporalKind.INSTANTANEOUS

    def __post_init__(self) -> None:
        if self.source is not None and (not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError("Variable source must be a nonempty string or None")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value)
            for value in (self.scale, self.offset)
        ):
            raise ValueError("Variable scale and offset must be finite numbers")
        if not isinstance(self.temporal_kind, TemporalKind):
            raise ValueError("temporal_kind must be a TemporalKind")


@dataclass(frozen=True)
class DatasetMapping:
    """Variable and dimension conventions for converting one dataset."""

    variables: dict[str, VarSpec] = field(default_factory=dict)
    coordinate_aliases: dict[str, str] = field(
        default_factory=lambda: {"latitude": "lat", "longitude": "lon"}
    )
    packed_variable: str | None = None
    packed_dimension: str | None = None
    squeeze_dims: tuple[str, ...] = ()
    transpose_dims: tuple[str, ...] = ()
    member_dim: str | None = None
    ensure_member_dim: bool = False
    case_insensitive: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.variables, Mapping) or any(
            not isinstance(name, str) or not name.strip() or not isinstance(spec, VarSpec)
            for name, spec in self.variables.items()
        ):
            raise ValueError("variables must map nonempty standard names to VarSpec")
        if bool(self.packed_variable) != bool(self.packed_dimension):
            raise ValueError("packed_variable and packed_dimension must be specified together")
        sources = [
            spec.source.lower() if self.case_insensitive else spec.source
            for spec in self.variables.values() if spec.source is not None
        ]
        if len(set(sources)) != len(sources):
            raise ValueError("Each source variable must map to exactly one standard name")

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


def load_mapping(value):
    """Load declarative coordinate/variable mapping from a dict or external YAML."""
    if isinstance(value, DatasetMapping):
        return value
    if isinstance(value, (str, Path)):
        import yaml

        candidate = Path(value).expanduser()
        content = candidate.read_text(encoding="utf-8")
        value = yaml.safe_load(content)
    if not isinstance(value, Mapping):
        raise TypeError(
            "Dataset mapping must be a mapping, DatasetMapping or YAML path"
        )
    if "mapping" in value:
        value = value["mapping"]
        if not isinstance(value, Mapping):
            raise TypeError("mapping must be a mapping")
    elif "layout" in value:
        raise ValueError("Dataset description does not define a mapping")
    fields = dict(value)
    variables = {}
    for name, declaration in fields.pop("variables", {}).items():
        if isinstance(declaration, VarSpec):
            variables[name] = declaration
        elif isinstance(declaration, str):
            variables[name] = VarSpec(source=declaration)
        else:
            spec = dict(declaration)
            if "temporal_kind" in spec:
                spec["temporal_kind"] = TemporalKind(spec["temporal_kind"])
            variables[name] = VarSpec(**spec)
    return DatasetMapping(variables=variables, **fields)


def apply_dataset_mapping(
    dataset: xr.Dataset, mapping: DatasetMapping, variables: Sequence[str]
) -> xr.Dataset | None:
    """坐标别名、变量改名、单位换算、维度规范化。"""
    if mapping.packed_variable and mapping.packed_variable in dataset.data_vars:
        dimension = mapping.packed_dimension
        if dimension not in dataset.coords:
            raise ValueError(
                f"Packed variable {mapping.packed_variable!r} requires {dimension!r} labels"
            )
        labels = np.asarray(dataset[dimension].values)
        if labels.ndim == 2:
            labels = [
                "".join(str(cell) for cell in row).strip()
                if labels.dtype.kind == "U"
                else b"".join(row).decode("utf-8", "replace").strip()
                for row in labels
            ]
        else:
            labels = [
                item.decode("utf-8", "replace").strip()
                if isinstance(item, bytes)
                else str(item).strip()
                for item in labels
            ]
        if len(set(labels)) != len(labels):
            raise ValueError("Packed variable labels must be unique")
        array = dataset[mapping.packed_variable]
        dataset = xr.Dataset(
            {
                name: array.isel({dimension: position}, drop=True)
                for position, name in enumerate(labels)
            },
            attrs=dataset.attrs,
        )
    renames = {
        alias: standard
        for alias, standard in mapping.coordinate_aliases.items()
        if alias in dataset.coords or alias in dataset.dims
    }
    if renames:
        dataset = dataset.rename(renames)

    renames = {}
    for file_variable in dataset.data_vars:
        standard_name = mapping.standard_name_for(str(file_variable))
        if standard_name is None:
            spec = mapping.spec_for(str(file_variable).lower())
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
            if (mapping.spec_for(str(name)) is not None)
            and mapping.spec_for(str(name)).source is None
        ]
        if len(dataset.data_vars) == 1 and unresolved:
            if len(unresolved) != 1:
                raise ValueError("A single source variable cannot resolve multiple unnamed mappings")
            dataset = dataset.rename({next(iter(dataset.data_vars)): unresolved[0]})
            available = [unresolved[0]]
    if not available:
        return None
    dataset = dataset[available]

    for name in available:
        spec = mapping.spec_for(name)
        if spec is None:
            continue
        value = dataset[name]
        if spec.scale != 1.0 or spec.offset != 0.0:
            value = value * spec.scale + spec.offset
        attributes = dict(dataset[name].attrs)
        if spec.unit != "unknown":
            attributes["units"] = spec.unit
        if spec.source_unit:
            attributes["original_units"] = spec.source_unit
        # Copy attributes without changing the source variable or loading its data.
        dataset[name] = value.assign_attrs(attributes)

    if mapping.squeeze_dims:
        singleton = [
            dim
            for dim in mapping.squeeze_dims
            if dim in dataset.dims and dataset.sizes[dim] == 1
        ]
        if singleton:
            dataset = dataset.squeeze(dim=singleton, drop=True)

    if mapping.transpose_dims:
        for name in dataset.data_vars:
            order = [dim for dim in mapping.transpose_dims if dim in dataset[name].dims]
            if order and len(order) == dataset[name].ndim:
                dataset[name] = dataset[name].transpose(*order)

    if (
        mapping.member_dim
        and mapping.ensure_member_dim
        and mapping.member_dim not in dataset.dims
    ):
        dataset = dataset.expand_dims(mapping.member_dim)

    return dataset


def profile_mapping(profile) -> DatasetMapping:
    return DatasetMapping(
        variables=profile.variables,
        coordinate_aliases=profile.coordinate_aliases,
        packed_variable=profile.packed_variable,
        packed_dimension=profile.packed_dimension,
        squeeze_dims=profile.squeeze_dims,
        transpose_dims=profile.transpose_dims,
        member_dim=profile.member_dim,
        ensure_member_dim=profile.ensure_member_dim,
        case_insensitive=profile.case_insensitive,
    )
