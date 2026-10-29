"""Strict standalone YAML dataset-standard loading and source verification."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .specifications import (
    ArrayDefinition,
    AuxiliaryDefinition,
    ChannelDefinition,
    ChannelMetadataDefinition,
    DatasetProduct,
    DatasetSchema,
    DatasetVariant,
    EvidenceReference,
    GridDefinition,
    MaskDefinition,
    QualityRule,
)


class _UniqueSafeLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases and merge inheritance are not supported")
        return super().compose_node(parent, index)


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ValueError(
                f"YAML mapping keys must be strings at line {key_node.start_mark.line + 1}"
            )
        if key in result:
            raise ValueError(
                f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def resolve_config_path(config_path: Path, value: str) -> Path:
    source = Path(_string(value, "path")).expanduser()
    return source if source.is_absolute() else config_path.parent / source


def verify_schema_unchanged(spec: DatasetSchema) -> None:
    if spec.schema_path is None or spec.schema_sha256 is None:
        raise ValueError("Expected a specification loaded from a frozen YAML source")
    if hashlib.sha256(spec.schema_path.read_bytes()).hexdigest() != spec.schema_sha256:
        raise ValueError(f"Schema changed during execution: {spec.schema_path}")


def _object(value, path: str, required=(), optional=()) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected a string-keyed mapping")
    missing, unknown = (
        set(required) - value.keys(),
        value.keys() - set(required) - set(optional),
    )
    if missing or unknown:
        raise ValueError(
            f"{path}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}"
        )
    return value


def _string(value, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: expected a nonempty string")
    return value


def _number(value, path: str, *, positive=False) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(
            f"{path}: expected a finite {'positive ' if positive else ''}number"
        )
    return value


def _integer(value, path: str, *, minimum=1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path}: expected an integer >= {minimum}")
    return value


def _boolean(value, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path}: expected a boolean")
    return value


def _list(value, path: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a list")
    return value


def _strings(value, path: str) -> tuple[str, ...]:
    values = tuple(_string(item, path) for item in _list(value, path))
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{path}: expected nonempty, unique strings")
    return values


def _equal(value, expected, path: str):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{path}: unsupported value {value!r}; expected {expected!r}")
    return value


def _reference(value, source: Path, path: str) -> EvidenceReference:
    item = _object(value, path, ("path", "sha256"))
    return EvidenceReference(
        resolve_config_path(source, _string(item["path"], f"{path}.path")),
        _string(item["sha256"], f"{path}.sha256"),
    )


def _steps(value, path: str, variables: set[str]) -> tuple[dict[str, Any], ...]:
    result = []
    fields = {
        "zscore": ("mean", "std"),
        "multiply": ("factor", "input_units", "output_units"),
        "divide": (
            "divisor",
            "input_units",
            "output_units",
            "accumulation_window_seconds",
        ),
        "clip_min": ("minimum",),
        "log1p": (),
        "hypot": ("inputs",),
    }
    for index, raw in enumerate(_list(value, path)):
        where = f"{path}[{index}]"
        if not isinstance(raw, dict) or raw.get("operation") not in fields:
            raise ValueError(f"{where}: unsupported preprocessing operation")
        operation = raw["operation"]
        item = _object(raw, where, ("operation", *fields[operation]))
        if operation == "zscore":
            _equal(item["mean"], "/mean", where)
            _equal(item["std"], "/std", where)
        elif operation in ("multiply", "divide"):
            key = "factor" if operation == "multiply" else "divisor"
            _number(item[key], f"{where}.{key}", positive=True)
            for key in ("input_units", "output_units"):
                _string(item[key], f"{where}.{key}")
            if operation == "divide":
                window = _number(
                    item["accumulation_window_seconds"], where, positive=True
                )
                if window != item["divisor"]:
                    raise ValueError(
                        f"{where}: accumulation window must match the divisor"
                    )
        elif operation == "clip_min":
            if _number(item["minimum"], where) < 0:
                raise ValueError(
                    f"{where}: log1p clipping requires a nonnegative minimum"
                )
        elif operation == "hypot":
            inputs = _strings(item["inputs"], where)
            if len(inputs) != 2 or set(inputs) - variables:
                raise ValueError(
                    f"{where}: hypot requires two declared source variables"
                )
        result.append(dict(item))
    signature = tuple(item["operation"] for item in result)
    if signature not in (
        ("zscore",),
        ("multiply", "zscore"),
        ("divide", "zscore"),
        ("hypot", "zscore"),
        ("multiply", "clip_min", "log1p"),
    ):
        raise ValueError(
            f"{path}: unsupported ordered preprocessing sequence {signature}"
        )
    return tuple(result)


def _channels(meta, variables: dict[str, Any]) -> tuple[ChannelDefinition, ...]:
    names = _strings(meta["channels"], "meta_info.channels")
    candidates: dict[str, list[tuple[str, str, float | int, str]]] = {}
    for kind in ("pressure", "depth_below_sea_surface"):
        if kind not in meta:
            continue
        levels_key = "levels" if kind == "pressure" else "level_codes"
        group = _object(
            meta[kind],
            f"meta_info.{kind}",
            ("variables", "units", "channel_name_template", levels_key),
        )
        template = (
            "{variable}{level}" if kind == "pressure" else "{variable}{level_code}"
        )
        _equal(
            group["channel_name_template"],
            template,
            f"meta_info.{kind}.channel_name_template",
        )
        units = _string(group["units"], f"meta_info.{kind}.units")
        if kind == "pressure":
            levels = _list(group[levels_key], f"meta_info.{kind}.{levels_key}")
            for level in levels:
                _integer(level, f"meta_info.{kind}.{levels_key}")
            if not levels or len(set(levels)) != len(levels):
                raise ValueError(
                    f"meta_info.{kind}: levels must be nonempty and unique"
                )
            codes = {str(level): level for level in levels}
        else:
            codes = _object(
                group[levels_key],
                f"meta_info.{kind}.{levels_key}",
                optional=group[levels_key]
                if isinstance(group[levels_key], dict)
                else (),
            )
            if not codes:
                raise ValueError(f"meta_info.{kind}: missing depth codes")
            for code, level in codes.items():
                if not code.isdigit() or _number(level, f"meta_info.{kind}.{code}") < 0:
                    raise ValueError(
                        f"meta_info.{kind}: invalid explicit depth mapping"
                    )
        for variable in _strings(group["variables"], f"meta_info.{kind}.variables"):
            if variable not in variables:
                raise ValueError(f"meta_info.{kind}: undefined variable {variable}")
            for code, level in codes.items():
                candidates.setdefault(f"{variable}{code}", []).append(
                    (variable, kind, level, units)
                )
    result, used = [], set()
    required = (
        "units",
        "long_name",
        "source_name",
        "source_units",
        "variable_type",
        "preprocessing",
    )
    optional = ("level_type", "level", "level_units", "unconfirmed_fields")
    for name in names:
        matches = candidates.get(name, [])
        if name in variables and matches or len(matches) > 1:
            raise ValueError(
                f"meta_info.channels: ambiguous variable/level mapping for {name}"
            )
        if matches:
            variable, kind, level, level_units = matches[0]
        elif name in variables:
            variable, kind, level, level_units = name, None, None, None
        else:
            raise ValueError(f"meta_info.channels: undefined channel {name}")
        used.add(variable)
        raw = _object(
            variables[variable], f"variable_metadata.{variable}", required, optional
        )
        _string(raw["long_name"], f"variable_metadata.{variable}.long_name")
        unconfirmed = (
            _strings(
                raw["unconfirmed_fields"],
                f"variable_metadata.{variable}.unconfirmed_fields",
            )
            if "unconfirmed_fields" in raw
            else ()
        )
        if set(unconfirmed) - {"units", "source_units"}:
            raise ValueError(
                f"variable_metadata.{variable}: only units may be unconfirmed"
            )
        for key in ("units", "source_units"):
            if raw[key] is not None or key not in unconfirmed:
                _string(raw[key], f"variable_metadata.{variable}.{key}")
        if kind is not None and set(raw) & {"level", "level_type", "level_units"}:
            raise ValueError(
                f"variable_metadata.{variable}: shared levels cannot be overridden"
            )
        if kind is None:
            kind = _string(
                raw.get("level_type"), f"variable_metadata.{variable}.level_type"
            )
            if kind == "surface":
                if "level" in raw or "level_units" in raw:
                    raise ValueError(
                        f"variable_metadata.{variable}: surface has no level"
                    )
            elif kind in ("height_above_ground", "soil_layer"):
                level = _number(
                    raw.get("level"),
                    f"variable_metadata.{variable}.level",
                    positive=True,
                )
                level_units = _string(
                    raw.get("level_units"), f"variable_metadata.{variable}.level_units"
                )
                if kind == "soil_layer":
                    _integer(level, f"variable_metadata.{variable}.level")
            else:
                raise ValueError(
                    f"variable_metadata.{variable}: unsupported level_type {kind}"
                )
        source = raw["source_name"]
        source = (
            _strings(source, f"variable_metadata.{variable}.source_name")
            if isinstance(source, list)
            else _string(source, f"variable_metadata.{variable}.source_name")
        )
        variable_type = _string(
            raw["variable_type"], f"variable_metadata.{variable}.variable_type"
        )
        if variable_type not in ("instantaneous", "accumulated", "derived", "static"):
            raise ValueError(f"variable_metadata.{variable}: unsupported variable_type")
        result.append(
            ChannelDefinition(
                name,
                variable,
                raw["units"],
                raw["source_units"],
                kind,
                level,
                level_units,
                variable_type=variable_type,
                source_name=source,
                unconfirmed_fields=unconfirmed,
                preprocessing=_steps(
                    raw["preprocessing"],
                    f"variable_metadata.{variable}.preprocessing",
                    set(variables),
                ),
            )
        )
    if used != set(variables):
        raise ValueError(
            f"variable_metadata: unused declarations {sorted(set(variables) - used)}"
        )
    return tuple(result)


def _grid(value) -> GridDefinition:
    grid = _object(
        value,
        "meta_info.grid",
        ("space_resolution", "coordinate_units", "latitude", "longitude"),
    )
    _equal(grid["coordinate_units"], "degree", "meta_info.grid.coordinate_units")
    resolution = _number(
        grid["space_resolution"], "meta_info.grid.space_resolution", positive=True
    )
    axes = {}
    for name in ("latitude", "longitude"):
        item = _object(
            grid[name],
            f"meta_info.grid.{name}",
            ("first", "last", "count", "direction"),
        )
        first = _number(item["first"], f"meta_info.grid.{name}.first")
        last = _number(item["last"], f"meta_info.grid.{name}.last")
        count = _integer(item["count"], f"meta_info.grid.{name}.count")
        direction = _string(item["direction"], f"meta_info.grid.{name}.direction")
        sign = -1 if name == "latitude" and direction == "north-to-south" else 1
        if direction not in (
            ("north-to-south", "south-to-north")
            if name == "latitude"
            else ("west-to-east",)
        ) or not np.isclose(
            first + sign * resolution * (count - 1), last, rtol=0, atol=1e-6
        ):
            raise ValueError(
                f"meta_info.grid.{name}: endpoints/count/resolution/direction disagree"
            )
        axes[name] = item
    lat, lon = axes["latitude"], axes["longitude"]
    return GridDefinition(
        "declared",
        lat["count"],
        lon["count"],
        resolution,
        max(lat["first"], lat["last"]),
        lon["first"],
        lat["direction"],
    )


def _arrays(value) -> dict[str, ArrayDefinition]:
    arrays = _object(
        value,
        "meta_info.arrays",
        (
            "data",
            "time",
            "channel",
            "lat",
            "lon",
            "mean",
            "std",
            "mask",
            "mask_channel",
        ),
    )
    result = {}
    for name, raw in arrays.items():
        item = _object(
            raw, f"meta_info.arrays.{name}", ("dimensions", "dtype"), ("chunks",)
        )
        dimensions = _strings(item["dimensions"], f"meta_info.arrays.{name}.dimensions")
        chunks = None
        if "chunks" in item:
            declared = _object(
                item["chunks"], f"meta_info.arrays.{name}.chunks", dimensions
            )
            chunks = tuple(
                value
                if value == "full"
                else _integer(value, f"meta_info.arrays.{name}.chunks")
                for value in (declared[dim] for dim in dimensions)
            )
        result[name] = ArrayDefinition(
            dimensions,
            _string(item["dtype"], f"meta_info.arrays.{name}.dtype"),
            chunks,
        )
    return result


def _auxiliaries(value) -> tuple[tuple[AuxiliaryDefinition, ...], dict[str, str], str]:
    section = _object(
        value,
        "meta_info.auxiliaries",
        ("check_variable_names", "unregistered_file_status", "files"),
    )
    check_names = _boolean(
        section["check_variable_names"], "meta_info.auxiliaries.check_variable_names"
    )
    files = section["files"]
    files = _object(
        files,
        "meta_info.auxiliaries.files",
        optional=files if isinstance(files, dict) else (),
    )
    result, provenance = [], {}
    for name, raw in files.items():
        where = f"meta_info.auxiliaries.files.{name}"
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(f"{where}: expected a direct-child filename")
        item = _object(
            raw,
            where,
            ("format", "role", "required", "dimensions", "dtype", "grid", "finite"),
            (
                "channels",
                "nonnegative_atol",
                "require_positive_value",
                "grid_provenance",
                "variable",
            ),
        )
        _equal(item["grid"], "main_data", f"{where}.grid")
        if item["format"] not in ("netcdf", "npy") or item["role"] not in (
            "const",
            "weight",
        ):
            raise ValueError(f"{where}: unsupported auxiliary format or role")
        dimensions = tuple(
            _string(dim, f"{where}.dimensions")
            for dim in _list(item["dimensions"], f"{where}.dimensions")
        )
        if not dimensions or any(
            dimensions.count(dim) > 1 for dim in dimensions if dim != "singleton"
        ):
            raise ValueError(f"{where}: only singleton dimensions may repeat")
        channels = (
            _strings(item["channels"], f"{where}.channels")
            if "channels" in item
            else ()
        )
        if set(dimensions) - {"channel", "lat", "lon", "singleton"} or (
            ("channel" in dimensions) != bool(channels)
        ):
            raise ValueError(f"{where}: invalid dimensions/channel declaration")
        dtype = _string(item["dtype"], f"{where}.dtype")
        if np.dtype(dtype).kind not in "fbiu":
            raise ValueError(f"{where}: auxiliaries must have numeric or boolean dtype")
        variable = (
            _string(item["variable"], f"{where}.variable")
            if "variable" in item
            else None
        )
        if check_names and item["format"] == "netcdf" and variable is None:
            raise ValueError(
                f"{where}: checking NetCDF variable names requires variable"
            )
        tolerance = 0.0
        positive = True
        if item["role"] == "weight":
            tolerance = _number(
                item.get("nonnegative_atol"), f"{where}.nonnegative_atol"
            )
            if tolerance < 0:
                raise ValueError(f"{where}: nonnegative tolerance required")
            positive = _boolean(
                item.get("require_positive_value"), f"{where}.require_positive_value"
            )
        elif set(item) & {"nonnegative_atol", "require_positive_value"}:
            raise ValueError(f"{where}: weight constraints require role=weight")
        if "grid_provenance" in item:
            provenance[name] = _string(
                item["grid_provenance"], f"{where}.grid_provenance"
            )
        result.append(
            AuxiliaryDefinition(
                name,
                item["format"],
                variable,
                dimensions,
                dtype,
                item["role"],
                channels,
                required=_boolean(item["required"], f"{where}.required"),
                check_variable_name=check_names,
                weight_nonnegative_atol=tolerance,
                finite=_boolean(item["finite"], f"{where}.finite"),
                require_positive_value=positive,
            )
        )
    return (
        tuple(result),
        provenance,
        _string(
            section["unregistered_file_status"],
            "meta_info.auxiliaries.unregistered_file_status",
        ),
    )


def load_schema(path: Path) -> tuple[DatasetSchema, dict[str, Any]]:
    """Load one complete standard; never infer expectations from inspected data."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"{path}: expected a standalone YAML standard")
    raw = path.read_bytes()
    try:
        root = _object(
            yaml.load(raw.decode("utf-8-sig"), Loader=_UniqueSafeLoader),
            "root",
            ("meta_info", "variable_metadata"),
        )
        required = (
            "name",
            "dataset_id",
            "template_version",
            "expected_root_dataset_id",
            "product",
            "schema_version",
            "status",
            "dataset_path",
            "auxiliary_directory",
            "dataset_version",
            "storage_format",
            "data_revision",
            "validation_scope",
            "time_coverage",
            "grid",
            "channels",
            "arrays",
            "mask",
            "metadata",
            "normalization",
            "auxiliaries",
        )
        optionals = (
            "pressure",
            "depth_below_sea_surface",
            "identity_mapping_evidence",
            "conversion_required",
            "quality",
            "allowed_omitted",
            "mask_provenance",
            "mask_derivation",
            "semantic_confirmation",
            "confirmed_optional_roles",
            "additional_metadata_files",
            "valid_regions",
            "valid_region_provenance",
        )
        meta = _object(root["meta_info"], "meta_info", required, optionals)
        for key in (
            "name",
            "dataset_id",
            "template_version",
            "expected_root_dataset_id",
            "schema_version",
            "status",
            "data_revision",
            "dataset_version",
            "validation_scope",
            "dataset_path",
            "auxiliary_directory",
        ):
            _string(meta[key], f"meta_info.{key}")
        product_info = _object(
            meta["product"], "meta_info.product", ("id", "version", "variant")
        )
        for key in product_info:
            _string(product_info[key], f"meta_info.product.{key}")
        storage = _object(
            meta["storage_format"],
            "meta_info.storage_format",
            ("name", "version", "allow_nested_groups"),
        )
        _equal(storage["name"], "zarr", "meta_info.storage_format.name")
        _equal(storage["version"], 3, "meta_info.storage_format.version")
        _equal(
            storage["allow_nested_groups"],
            False,
            "meta_info.storage_format.allow_nested_groups",
        )
        time = _object(
            meta["time_coverage"],
            "meta_info.time_coverage",
            ("first", "last", "time_count", "time_resolution", "timezone"),
        )
        _equal(time["timezone"], "UTC", "meta_info.time_coverage.timezone")
        for key in ("first", "last"):
            _string(time[key], f"meta_info.time_coverage.{key}")
        count = _integer(time["time_count"], "meta_info.time_coverage.time_count")
        resolution = _object(
            time["time_resolution"],
            "meta_info.time_coverage.time_resolution",
            ("value", "unit"),
        )
        _equal(
            resolution["unit"], "hour", "meta_info.time_coverage.time_resolution.unit"
        )
        cadence = _integer(
            resolution["value"], "meta_info.time_coverage.time_resolution.value"
        )
        if cadence not in (6, 24):
            raise ValueError(
                "meta_info.time_coverage: supported cadence is 6 or 24 hours"
            )
        variables = root["variable_metadata"]
        variables = _object(
            variables,
            "variable_metadata",
            optional=variables if isinstance(variables, dict) else (),
        )
        channels, grid, arrays = (
            _channels(meta, variables),
            _grid(meta["grid"]),
            _arrays(meta["arrays"]),
        )
        mask = _object(
            meta["mask"],
            "meta_info.mask",
            ("channels", "allowed_values", "valid_value", "complement_sum"),
        )
        mask_schema = MaskDefinition(
            dimensions=arrays["mask"].dimensions,
            channel_dimensions=arrays["mask_channel"].dimensions,
            dtype=arrays["mask"].dtype,
            channel_dtype=arrays["mask_channel"].dtype,
            channels=_strings(mask["channels"], "meta_info.mask.channels"),
            allowed_values=tuple(
                _integer(value, "meta_info.mask.allowed_values", minimum=0)
                for value in _list(
                    mask["allowed_values"], "meta_info.mask.allowed_values"
                )
            ),
            valid_value=_integer(
                mask["valid_value"], "meta_info.mask.valid_value", minimum=0
            ),
            complement_sum=_integer(
                mask["complement_sum"], "meta_info.mask.complement_sum", minimum=0
            ),
        )
        metadata = _object(
            meta["metadata"],
            "meta_info.metadata",
            ("data_attributes", "channel_metadata"),
        )
        channel_meta = _object(
            metadata["channel_metadata"],
            "meta_info.metadata.channel_metadata",
            (
                "attribute",
                "version_attribute",
                "version",
                "require_exact_channels",
                "long_name_required",
                "scale_factor_from",
                "add_offset_from",
                "scale_offset_formula",
            ),
        )
        for key in (
            "attribute",
            "version_attribute",
            "version",
            "scale_offset_formula",
        ):
            _string(channel_meta[key], f"meta_info.metadata.channel_metadata.{key}")
        _equal(
            channel_meta["scale_factor_from"],
            "std",
            "meta_info.metadata.channel_metadata.scale_factor_from",
        )
        _equal(
            channel_meta["add_offset_from"],
            "mean",
            "meta_info.metadata.channel_metadata.add_offset_from",
        )
        _equal(
            channel_meta["scale_offset_formula"],
            "preprocessed_value = normalized_value * scale_factor + add_offset",
            "meta_info.metadata.channel_metadata.scale_offset_formula",
        )
        channel_definition = ChannelMetadataDefinition(
            **{
                key: channel_meta[key]
                for key in (
                    "attribute",
                    "version_attribute",
                    "version",
                    "scale_offset_formula",
                )
            },
            require_exact_channels=_boolean(
                channel_meta["require_exact_channels"],
                "meta_info.metadata.channel_metadata.require_exact_channels",
            ),
            long_name_required=_boolean(
                channel_meta["long_name_required"],
                "meta_info.metadata.channel_metadata.long_name_required",
            ),
        )
        data_attributes = _object(
            metadata["data_attributes"],
            "meta_info.metadata.data_attributes",
            (
                "data_representation",
                "normalization_method",
                "mean_variable",
                "std_variable",
                "scale_factor_location",
                "add_offset_location",
                "inverse_transform",
            ),
        )
        for key, expected in (
            ("data_representation", "normalized"),
            ("normalization_method", "channel_dependent"),
            ("mean_variable", "mean"),
            ("std_variable", "std"),
            (
                "inverse_transform",
                "preprocessed_value = data * scale_factor + add_offset",
            ),
            (
                "scale_factor_location",
                f"channel.{channel_definition.attribute}[*].scale_factor",
            ),
            (
                "add_offset_location",
                f"channel.{channel_definition.attribute}[*].add_offset",
            ),
        ):
            _equal(
                data_attributes[key],
                expected,
                f"meta_info.metadata.data_attributes.{key}",
            )
        normalization = _object(
            meta["normalization"],
            "meta_info.normalization",
            ("statistics",),
            ("effective_statistics",),
        )
        statistics = _object(
            normalization["statistics"],
            "meta_info.normalization.statistics",
            ("mean_finite", "std_finite", "std_positive"),
        )
        for key, value in statistics.items():
            _equal(value, True, f"meta_info.normalization.statistics.{key}")
        effective = normalization.get("effective_statistics", {})
        effective = _object(
            effective,
            "meta_info.normalization.effective_statistics",
            optional=effective if isinstance(effective, dict) else (),
        )
        effective_values = {}
        for variable, raw_values in effective.items():
            values = _object(
                raw_values,
                f"meta_info.normalization.effective_statistics.{variable}",
                ("mean", "std"),
            )
            effective_values[variable] = (
                _number(values["mean"], f"effective_statistics.{variable}.mean"),
                _number(
                    values["std"], f"effective_statistics.{variable}.std", positive=True
                ),
            )
        for channel in channels:
            if channel.preprocessing[-1][
                "operation"
            ] != "zscore" and effective_values.get(channel.variable) != (0, 1):
                raise ValueError(
                    f"effective_statistics.{channel.variable}: unnormalized data require mean=0/std=1"
                )
        auxiliaries, provenance, unregistered = _auxiliaries(meta["auxiliaries"])
        product = DatasetProduct(
            product_info["id"],
            product_info["version"],
            channels,
            (DatasetVariant(product_info["variant"], grid, cadence, auxiliaries),),
            semantic_review=tuple(
                f"{variable}: unconfirmed units"
                for variable in sorted(
                    {c.variable for c in channels if c.unconfirmed_fields}
                )
            ),
            mask=mask_schema,
            effective_statistics=effective_values,
        )
        optional_values = {}
        for key in (
            "identity_mapping_evidence",
            "mask_provenance",
            "mask_derivation",
            "semantic_confirmation",
        ):
            if key in meta:
                optional_values[key] = _string(meta[key], f"meta_info.{key}")
        if "conversion_required" in meta:
            optional_values["conversion_required"] = _boolean(
                meta["conversion_required"],
                "meta_info.conversion_required",
            )
        if "quality" in meta:
            rules = []
            for raw_rule in _list(meta["quality"], "meta_info.quality"):
                rule = dict(
                    _object(
                        raw_rule,
                        "meta_info.quality",
                        (
                            "channel",
                            "valid_region",
                            "max_nan_fraction",
                            "allow_constant",
                            "evidence",
                        ),
                        ("bounds", "domain", "max_block_nan_fraction"),
                    )
                )
                if rule.get("bounds") is not None:
                    rule["bounds"] = tuple(
                        _list(rule["bounds"], "meta_info.quality.bounds")
                    )
                rules.append(QualityRule(**rule))
            optional_values["quality"] = tuple(rules)
        if "allowed_omitted" in meta:
            optional_values["allowed_omitted"] = tuple(
                tuple(
                    _integer(i, "meta_info.allowed_omitted", minimum=0)
                    for i in _list(index, "meta_info.allowed_omitted")
                )
                for index in _list(meta["allowed_omitted"], "meta_info.allowed_omitted")
            )
        if "additional_metadata_files" in meta:
            optional_values["additional_metadata_files"] = _strings(
                meta["additional_metadata_files"], "meta_info.additional_metadata_files"
            )
        for key in ("confirmed_optional_roles", "valid_region_provenance"):
            if key in meta:
                mapping = _object(
                    meta[key],
                    f"meta_info.{key}",
                    optional=meta[key] if isinstance(meta[key], dict) else (),
                )
                optional_values[key] = {
                    name: _string(value, f"meta_info.{key}.{name}")
                    for name, value in mapping.items()
                }
        if "valid_regions" in meta:
            mapping = _object(
                meta["valid_regions"],
                "meta_info.valid_regions",
                optional=meta["valid_regions"]
                if isinstance(meta["valid_regions"], dict)
                else (),
            )
            optional_values["valid_regions"] = {
                name: _reference(value, path, f"meta_info.valid_regions.{name}")
                for name, value in mapping.items()
            }
        spec = DatasetSchema(
            meta["expected_root_dataset_id"],
            product,
            product_info["variant"],
            time["first"],
            time["last"],
            meta["data_revision"],
            template_version=meta["template_version"],
            schema_version=meta["schema_version"],
            content_version=meta["dataset_version"],
            zarr_format=storage["version"],
            directory_name=Path(meta["dataset_path"]).name,
            auxiliary_provenance=provenance,
            validation_scope=meta["validation_scope"],
            config_id=meta["dataset_id"],
            schema_path=path,
            schema_sha256=hashlib.sha256(raw).hexdigest(),
            schema_status=meta["status"],
            arrays=arrays,
            data_attributes=dict(data_attributes),
            channel_metadata=channel_definition,
            unregistered_auxiliary_status=unregistered,
            **optional_values,
        )
        spec.validate_configuration()
        if spec.expected_shape[0] != count:
            raise ValueError(
                "meta_info.time_coverage.time_count differs from independently generated coverage"
            )
        if (
            any(c.unconfirmed_fields for c in channels)
            and spec.validation_scope != "confirmed-format"
        ):
            raise ValueError(
                "Unconfirmed units are only supported in confirmed-format scope"
            )
        document = dict(meta)
        for key in ("dataset_path", "auxiliary_directory"):
            document[key] = str(resolve_config_path(path, meta[key]))
        return spec, document
    except (yaml.YAMLError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
