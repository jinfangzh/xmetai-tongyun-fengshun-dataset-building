"""按描述定位文件、解析路径字段、组合数据，并按布局拆分写回。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import ExitStack
from glob import escape
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._layout import templates
from ..combination import combine_partitions
from ..records import FileRecord
from . import _text_collection, files
from ._writing import (
    check_outputs, check_source_coordinates, check_source_variables,
    project_coordinates, write_mode,
)
from .files import open_xarray, write_file


def discover(
    root: Path, description: Mapping[str, Any], *, path_fields=None
) -> list[FileRecord]:
    """Discover records, optionally narrowing known path fields before listing."""
    layout = description["layout"]
    path_fields = path_fields or {}
    patterns = templates(description)
    if not root.exists():
        raise FileNotFoundError(root)
    if not layout and not root.is_file() and not _is_store(root):
        raise ValueError(
            "A directory of files requires a dataset description; pass a file or Zarr store for direct access"
        )
    if not patterns and layout.get("directory") and layout.get("pattern"):
        from .._layout import PathTemplate

        template = PathTemplate(layout["directory"] + "/" + layout["pattern"])
        template.regex = re.compile(
            template.regex.pattern.replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
        )
        patterns = [(None, template)]
    if not patterns:
        if root.is_file() or _is_store(root):
            paths = [root]
        else:
            paths = sorted(root.glob(layout.get("pattern", "*")))
        records = [
            FileRecord(path, {}) for path in paths if path.is_file() or _is_store(path)
        ]
        if not records:
            raise FileNotFoundError("No files match the dataset description")
        return records
    records = []
    seen = set()
    for group, template in patterns:
        bound = list(
            dict.fromkeys(
                field for _, field, _, _ in template.parts if field in path_fields
            )
        )
        globs = set()
        for values in product(*(path_fields[field] for field in bound)):
            fields = dict(zip(bound, values))
            parts = []
            for literal, field, spec, _ in template.parts:
                parts.append(literal)
                if field in fields:
                    value = fields[field]
                    if "%" not in spec and spec.endswith("d"):
                        integer = int(value)
                        if integer != value:
                            raise ValueError(
                                f"Path field {field!r} requires an integer value, got {value!r}"
                            )
                        value = integer
                    parts.append(escape(format(value, spec)))
                elif field:
                    parts.append("*")
            globs.add("".join(parts))
        candidates = (
            [root]
            if root.is_file()
            else sorted(
                {
                    path
                    for pattern in globs
                    for path in (
                        root.glob(pattern)
                        if any(char in pattern for char in "*?[")
                        else [root / pattern]
                    )
                }
            )
        )
        for path in candidates:
            if not (path.is_file() or _is_store(path)):
                continue
            relative = (
                path.name if root.is_file() else path.relative_to(root).as_posix()
            )
            fields = template.parse(relative)
            if fields is None and root.is_file():
                # A single supplied file may use a directory-bearing template.
                for count in range(2, len(path.parts) + 1):
                    fields = template.parse("/".join(path.parts[-count:]))
                    if fields is not None:
                        break
            if fields is None:
                continue
            if group is not None:
                fields["group"] = group
            if "groups" in layout and fields.get("group") not in layout["groups"]:
                continue
            if path in seen:
                raise ValueError(f"A file matches multiple layout entries: {path}")
            seen.add(path)
            if "lead_index" in fields:
                step = description.get("lead_time", {}).get("hours_per_index")
                if not isinstance(step, (int, float)) or step <= 0:
                    raise ValueError(
                        "lead_index requires positive lead_time.hours_per_index"
                    )
                fields["lead_time"] = fields["lead_index"] * step
            records.append(FileRecord(path, fields))
    if not records:
        raise FileNotFoundError("No files match the dataset description")
    return records


def _is_store(path):
    return path.is_dir() and any(
        (path / name).is_file() for name in ("zarr.json", ".zgroup")
    )


def read_records(records, description, **options):
    """读取已发现的文件记录，供统一入口和逐文件处理复用。"""
    if description.get("format", {}).get("name") in _text_collection.FORMATS:
        return _text_collection.read_records(records, description, **options)
    if not description["layout"] and len(records) == 1 and not records[0].fields:
        return files.read_records(records, description, **options)
    from .files import declared_open_options

    options = declared_open_options(description, options)
    load = options.pop("load_partitions", False)
    options.setdefault("chunks", None if load or len(records) == 1 else {})
    if options["chunks"] is None and len(records) > 1 and not load:
        raise ValueError(
            "Multi-file reads require chunks for bounded loading; use iter_datasets() for eager partitions"
        )
    stack = ExitStack()
    frames = []
    try:
        for record in records:
            if load:
                with open_xarray(record.path, **options) as source:
                    original = source.load()
            else:
                original = stack.enter_context(open_xarray(record.path, **options))
            frames.append(original)
        result = combine_partitions(records, frames, description)
        result.set_close(stack.close)
        return result
    except BaseException:
        stack.close()
        raise


def _output_records(dataset, description, patterns):
    """从读取清单或坐标生成写出计划，不执行写入。"""
    manifest = dataset.encoding.get("dataset_io_manifest")
    if not patterns:
        raise NotImplementedError(
            "A discovery pattern cannot define output names; supply a complete filename template"
        )
    fields = {field for _, template in patterns for field, _, _ in template.fields}
    # A fixed filename explicitly writes one current Dataset, without source routing.
    if len(patterns) == 1 and not fields and not description.get("structure", {}).get("record_dimensions"):
        yield {"fields": {}, "variables": list(dataset.data_vars), "dimensions": {}, "added": []}
        return
    if manifest and all(fields.issubset(record["fields"]) for record in manifest):
        yield from manifest
        return
    needs_manifest = len(patterns) > 1 or bool(
        fields & {"group", "variable", "file_variable", "date", "month"}
    )
    if needs_manifest:
        if not manifest:
            raise ValueError(
                "This output layout requires file grouping metadata from read(); arbitrary Xarray data needs an explicit output description"
            )
        missing = set().union(*(fields - record["fields"].keys() for record in manifest))
        if missing:
            raise ValueError(f"Output layout fields have no source assignment: {sorted(missing)}")
        yield from manifest
        return
    dimensions = [
        "lead_time" if field == "lead_index" else field for field in sorted(fields)
    ]
    for dim in dimensions:
        if dim not in dataset.coords or dataset[dim].ndim > 1:
            raise ValueError(
                f"Output requires a scalar or one-dimensional coordinate {dim!r}"
            )
    for values in product(*(np.atleast_1d(dataset[dim].values) for dim in dimensions)):
        record = {}
        for dimension, value in zip(dimensions, values):
            if isinstance(value, np.datetime64):
                value = pd.Timestamp(value).to_pydatetime()
            elif isinstance(value, np.generic):
                value = value.item()
            record[dimension] = value
        if "lead_index" in fields:
            step = description["lead_time"]["hours_per_index"]
            index = record["lead_time"] / step
            if index < 0 or not float(index).is_integer():
                raise ValueError(
                    "lead_time cannot be represented by this filename index"
                )
            record["lead_index"] = int(index)
        yield {
            "fields": record,
            "variables": list(dataset.data_vars),
            "dimensions": {},
            "added": dimensions,
        }


def _record_selectors(record):
    selectors = dict(record.get("indexes", {}))
    for dim in (*record["added"], *record.get("expanded", [])):
        selectors[dim] = [record["fields"][dim]]
    return selectors


def _project_record(dataset, record):
    variables = [name for name in dataset.data_vars if name in record["variables"]]
    if not variables:
        return None
    part = dataset[variables]
    original_dimensions = set().union(*(
        set(record.get("variable_dims", {}).get(name, part[name].dims))
        for name in variables
    ))
    selectors = {}
    for dim in (*record["added"], *record.get("expanded", [])):
        selectors[dim] = [record["fields"][dim]]
    selectors.update({
        dim: labels for dim, labels in record.get("indexes", {}).items()
        if dim in original_dimensions or dim in part.dims or dim in part.coords
    })
    scalar_coordinates = (
        set(record.get("squeezed", []))
        - set(record["added"])
        - set(record.get("expanded", []))
    )
    part = project_coordinates(part, selectors, scalar_coordinates=scalar_coordinates)
    if part is None:
        return None
    for dim in record["added"]:
        if dim in part.dims:
            part = part.isel({dim: 0}, drop=False)
        if dim in part.coords:
            part = part.drop_vars(dim)
    for dim in record.get("expanded", []):
        if dim in part.dims:
            part = part.isel({dim: 0}, drop=False)
    for name in variables:
        if name not in record.get("variable_dims", {}):
            continue
        dimensions = record["variable_dims"][name]
        value = part[name]
        if set(value.dims) - set(dimensions):
            raise ValueError(f"Cannot restore source dimensions for {name!r}; use an explicit output layout")
        for dim in dimensions:
            if dim not in value.dims:
                if dim in part.coords and part[dim].ndim == 0:
                    value = value.expand_dims(dim)
                elif record["dimensions"][dim] == 1 and dim not in record.get("indexes", {}):
                    value = value.expand_dims(dim)
                else:
                    raise ValueError(f"Cannot restore source dimension {dim!r} for {name!r}")
        part[name] = value.transpose(*dimensions)
    for dim in record["dimensions"]:
        if dim not in part.dims and dim in part.coords and part[dim].ndim == 0:
            part = part.assign_coords({dim: part[dim].expand_dims(dim)})
    return part


def write(dataset, root, description, **options):
    """按当前布局写出 Dataset，返回输出文件路径列表。"""
    if description.get("format", {}).get("name") in _text_collection.FORMATS:
        return _text_collection.write(dataset, root, description, **options)
    if not description["layout"]:
        return files.write(dataset, root, description, **options)
    mode = write_mode(options)
    patterns = templates(description)
    plans = []
    planned_paths = set()
    records = list(_output_records(dataset, description, patterns))
    check_source_variables(dataset, (name for record in records for name in record["variables"]))
    scalar_coordinates = {dim for record in records for dim in record.get("squeezed", [])}
    check_source_coordinates(
        dataset,
        (_record_selectors(record) for record in records),
        scalar_coordinates=scalar_coordinates,
    )
    for record in records:
        part = _project_record(dataset, record)
        if part is None:
            continue
        member_dimension = description.get("member", {}).get("dimension")
        if member_dimension and member_dimension not in part.dims:
            raise ValueError(f"Output layout requires member dimension {member_dimension!r}")
        fields = record["fields"]
        pattern = next(
            (
                template
                for group, template in patterns
                if group is None or group == fields.get("group")
            ),
            None,
        )
        if pattern is None:
            raise ValueError(f"No output template for {fields}")
        path = root / pattern.render(fields)
        if path in planned_paths:
            raise ValueError(f"Output paths collide: {path}")
        planned_paths.add(path)
        plans.append((path, part))
    from .files import declared_open_options

    if not plans:
        raise ValueError("No data remains for the output layout")
    check_outputs([path for path, _ in plans], mode)
    options["mode"] = mode
    return [
        write_file(part, path, declared_open_options(description, options))
        for path, part in plans
    ]
