"""Structural coordinate checks and assembly of decoded file partitions."""

from datetime import datetime

import numpy as np
import xarray as xr

from ._layout import get_layout
from .records import FileRecord


_COORDINATES = ("cycle_time", "time", "member", "lead_time", "init_time", "valid_time")
_PARTITION_AXES = {
    "init_time",
    "lead_time",
    "member",
    "time",
    "valid_time",
    "channel",
    "level",
    "cycle_time",
}


def combine_partitions(records, datasets, description=None):
    """Combine supplied partitions without loading arrays or owning their handles.

    Records and datasets correspond by position. File existence is not checked;
    inputs may already be loaded. The caller keeps lazy sources alive until the
    result has been consumed. Only declared structural rules are applied.
    """
    records = list(records)
    frames = list(datasets)
    if not records or len(records) != len(frames):
        raise ValueError("Provide one Dataset for each non-empty file record")
    if not all(isinstance(record, FileRecord) for record in records):
        raise TypeError("records must contain FileRecord objects")
    if not all(isinstance(dataset, xr.Dataset) for dataset in frames):
        raise TypeError("datasets must contain xarray.Dataset objects")
    description = get_layout(description) if description is not None else {}
    datasets, manifest = [], []
    for record, original in zip(records, frames):
        dataset = original.copy(deep=False)
        member_dimension = description.get("member", {}).get("dimension")
        if member_dimension and member_dimension not in dataset.dims:
            raise ValueError(
                f"Expected internal member dimension {member_dimension!r}: {record.path}"
            )
        original_dims = dict(dataset.sizes)
        # Only explicitly declared singleton record axes may be removed.
        squeezed = []
        for dim in description.get("structure", {}).get("record_dimensions", []):
            if dim in dataset.dims:
                if dataset.sizes[dim] != 1:
                    raise ValueError(
                        f"Expected singleton record dimension {dim!r}: {record.path}"
                    )
                # Keep its value as an auxiliary coordinate, not a guessed time.
                dataset = dataset.squeeze(dim, drop=False)
                squeezed.append(dim)
        added, expanded = [], []
        for key in _COORDINATES:
            if key not in record.fields:
                continue
            value = record.fields[key]
            if isinstance(value, datetime):
                value = np.datetime64(value, "ns")
            if key in dataset.coords:
                existing = dataset[key].values
                if existing.size != 1 or not np.all(existing == value):
                    raise ValueError(
                        f"Path and internal {key!r} disagree: {record.path}"
                    )
                if key not in dataset.dims:
                    dataset = dataset.expand_dims(key)
                    expanded.append(key)
            elif key in dataset.dims:
                raise ValueError(f"Cannot replace an unlabeled dimension {key!r}")
            else:
                dataset = dataset.expand_dims({key: [value]})
                added.append(key)
        for dimension, index in dataset.indexes.items():
            if not index.is_unique:
                raise ValueError(f"Duplicate {dimension!r} labels in {record.path}")
        datasets.append(dataset)
        manifest.append(
            {
                "fields": record.fields,
                "variables": list(original.data_vars),
                "dimensions": original_dims,
                "added": added,
                "expanded": expanded,
                "squeezed": squeezed,
                "indexes": {dim: original[dim].values for dim in original.indexes},
                "variable_dims": {
                    name: field.dims for name, field in original.data_vars.items()
                },
            }
        )
    result = _combine(datasets)
    result.encoding["dataset_io_manifest"] = manifest
    result.set_close(None)
    return result


def _combine(frames):
    """检查坐标和重叠记录，先合并变量分组，再组合分区轴。"""
    if len(frames) == 1:
        return frames[0].copy(deep=False)
    # 分区轴允许扩展；空间等非分区坐标必须一致，IO 不隐式插值。
    dimensions = set().union(*(dataset.dims for dataset in frames))
    for dimension in dimensions - _PARTITION_AXES:
        indexes = [
            dataset.indexes[dimension]
            for dataset in frames
            if dimension in dataset.indexes
        ]
        if indexes and any(not indexes[0].equals(index) for index in indexes[1:]):
            raise ValueError(
                f"Files have different {dimension!r} coordinates; regridding is an explicit later step"
            )
    # 只检查坐标标签，避免读取数据或展开空间网格。
    partitions = {}
    for dataset in frames:
        for name, variable in dataset.data_vars.items():
            axes = tuple(
                sorted(set(variable.dims) & _PARTITION_AXES & dataset.indexes.keys())
            )
            previous_axes, records = partitions.setdefault(name, (axes, []))
            if axes != previous_axes:
                raise ValueError(f"Inconsistent partition dimensions for {name!r}")
            records.append({axis: dataset.indexes[axis] for axis in axes})
    for name, (axes, records) in partitions.items():
        if not axes:
            if len(records) > 1:
                raise ValueError(f"Overlapping file records for variable {name!r}")
            continue
        primary = next(
            (
                axis
                for axis in ("time", "valid_time", "init_time", "cycle_time")
                if axis in axes
            ),
            axes[0],
        )
        records.sort(key=lambda record: record[primary].min())
        active = []
        for record in records:
            lower = record[primary].min()
            active = [other for other in active if other[primary].max() >= lower]
            if any(
                all(not record[axis].intersection(other[axis]).empty for axis in axes)
                for other in active
            ):
                raise ValueError(f"Overlapping file records for variable {name!r}")
            active.append(record)
    # First merge same-grid variable groups; then combine partitions by coordinates.
    buckets = {}
    for dataset in frames:
        key = tuple(
            (str(name), tuple(index)) for name, index in sorted(dataset.indexes.items())
        )
        buckets.setdefault(key, []).append(dataset)
    merged = []
    for bucket in buckets.values():
        dataset = bucket[0]
        if len(bucket) > 1:
            dataset = xr.merge(
                bucket,
                join="exact",
                compat="no_conflicts",
                combine_attrs="drop_conflicts",
            )
        merged.append(dataset)
    if len(merged) == 1:
        return merged[0].copy(deep=False)
    return xr.combine_by_coords(
        merged,
        join="exact",
        data_vars="all",
        coords="different",
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
    )

