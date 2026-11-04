"""Read-only layout, coordinate and declared-semantic checks."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from datakit.schema.specifications import (
    AuxiliaryDefinition,
    DatasetSchema,
)

from .check_result import CheckResult, add_check, require_check

STANDARD = "standard_conformance"


def check_declared_root(
    spec: DatasetSchema,
    *,
    expected: set[str],
    undeclared: list[str],
    undeclared_auxiliary: list[str],
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    add_check(
        checks,
        "layout.declared_root_objects",
        "fail"
        if set(undeclared) - set(undeclared_auxiliary)
        else spec.unregistered_auxiliary_status
        if undeclared_auxiliary
        else "pass",
        STANDARD,
        sorted(expected),
        undeclared,
        "Unregistered NC/NPY auxiliaries are potential issues, not acceptance blockers; undeclared main arrays/other objects remain failures",
    )
    return checks


def check_declared_auxiliaries(
    spec: DatasetSchema, unexpected_auxiliary: list[str]
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    add_check(
        checks,
        "auxiliary.declared_files",
        spec.unregistered_auxiliary_status if unexpected_auxiliary else "pass",
        STANDARD,
        sorted((definition.filename for definition in spec.variant.auxiliaries)),
        unexpected_auxiliary,
        "Unregistered auxiliaries were not used as approved roles and need separate review",
    )
    return checks


def check_region_layout(
    region: np.ndarray, name: str, spec: DatasetSchema
) -> tuple[bool, list[CheckResult]]:
    checks: list[CheckResult] = []
    valid = (
        isinstance(region, np.ndarray)
        and region.shape == spec.variant.grid.shape
        and (region.dtype == np.dtype("bool"))
    )
    return (
        require_check(
            checks,
            f"quality.region.{name}.layout",
            valid,
            "data_quality",
            (spec.variant.grid.shape, "bool"),
            (getattr(region, "shape", None), str(getattr(region, "dtype", None))),
            spec.valid_region_provenance[name],
        ),
        checks,
    )


def _semantic_equal(observed: Any, expected: Any) -> bool:
    """JSON numbers may be int/float, but booleans are never coefficients."""
    if isinstance(expected, Real) and not isinstance(expected, (bool, np.bool_)):
        return (
            isinstance(observed, Real)
            and not isinstance(observed, (bool, np.bool_))
            and observed == expected
        )
    if isinstance(expected, dict):
        return (
            isinstance(observed, dict)
            and set(observed) == set(expected)
            and all(_semantic_equal(observed[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(_semantic_equal(o, e) for o, e in zip(observed, expected))
        )
    return type(observed) == type(expected) and observed == expected


def check_dataset_format(
    dataset: xr.Dataset, spec: DatasetSchema
) -> tuple[bool | None, list[CheckResult]]:
    """Return channel validity, or None when raw layout blocks further checks.

    Input must retain raw numeric dtypes (no CF fill masking or scale/offset).
    Only coordinates, mean/std and the two spatial masks are materialized.
    """
    checks: list[CheckResult] = []
    layouts = spec.array_layouts()
    usable = True
    for name, (dims, expected_shape, dtype) in layouts.items():
        if name not in dataset.variables:
            add_check(
                checks,
                f"layout.{name}",
                "fail",
                STANDARD,
                (dims, expected_shape, dtype),
                "missing",
            )
            usable = False
            continue
        array = dataset[name]
        dtype_ok = (
            array.dtype.kind in "UST"
            if dtype == "string"
            else array.dtype == np.dtype(dtype)
        )
        usable = (
            require_check(
                checks,
                f"layout.{name}",
                tuple(array.dims) == dims
                and array.shape == expected_shape
                and dtype_ok,
                STANDARD,
                (dims, expected_shape, dtype),
                (array.dims, array.shape, str(array.dtype)),
            )
            and usable
        )
    require_check(
        checks,
        "layout.declared_arrays",
        set(dataset.variables) == set(layouts),
        STANDARD,
        sorted(layouts),
        sorted(dataset.variables),
    )
    lat, lon = spec.variant.grid.coordinates()
    root_expected = {
        "dataset_id": spec.dataset_id,
        "schema_version": spec.schema_version,
        "content_version": spec.content_version,
        "data_revision": spec.data_revision,
        "geospatial_lat_range": [float(lat.min()), float(lat.max())],
        "geospatial_lon_range": [float(lon[0]), float(lon[-1])],
    }
    for key, value in root_expected.items():
        require_check(
            checks,
            f"attributes.root.{key}",
            _semantic_equal(dataset.attrs.get(key), value),
            STANDARD,
            value,
            dataset.attrs.get(key),
        )
    if "data" in dataset:
        for key, value in spec.data_attributes.items():
            require_check(
                checks,
                f"attributes.data.{key}",
                dataset.data.attrs.get(key) == value,
                STANDARD,
                value,
                dataset.data.attrs.get(key),
            )
    if not usable:
        add_check(
            checks,
            "semantics.prerequisites",
            "blocked",
            STANDARD,
            evidence="Invalid raw layout; semantic checks were not coerced",
        )
        return (None, checks)
    for name, expected in (("lat", lat), ("lon", lon)):
        values = dataset[name].values
        require_check(
            checks,
            f"coordinates.{name}.complete",
            np.array_equal(values, expected),
            STANDARD,
            {
                "count": len(expected),
                "first": float(expected[0]),
                "last": float(expected[-1]),
            },
            {
                "count": len(values),
                "first": float(values[0]),
                "last": float(values[-1]),
                "mismatched_indices": np.flatnonzero(values != expected)[:20],
            },
        )
    time = dataset.time.values
    expected = spec.expected_times()
    valid = (
        not np.isnat(time).any()
        and len(np.unique(time)) == len(time)
        and bool(np.all(np.diff(time) > np.timedelta64(0, "ns")))
    )
    require_check(
        checks,
        "time.complete",
        valid and np.array_equal(time, expected),
        STANDARD,
        {
            "count": len(expected),
            "first": str(expected[0]),
            "last": str(expected[-1]),
            "cadence_hours": spec.variant.cadence_hours,
        },
        {
            "count": len(time),
            "missing": [str(t) for t in np.setdiff1d(expected, time)[:20]],
            "extra": [str(t) for t in np.setdiff1d(time, expected)[:20]],
            "mismatched_indices": np.flatnonzero(time != expected)[:20],
            "unique_increasing": valid,
        },
    )
    require_check(
        checks,
        "time.utc",
        dataset.time.attrs.get("timezone") == "UTC",
        STANDARD,
        "UTC",
        dataset.time.attrs.get("timezone"),
    )
    channels = tuple((str(value) for value in dataset.channel.values))
    channel_ok = require_check(
        checks,
        "channels.exact_order",
        channels == spec.product.channel_names,
        STANDARD,
        spec.product.channel_names,
        channels,
    )
    return (channel_ok, checks)


def check_mask_labels(dataset: xr.Dataset, spec: DatasetSchema) -> list[CheckResult]:
    checks: list[CheckResult] = []
    mask_schema = spec.product.mask
    mask_channels = dataset[mask_schema.channel_array_name].values
    require_check(
        checks,
        "mask.labels",
        tuple((str(v) for v in mask_channels)) == mask_schema.channels,
        STANDARD,
        mask_schema.channels,
        mask_channels,
    )
    return checks


def check_channel_semantics(
    dataset: xr.Dataset, spec: DatasetSchema, *, channel_ok: bool, stats_ok: bool
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    channel_metadata = spec.channel_metadata
    mean, std = (dataset["mean"].values, dataset["std"].values)
    info = dataset.channel.attrs.get(channel_metadata.attribute)
    info_ok = require_check(
        checks,
        "semantics.channel_info",
        isinstance(info, dict)
        and (
            set(info) == set(spec.product.channel_names)
            if channel_metadata.require_exact_channels
            else set(spec.product.channel_names).issubset(info)
        )
        and all((isinstance(v, dict) for v in info.values())),
        STANDARD,
        spec.product.channel_names,
        sorted(info) if isinstance(info, dict) else info,
    )
    require_check(
        checks,
        "semantics.info_version",
        dataset.channel.attrs.get(channel_metadata.version_attribute)
        == channel_metadata.version,
        STANDARD,
        channel_metadata.version,
        dataset.channel.attrs.get(channel_metadata.version_attribute),
    )
    if channel_ok and info_ok and stats_ok:
        for index, definition in enumerate(spec.product.channels):
            observed = info[definition.name]
            fields = definition.semantic_fields()
            if spec.validation_scope == "confirmed-format":
                for key in definition.unconfirmed_fields:
                    fields.pop(key, None)
                if definition.unconfirmed_fields:
                    add_check(
                        checks,
                        f"semantics.{definition.name}.excluded_fields",
                        "not-applicable",
                        STANDARD,
                        evidence={
                            "outside_validation_scope": definition.unconfirmed_fields
                        },
                    )
            if channel_metadata.long_name_required:
                require_check(
                    checks,
                    f"semantics.{definition.name}.description",
                    isinstance(observed.get("long_name"), str)
                    and bool(observed["long_name"].strip()),
                    STANDARD,
                    "nonempty long_name",
                    observed.get("long_name"),
                )
            fields.update(
                scale_factor=float(std[index]),
                add_offset=float(mean[index]),
                scale_offset_formula=channel_metadata.scale_offset_formula,
            )
            mismatches = {
                key: {"expected": value, "observed": observed.get(key)}
                for key, value in fields.items()
                if not _semantic_equal(observed.get(key), value)
            }
            require_check(
                checks,
                f"semantics.{definition.name}",
                not mismatches,
                STANDARD,
                fields,
                mismatches,
            )
            expected_statistics = spec.product.effective_statistics.get(
                definition.variable
            )
            if expected_statistics is not None:
                require_check(
                    checks,
                    f"statistics.{definition.variable}.identity",
                    (mean[index], std[index]) == expected_statistics,
                    STANDARD,
                    expected_statistics,
                    [mean[index], std[index]],
                )
    return checks


def check_auxiliary_format(
    obj: np.ndarray | xr.Dataset, definition: AuxiliaryDefinition, spec: DatasetSchema
) -> tuple[np.ndarray | None, list[CheckResult]]:
    """Validate the header before materializing an auxiliary's values."""
    checks: list[CheckResult] = []
    prefix = f"auxiliary.{definition.filename}"
    lat, lon = spec.variant.grid.coordinates()
    sizes = {
        "lat": len(lat),
        "lon": len(lon),
        "singleton": 1,
        "channel": len(definition.channels),
    }
    expected_shape = tuple((sizes[dim] for dim in definition.dimensions))
    if isinstance(obj, xr.Dataset):
        if definition.check_variable_name:
            if not require_check(
                checks,
                f"{prefix}.variable",
                definition.variable in obj.data_vars,
                STANDARD,
                definition.variable,
                sorted(obj.data_vars),
            ):
                return (None, checks)
            variable = definition.variable
        else:
            add_check(
                checks,
                f"{prefix}.variable_name",
                "not-applicable",
                STANDARD,
                observed=sorted(obj.data_vars),
                evidence="Auxiliary variable names are outside the current approved rule scope",
            )
            candidates = [
                name
                for name, value in obj.data_vars.items()
                if tuple(value.dims) == definition.dimensions
                and value.shape == expected_shape
                and (value.dtype == np.dtype(definition.dtype))
            ]
            if len(obj.data_vars) == 1:
                variable = next(iter(obj.data_vars))
            elif len(candidates) == 1:
                variable = candidates[0]
            elif not obj.data_vars:
                add_check(
                    checks, f"{prefix}.array_present", "fail", STANDARD, observed=[]
                )
                return (None, checks)
            else:
                add_check(
                    checks,
                    f"{prefix}.variable_selection",
                    "warning",
                    STANDARD,
                    observed=sorted(obj.data_vars),
                    evidence="Multiple auxiliary variables have no unique declared layout; value checks deferred, no variable guessed",
                )
                return (None, checks)
        array = obj[variable]
        add_check(
            checks,
            f"{prefix}.additional_variables",
            "pass" if set(obj.data_vars) == {variable} else "warning",
            STANDARD,
            observed=sorted(obj.data_vars),
            evidence={"checked_variable": variable},
        )
        dims_ok = tuple(array.dims) == definition.dimensions
        header_ok = require_check(
            checks,
            f"{prefix}.header_layout",
            dims_ok
            and array.shape == expected_shape
            and (array.dtype == np.dtype(definition.dtype)),
            STANDARD,
            (definition.dimensions, expected_shape, definition.dtype),
            (array.dims, array.shape, str(array.dtype)),
        )
        if not header_ok:
            add_check(
                checks,
                f"{prefix}.values",
                "blocked",
                "data_quality",
                evidence="Invalid auxiliary header; no unbounded materialization",
            )
            return (None, checks)
        for dim, expected in (("lat", lat), ("lon", lon)):
            if dim in array.dims:
                require_check(
                    checks,
                    f"{prefix}.{dim}",
                    dim in obj.coords and np.array_equal(obj[dim].values, expected),
                    STANDARD,
                    expected,
                    obj[dim].values if dim in obj.coords else "missing",
                )
        if definition.channels:
            require_check(
                checks,
                f"{prefix}.channels",
                "channel" in obj.coords
                and tuple((str(v) for v in obj.channel.values)) == definition.channels,
                STANDARD,
                definition.channels,
                obj.channel.values if "channel" in obj else None,
            )
        values = array.values
    else:
        values, dims_ok = (obj, True)
    layout = require_check(
        checks,
        f"{prefix}.layout",
        dims_ok
        and values.shape == expected_shape
        and (values.dtype == np.dtype(definition.dtype)),
        STANDARD,
        (definition.dimensions, expected_shape, definition.dtype),
        (values.shape, str(values.dtype)),
    )
    if not layout:
        add_check(
            checks,
            f"{prefix}.values",
            "blocked",
            "data_quality",
            evidence="Invalid raw auxiliary shape/dtype",
        )
        return (None, checks)
    return (values, checks)


def collect_structure(array: xr.DataArray) -> dict[str, object]:
    """Return a serializable structure summary."""

    return {
        "dims": list(array.dims),
        "sizes": {name: int(size) for name, size in array.sizes.items()},
    }


def check_dimensions(
    obj: xr.Dataset | xr.DataArray,
    *,
    required_dims: Sequence[str] = ("time", "channel", "lat", "lon"),
) -> list[CheckResult]:
    """Check required dimensions without reading the main array."""
    checks: list[CheckResult] = []
    for dim in required_dims:
        present = dim in obj.dims
        require_check(
            checks,
            f"dimensions.{dim}.present",
            present,
            STANDARD,
            "required dimension",
            list(obj.dims),
            None if present else f"Missing required dimension: {dim!r}",
        )
        if present:
            size = obj.sizes[dim]
            require_check(
                checks,
                f"dimensions.{dim}.nonempty",
                size > 0,
                STANDARD,
                "positive length",
                size,
                None if size > 0 else f"Dimension {dim!r} is empty",
            )
    return checks


def check_coordinates(
    obj: xr.Dataset | xr.DataArray,
    *,
    coordinates: Sequence[str] = ("time", "lat", "lon"),
) -> list[CheckResult]:
    """Check one-dimensional coordinates and increasing valid time labels."""
    checks: list[CheckResult] = []
    for name in coordinates:
        if not require_check(
            checks,
            f"coordinates.{name}.present",
            name in obj.coords,
            STANDARD,
            "required coordinate",
            list(obj.coords),
            None if name in obj.coords else f"Missing coordinate: {name!r}",
        ):
            continue
        coordinate = obj[name]
        if not require_check(
            checks,
            f"coordinates.{name}.dimensions",
            coordinate.ndim == 1,
            STANDARD,
            "one dimension",
            coordinate.dims,
            None
            if coordinate.ndim == 1
            else f"Coordinate {name!r} must be one-dimensional",
        ):
            continue
        values = np.asarray(coordinate.values)
        if not require_check(
            checks,
            f"coordinates.{name}.nonempty",
            values.size > 0,
            STANDARD,
            "positive length",
            values.size,
            None if values.size else f"Coordinate {name!r} is empty",
        ):
            continue
        observed = [str(value) for value in values] if name == "time" else values
        try:
            unique = bool(pd.Index(values).is_unique)
        except (TypeError, ValueError):
            unique = False
        require_check(
            checks,
            f"coordinates.{name}.unique",
            unique,
            STANDARD,
            "unique labels",
            observed,
            None if unique else f"Coordinate {name!r} contains duplicates",
        )
        if name == "time":
            try:
                times = pd.DatetimeIndex(pd.to_datetime(values))
            except (ValueError, TypeError, OverflowError) as exc:
                add_check(
                    checks,
                    "coordinates.time.valid",
                    "fail",
                    STANDARD,
                    "datetime labels",
                    observed,
                    str(exc),
                )
                continue
            valid = not times.hasnans
            require_check(
                checks,
                "coordinates.time.valid",
                valid,
                STANDARD,
                "no NaT",
                observed,
                None if valid else "Time coordinate contains NaT",
            )
            increasing = times.is_monotonic_increasing
            require_check(
                checks,
                "coordinates.time.increasing",
                increasing,
                STANDARD,
                "increasing time labels",
                observed,
                None
                if increasing
                else "Time coordinate is not monotonically increasing",
            )
    return checks


def check_channels(
    obj: xr.Dataset | xr.DataArray,
    *,
    coordinate: str = "channel",
) -> list[CheckResult]:
    """Check channel labels; level-packed arrays may explicitly select 'level'."""
    checks: list[CheckResult] = []
    if not require_check(
        checks,
        f"channels.{coordinate}.present",
        coordinate in obj.coords,
        STANDARD,
        "required coordinate",
        list(obj.coords),
        None if coordinate in obj.coords else f"Missing coordinate: {coordinate!r}",
    ):
        return checks
    labels = obj[coordinate]
    if not require_check(
        checks,
        f"channels.{coordinate}.dimensions",
        labels.ndim == 1,
        STANDARD,
        "one dimension",
        labels.dims,
        None if labels.ndim == 1 else "Channel coordinate must be one-dimensional",
    ):
        return checks
    values = [str(value) for value in labels.values.tolist()]
    require_check(
        checks,
        f"channels.{coordinate}.nonempty",
        bool(values),
        STANDARD,
        "nonempty labels",
        values,
        None if values else "Channel coordinate is empty",
    )
    unique = len(set(values)) == len(values)
    require_check(
        checks,
        f"channels.{coordinate}.unique",
        unique,
        STANDARD,
        "unique labels",
        values,
        None if unique else "Channel coordinate contains duplicates",
    )
    return checks


def check_same_layout(left: xr.DataArray, right: xr.DataArray) -> list[CheckResult]:
    """Report shape and dimension-order mismatches before elementwise arithmetic."""
    checks: list[CheckResult] = []
    require_check(
        checks,
        "layout.dimensions",
        left.dims == right.dims,
        STANDARD,
        left.dims,
        right.dims,
        None
        if left.dims == right.dims
        else f"Dimension order mismatch: left={left.dims}, right={right.dims}",
    )
    require_check(
        checks,
        "layout.shape",
        left.shape == right.shape,
        STANDARD,
        left.shape,
        right.shape,
        None
        if left.shape == right.shape
        else f"Shape mismatch: left={left.shape}, right={right.shape}",
    )
    return checks
