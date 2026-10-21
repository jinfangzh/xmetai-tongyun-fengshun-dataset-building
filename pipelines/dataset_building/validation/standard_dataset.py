"""Strict checks of already opened standard datasets; no IO or coercion."""

from __future__ import annotations

from typing import Any
from numbers import Real

import numpy as np
import xarray as xr

from ..definitions.standard_dataset import DatasetValidationSpec
from .reports import ValidationReport

STANDARD = "standard_conformance"
DATA_ATTRIBUTES = {
    "data_representation": "normalized",
    "normalization_method": "channel_dependent",
    "mean_variable": "mean",
    "std_variable": "std",
    "scale_factor_location": "channel.channel_info[*].scale_factor",
    "add_offset_location": "channel.channel_info[*].add_offset",
    "inverse_transform": "preprocessed_value = data * scale_factor + add_offset",
}


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


def check_standard_dataset(
    dataset: xr.Dataset,
    spec: DatasetValidationSpec,
    report: ValidationReport,
) -> bool:
    """Check metadata, coordinates, statistics and masks without reading data.

    Input must retain raw numeric dtypes (no CF fill masking or scale/offset).
    Only coordinates, mean/std and the two spatial masks are materialized.
    """
    shape = spec.expected_shape
    mask_schema = spec.product.mask
    layouts = {
        "data": (("time", "channel", "lat", "lon"), shape, "float16"),
        "time": (("time",), (shape[0],), "datetime64[ns]"),
        "channel": (("channel",), (shape[1],), "string"),
        "lat": (("lat",), (shape[2],), "float32"),
        "lon": (("lon",), (shape[3],), "float32"),
        "mean": (("channel",), (shape[1],), "float32"),
        "std": (("channel",), (shape[1],), "float32"),
    }
    layouts.update(mask_schema.layouts(spec.variant.grid))
    usable = True
    for name, (dims, expected_shape, dtype) in layouts.items():
        if name not in dataset.variables:
            report.add(f"layout.{name}", "fail", STANDARD, (dims, expected_shape, dtype), "missing")
            usable = False
            continue
        array = dataset[name]
        dtype_ok = (
            array.dtype.kind in "UST" if dtype == "string" else array.dtype == np.dtype(dtype)
        )
        usable = (
            report.require(
                f"layout.{name}",
                tuple(array.dims) == dims and array.shape == expected_shape and dtype_ok,
                STANDARD,
                (dims, expected_shape, dtype),
                (array.dims, array.shape, str(array.dtype)),
            )
            and usable
        )
    report.require(
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
        report.require(
            f"attributes.root.{key}",
            _semantic_equal(dataset.attrs.get(key), value),
            STANDARD,
            value,
            dataset.attrs.get(key),
        )
    if "data" in dataset:
        for key, value in DATA_ATTRIBUTES.items():
            report.require(
                f"attributes.data.{key}",
                dataset.data.attrs.get(key) == value,
                STANDARD,
                value,
                dataset.data.attrs.get(key),
            )
    if not usable:
        report.add(
            "semantics.prerequisites",
            "blocked",
            STANDARD,
            evidence="Invalid raw layout; semantic checks were not coerced",
        )
        return False
    for name, expected in (("lat", lat), ("lon", lon)):
        values = dataset[name].values
        report.require(
            f"coordinates.{name}.complete",
            np.array_equal(values, expected),
            STANDARD,
            {"count": len(expected), "first": float(expected[0]), "last": float(expected[-1])},
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
    report.require(
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
    report.require(
        "time.utc",
        dataset.time.attrs.get("timezone") == "UTC",
        STANDARD,
        "UTC",
        dataset.time.attrs.get("timezone"),
    )
    channels = tuple(str(value) for value in dataset.channel.values)
    channel_ok = report.require(
        "channels.exact_order",
        channels == spec.product.channel_names,
        STANDARD,
        spec.product.channel_names,
        channels,
    )
    mean, std = dataset["mean"].values, dataset["std"].values
    stats_ok = report.require(
        "statistics.finite_positive",
        bool(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()),
        STANDARD,
        "finite mean/std and std > 0",
        {"mean": mean, "std": std},
    )
    mask = dataset[mask_schema.array_name].values
    mask_channels = dataset[mask_schema.channel_array_name].values
    report.require(
        "mask.labels",
        tuple(str(v) for v in mask_channels) == mask_schema.channels,
        STANDARD,
        mask_schema.channels,
        mask_channels,
    )
    mask_ok = report.require(
        "mask.binary_complement",
        bool(
            np.isin(mask, mask_schema.allowed_values).all()
            and np.all(
                np.add(
                    mask[mask_schema.region_index("land")],
                    mask[mask_schema.region_index("sea")],
                    dtype=np.int64,
                )
                == mask_schema.complement_sum
            )
        ),
        STANDARD,
        {
            "allowed_values": mask_schema.allowed_values,
            "complement_sum": mask_schema.complement_sum,
        },
    )
    report.add(
        "mask.provenance",
        (
            ("pass" if spec.mask_provenance else "needs-review")
            if mask_schema.require_provenance
            else "not-applicable"
        ),
        STANDARD,
        evidence=(
            spec.mask_provenance
            if mask_schema.require_provenance
            else "Acceptance uses the embedded Zarr mask layout, labels, binary values and complement; source derivation is outside this scope"
        ),
    )
    info = dataset.channel.attrs.get("channel_info")
    info_ok = report.require(
        "semantics.channel_info",
        isinstance(info, dict)
        and set(info) == set(spec.product.channel_names)
        and all(isinstance(v, dict) for v in info.values()),
        STANDARD,
        spec.product.channel_names,
        sorted(info) if isinstance(info, dict) else info,
    )
    report.require(
        "semantics.info_version",
        dataset.channel.attrs.get("channel_info_schema_version") == "1.0",
        STANDARD,
        "1.0",
        dataset.channel.attrs.get("channel_info_schema_version"),
    )
    if channel_ok and info_ok and stats_ok:
        for index, definition in enumerate(spec.product.channels):
            observed = info[definition.name]
            fields = definition.semantic_fields()
            if spec.validation_scope == "confirmed-format":
                for key in definition.unconfirmed_fields:
                    fields.pop(key, None)
                if definition.unconfirmed_fields:
                    report.add(
                        f"semantics.{definition.name}.excluded_fields",
                        "not-applicable",
                        STANDARD,
                        evidence={"outside_validation_scope": definition.unconfirmed_fields},
                    )
            report.require(
                f"semantics.{definition.name}.description",
                isinstance(observed.get("long_name"), str) and bool(observed["long_name"].strip()),
                STANDARD,
                "nonempty long_name",
                observed.get("long_name"),
            )
            fields.update(
                scale_factor=float(std[index]),
                add_offset=float(mean[index]),
                scale_offset_formula="preprocessed_value = normalized_value * scale_factor + add_offset",
            )
            mismatches = {
                key: {"expected": value, "observed": observed.get(key)}
                for key, value in fields.items()
                if not _semantic_equal(observed.get(key), value)
            }
            report.require(
                f"semantics.{definition.name}", not mismatches, STANDARD, fields, mismatches
            )
            if definition.variable == "tp":
                report.require(
                    "statistics.tp.identity",
                    mean[index] == 0 and std[index] == 1,
                    STANDARD,
                    [0, 1],
                    [mean[index], std[index]],
                )
    semantic_status = "pass"
    if spec.product.semantic_review:
        if spec.validation_scope == "confirmed-format":
            semantic_status = "not-applicable"
        elif not spec.semantic_confirmation:
            semantic_status = "needs-review"
    report.add(
        "semantics.product_confirmation",
        semantic_status,
        STANDARD,
        spec.product.semantic_review,
        evidence=spec.semantic_confirmation,
    )
    return channel_ok and stats_ok and mask_ok


def check_standard_time_alignment(
    times: list[np.ndarray], *, policy: str = "exact"
) -> dict[str, Any]:
    """Validate original time axes before calculating an explicit join policy."""
    if policy not in ("exact", "intersection") or len(times) < 2:
        raise ValueError("Expected at least two axes and exact/intersection policy")
    for axis in times:
        if (
            axis.dtype != np.dtype("datetime64[ns]")
            or not len(axis)
            or np.isnat(axis).any()
            or not np.all(np.diff(axis) > np.timedelta64(0, "ns"))
        ):
            raise ValueError(
                "Original time axes must be nanosecond, nonempty, unique and increasing"
            )
    common = times[0]
    for axis in times[1:]:
        common = np.intersect1d(common, axis)
    passed = bool(len(common)) and (
        policy == "intersection" or all(np.array_equal(times[0], axis) for axis in times[1:])
    )
    return {
        "passed": passed,
        "policy": policy,
        "common_count": len(common),
        "lost_counts": [len(axis) - len(common) for axis in times],
        "common_times": common,
    }
