"""Logical layouts, coordinates and metadata of already opened dataset objects."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import xarray as xr

from datakit.schema.datasets.specifications import AuxiliaryDefinition, DatasetSchema

from ..support.reports import ValidationReport

STANDARD = "standard_conformance"


def check_dataset_identity(
    root: str, spec: DatasetSchema, report: ValidationReport,
) -> None:
    report.add(
        "publication.schema_approval",
        "pass" if spec.schema_status == "approved" else "needs-review",
        STANDARD, "approved", spec.schema_status,
        "Approval applies to the declared rules, not to the data being inspected",
    )
    if spec.directory_name:
        report.require(
            "publication.directory_name", root == spec.directory_name,
            STANDARD, spec.directory_name, root,
        )
        report.add(
            "publication.identity_mapping",
            "pass" if spec.directory_name == spec.dataset_id or spec.identity_mapping_evidence
            else "needs-review",
            STANDARD, spec.dataset_id, spec.directory_name, spec.identity_mapping_evidence,
        )


def check_declared_root(
    spec: DatasetSchema,
    report: ValidationReport,
    *,
    expected: set[str],
    undeclared: list[str],
    undeclared_auxiliary: list[str],
) -> None:
    report.add(
        "layout.declared_root_objects",
        "fail" if set(undeclared) - set(undeclared_auxiliary)
        else spec.unregistered_auxiliary_status if undeclared_auxiliary else "pass",
        STANDARD, sorted(expected), undeclared,
        "Unregistered NC/NPY auxiliaries are potential issues, not acceptance blockers; undeclared main arrays/other objects remain failures",
    )


def check_declared_auxiliaries(
    spec: DatasetSchema, report: ValidationReport, unexpected_auxiliary: list[str],
) -> None:
    report.add(
        "auxiliary.declared_files",
        spec.unregistered_auxiliary_status if unexpected_auxiliary else "pass",
        STANDARD, sorted(definition.filename for definition in spec.variant.auxiliaries),
        unexpected_auxiliary,
        "Unregistered auxiliaries were not used as approved roles and need separate review",
    )


def check_region_layout(
    region: np.ndarray, name: str, spec: DatasetSchema, report: ValidationReport,
) -> bool:
    valid = (
        isinstance(region, np.ndarray)
        and region.shape == spec.variant.grid.shape
        and region.dtype == np.dtype("bool")
    )
    return report.require(
        f"quality.region.{name}.layout", valid, "data_quality",
        (spec.variant.grid.shape, "bool"),
        (getattr(region, "shape", None), str(getattr(region, "dtype", None))),
        spec.valid_region_provenance[name],
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
    dataset: xr.Dataset,
    spec: DatasetSchema,
    report: ValidationReport,
) -> bool | None:
    """Return channel validity, or None when raw layout blocks further checks.

    Input must retain raw numeric dtypes (no CF fill masking or scale/offset).
    Only coordinates, mean/std and the two spatial masks are materialized.
    """
    layouts = spec.array_layouts()
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
        for key, value in spec.data_attributes.items():
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
        return None
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
    return channel_ok


def check_mask_labels(
    dataset: xr.Dataset, spec: DatasetSchema, report: ValidationReport,
) -> None:
    mask_schema = spec.product.mask
    mask_channels = dataset[mask_schema.channel_array_name].values
    report.require(
        "mask.labels",
        tuple(str(v) for v in mask_channels) == mask_schema.channels,
        STANDARD,
        mask_schema.channels,
        mask_channels,
    )


def check_mask_provenance(spec: DatasetSchema, report: ValidationReport) -> None:
    mask_schema = spec.product.mask
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


def check_channel_semantics(
    dataset: xr.Dataset,
    spec: DatasetSchema,
    report: ValidationReport,
    *,
    channel_ok: bool,
    stats_ok: bool,
) -> None:
    channel_metadata = spec.channel_metadata
    mean, std = dataset["mean"].values, dataset["std"].values
    info = dataset.channel.attrs.get(channel_metadata.attribute)
    info_ok = report.require(
        "semantics.channel_info",
        isinstance(info, dict)
        and (
            set(info) == set(spec.product.channel_names)
            if channel_metadata.require_exact_channels
            else set(spec.product.channel_names).issubset(info)
        )
        and all(isinstance(v, dict) for v in info.values()),
        STANDARD,
        spec.product.channel_names,
        sorted(info) if isinstance(info, dict) else info,
    )
    report.require(
        "semantics.info_version",
        dataset.channel.attrs.get(channel_metadata.version_attribute) == channel_metadata.version,
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
                    report.add(
                        f"semantics.{definition.name}.excluded_fields",
                        "not-applicable",
                        STANDARD,
                        evidence={"outside_validation_scope": definition.unconfirmed_fields},
                    )
            if channel_metadata.long_name_required:
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
                scale_offset_formula=channel_metadata.scale_offset_formula,
            )
            mismatches = {
                key: {"expected": value, "observed": observed.get(key)}
                for key, value in fields.items()
                if not _semantic_equal(observed.get(key), value)
            }
            report.require(
                f"semantics.{definition.name}", not mismatches, STANDARD, fields, mismatches
            )
            expected_statistics = spec.product.effective_statistics.get(definition.variable)
            if expected_statistics is not None:
                report.require(
                    f"statistics.{definition.variable}.identity",
                    (mean[index], std[index]) == expected_statistics,
                    STANDARD,
                    expected_statistics,
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


def check_auxiliary_format(
    obj: np.ndarray | xr.Dataset,
    definition: AuxiliaryDefinition,
    spec: DatasetSchema,
    report: ValidationReport,
) -> np.ndarray | None:
    """Validate the header before materializing an auxiliary's values."""
    prefix = f"auxiliary.{definition.filename}"
    lat, lon = spec.variant.grid.coordinates()
    sizes = {"lat": len(lat), "lon": len(lon), "singleton": 1, "channel": len(definition.channels)}
    expected_shape = tuple(sizes[dim] for dim in definition.dimensions)
    if isinstance(obj, xr.Dataset):
        if definition.check_variable_name:
            if not report.require(
                f"{prefix}.variable",
                definition.variable in obj.data_vars,
                STANDARD,
                definition.variable,
                sorted(obj.data_vars),
            ):
                return None
            variable = definition.variable
        else:
            report.add(
                f"{prefix}.variable_name", "not-applicable", STANDARD,
                observed=sorted(obj.data_vars),
                evidence="Auxiliary variable names are outside the current approved rule scope",
            )
            candidates = [
                name for name, value in obj.data_vars.items()
                if tuple(value.dims) == definition.dimensions
                and value.shape == expected_shape
                and value.dtype == np.dtype(definition.dtype)
            ]
            if len(obj.data_vars) == 1:
                variable = next(iter(obj.data_vars))
            elif len(candidates) == 1:
                variable = candidates[0]
            elif not obj.data_vars:
                report.add(f"{prefix}.array_present", "fail", STANDARD, observed=[])
                return None
            else:
                report.add(
                    f"{prefix}.variable_selection", "warning", STANDARD,
                    observed=sorted(obj.data_vars),
                    evidence="Multiple auxiliary variables have no unique declared layout; value checks deferred, no variable guessed",
                )
                return None
        array = obj[variable]
        report.add(
            f"{prefix}.additional_variables",
            "pass" if set(obj.data_vars) == {variable} else "warning",
            STANDARD, observed=sorted(obj.data_vars),
            evidence={"checked_variable": variable},
        )
        dims_ok = tuple(array.dims) == definition.dimensions
        header_ok = report.require(
            f"{prefix}.header_layout",
            dims_ok and array.shape == expected_shape and array.dtype == np.dtype(definition.dtype),
            STANDARD, (definition.dimensions, expected_shape, definition.dtype),
            (array.dims, array.shape, str(array.dtype)),
        )
        if not header_ok:
            report.add(
                f"{prefix}.values", "blocked", "data_quality",
                evidence="Invalid auxiliary header; no unbounded materialization",
            )
            return None
        for dim, expected in (("lat", lat), ("lon", lon)):
            if dim in array.dims:
                report.require(
                    f"{prefix}.{dim}",
                    dim in obj.coords and np.array_equal(obj[dim].values, expected),
                    STANDARD, expected,
                    obj[dim].values if dim in obj.coords else "missing",
                )
        if definition.channels:
            report.require(
                f"{prefix}.channels",
                "channel" in obj.coords
                and tuple(str(v) for v in obj.channel.values) == definition.channels,
                STANDARD, definition.channels,
                obj.channel.values if "channel" in obj else None,
            )
        values = array.values
    else:
        values, dims_ok = obj, True
        report.add(
            f"{prefix}.grid_association",
            "pass" if spec.auxiliary_provenance.get(definition.filename) else "needs-review",
            STANDARD, evidence=spec.auxiliary_provenance.get(definition.filename),
        )
    layout = report.require(
        f"{prefix}.layout",
        dims_ok and values.shape == expected_shape and values.dtype == np.dtype(definition.dtype),
        STANDARD, (definition.dimensions, expected_shape, definition.dtype),
        (values.shape, str(values.dtype)),
    )
    if not layout:
        report.add(
            f"{prefix}.values", "blocked", "data_quality",
            evidence="Invalid raw auxiliary shape/dtype",
        )
        return None
    return values


def check_auxiliary_role(
    definition: AuxiliaryDefinition, spec: DatasetSchema, report: ValidationReport,
) -> None:
    confirmed = definition.semantics_confirmed or bool(
        spec.confirmed_optional_roles.get(definition.filename)
    )
    report.add(
        f"auxiliary.{definition.filename}.role",
        "pass" if confirmed else "needs-review",
        STANDARD, definition.role, spec.confirmed_optional_roles.get(definition.filename),
    )
