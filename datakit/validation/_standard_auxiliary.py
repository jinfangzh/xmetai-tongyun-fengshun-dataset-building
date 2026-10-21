"""Checks of already read auxiliary objects against explicit product roles."""

from __future__ import annotations

import numpy as np
import xarray as xr

from datakit.schema.standard_dataset import AuxiliaryDefinition, DatasetValidationSpec
from .reports import ValidationReport
from .standard_dataset import STANDARD


def check_standard_auxiliary(
    obj: np.ndarray | xr.Dataset,
    definition: AuxiliaryDefinition,
    spec: DatasetValidationSpec,
    mask: np.ndarray | None,
    report: ValidationReport,
) -> None:
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
                return
            variable = definition.variable
        else:
            report.add(
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
                and value.dtype == np.dtype(definition.dtype)
            ]
            if len(obj.data_vars) == 1:
                variable = next(iter(obj.data_vars))
            elif len(candidates) == 1:
                variable = candidates[0]
            elif not obj.data_vars:
                report.add(f"{prefix}.array_present", "fail", STANDARD, observed=[])
                return
            else:
                report.add(
                    f"{prefix}.variable_selection",
                    "warning",
                    STANDARD,
                    observed=sorted(obj.data_vars),
                    evidence="Multiple auxiliary variables have no unique declared layout; value checks deferred, no variable guessed",
                )
                return
        array = obj[variable]
        report.add(
            f"{prefix}.additional_variables",
            "pass" if set(obj.data_vars) == {variable} else "warning",
            STANDARD,
            observed=sorted(obj.data_vars),
            evidence={"checked_variable": variable},
        )
        dims_ok = tuple(array.dims) == definition.dimensions
        header_ok = report.require(
            f"{prefix}.header_layout",
            dims_ok and array.shape == expected_shape and array.dtype == np.dtype(definition.dtype),
            STANDARD,
            (definition.dimensions, expected_shape, definition.dtype),
            (array.dims, array.shape, str(array.dtype)),
        )
        if not header_ok:
            report.add(
                f"{prefix}.values",
                "blocked",
                "data_quality",
                evidence="Invalid auxiliary header; no unbounded materialization",
            )
            return
        for dim, expected in (("lat", lat), ("lon", lon)):
            if dim in array.dims:
                report.require(
                    f"{prefix}.{dim}",
                    dim in obj.coords and np.array_equal(obj[dim].values, expected),
                    STANDARD,
                    expected,
                    obj[dim].values if dim in obj.coords else "missing",
                )
        if definition.channels:
            report.require(
                f"{prefix}.channels",
                "channel" in obj.coords
                and tuple(str(v) for v in obj.channel.values) == definition.channels,
                STANDARD,
                definition.channels,
                obj.channel.values if "channel" in obj else None,
            )
        values = array.values
    else:
        values, dims_ok = obj, True
        report.add(
            f"{prefix}.grid_association",
            "pass" if spec.auxiliary_provenance.get(definition.filename) else "needs-review",
            STANDARD,
            evidence=spec.auxiliary_provenance.get(definition.filename),
        )
    layout = report.require(
        f"{prefix}.layout",
        dims_ok and values.shape == expected_shape and values.dtype == np.dtype(definition.dtype),
        STANDARD,
        (definition.dimensions, expected_shape, definition.dtype),
        (values.shape, str(values.dtype)),
    )
    if not layout:
        report.add(
            f"{prefix}.values",
            "blocked",
            "data_quality",
            evidence="Invalid raw auxiliary shape/dtype",
        )
        return
    report.require(f"{prefix}.finite", bool(np.isfinite(values).all()), "data_quality")
    confirmed = definition.semantics_confirmed or bool(
        spec.confirmed_optional_roles.get(definition.filename)
    )
    report.add(
        f"{prefix}.role",
        "pass" if confirmed else "needs-review",
        STANDARD,
        definition.role,
        spec.confirmed_optional_roles.get(definition.filename),
    )
    if definition.role == "weight" and layout:
        tolerance = spec.auxiliary_tolerances.get(
            definition.filename, definition.weight_nonnegative_atol
        )
        nonnegative = report.require(
            f"{prefix}.weight_nonnegative",
            bool((values >= -tolerance).all() and (values > 0).any()),
            "data_quality",
            {"minimum": -tolerance, "positive_pixels_required": True},
            {
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "negative_count": int((values < 0).sum()),
                "below_tolerance_count": int((values < -tolerance).sum()),
            },
            spec.auxiliary_provenance.get(definition.filename),
        )
        if nonnegative and bool((values < 0).any()):
            report.add(
                f"{prefix}.weight_roundoff",
                "warning",
                "data_quality",
                {"absolute_tolerance": tolerance},
                {"minimum": float(values.min()), "negative_count": int((values < 0).sum())},
                "Within the approved nonnegative tolerance; raw weights were not clipped or changed",
            )
    if definition.equals_mask and layout:
        if mask is None:
            report.add(
                f"{prefix}.mask_equal", "blocked", STANDARD, evidence="Main mask layout invalid"
            )
        else:
            report.require(
                f"{prefix}.mask_equal",
                np.array_equal(
                    values, mask[spec.product.mask.channels.index(definition.equals_mask)]
                ),
                STANDARD,
            )
    if (
        definition.role == "const"
        and layout
        and spec.product.mask.require_provenance
        and spec.mask_derivation == "lsm>0.5"
    ):
        if "lsm" not in definition.channels or mask is None:
            report.add(
                f"{prefix}.lsm_derivation",
                "blocked",
                STANDARD,
                evidence="Approved derivation requires declared lsm channel",
            )
        else:
            report.require(
                f"{prefix}.lsm_derivation",
                np.array_equal(
                    values[definition.channels.index("lsm")] > 0.5,
                    mask[spec.product.mask.region_index("land")] == spec.product.mask.valid_value,
                ),
                STANDARD,
                "Approved lsm>0.5 relation to unchanged binary mask",
            )
