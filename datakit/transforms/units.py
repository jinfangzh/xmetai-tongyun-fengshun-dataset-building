from __future__ import annotations

from collections.abc import Collection, Mapping
from numbers import Real
from typing import TypeAlias

import numpy as np
import xarray as xr

from datakit.schema import UNIT_SCALE, WATER_VARIABLES


XarrayObject: TypeAlias = xr.Dataset | xr.DataArray

ERA5_RADIATION_VARS = frozenset({"ssr", "ssrd", "fdir", "ttr"})


def _require_data_array(value: xr.DataArray, *, name: str) -> xr.DataArray:
    if not isinstance(value, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    return value


def _finite_scalar(
    value: float, *, name: str, positive: bool = False, nonzero: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonzero and result == 0:
        raise ValueError(f"{name} must be non-zero")
    return result


def _affine_parameters(scale: float, offset: float) -> tuple[float, float]:
    return (
        _finite_scalar(scale, name="scale", nonzero=True),
        _finite_scalar(offset, name="offset"),
    )


def apply_affine(
    obj: xr.DataArray,
    *,
    scale: float,
    offset: float = 0.0,
    units: str | None = None,
    name: str | None = None,
) -> xr.DataArray:
    """Apply ``output = input * scale + offset`` to one physical variable."""

    source = _require_data_array(obj, name="obj")
    factor, translation = _affine_parameters(scale, offset)
    result = source * factor + translation
    result.name = name if name is not None else source.name
    result.attrs = dict(source.attrs)
    if units is not None:
        result.attrs["units"] = units
    return result


def invert_affine(
    obj: xr.DataArray,
    *,
    scale: float,
    offset: float = 0.0,
    units: str | None = None,
    name: str | None = None,
) -> xr.DataArray:
    """Undo ``output = input * scale + offset`` using the same parameters."""

    factor, translation = _affine_parameters(scale, offset)
    return apply_affine(
        obj,
        scale=1.0 / factor,
        offset=-translation / factor,
        units=units,
        name=name,
    )


def apply_scale(obj: XarrayObject, *, factors: Mapping[str, float]) -> XarrayObject:
    """Multiply explicitly named variables by finite real factors, without clipping.

    Unselected variables and attributes are retained; units are not inferred or
    updated. Unknown variable names are errors. Main-data computation stays lazy.
    """
    if not isinstance(obj, (xr.Dataset, xr.DataArray)):
        raise TypeError("obj must be an xarray Dataset or DataArray")
    coefficients = {name: _finite_scalar(value, name=f"factors[{name!r}]")
                    for name, value in factors.items()}
    available = set(obj.data_vars) if isinstance(obj, xr.Dataset) else {obj.name}
    missing = set(coefficients) - available
    if missing:
        raise ValueError(f"Unknown scale variables: {sorted(missing)}")

    def scale(array):
        if array.name not in coefficients:
            return array.copy(deep=False)
        _require_data_array(array, name=str(array.name))
        if array.dtype.kind == "c":
            raise TypeError("apply_scale requires real-valued data")
        result = array * coefficients[array.name]
        result.attrs = dict(array.attrs)
        return result

    return obj.map(scale, keep_attrs=True) if isinstance(obj, xr.Dataset) else scale(obj)


def apply_log1p(obj: XarrayObject, *, variables: Collection[str]) -> XarrayObject:
    """Apply log1p to explicit variables, with no implicit selection or clipping.

    NaNs propagate. Values <= -1 raise; this domain check scans selected data
    (including lazy arrays). Coordinates and attributes are retained unchanged.
    """
    if not isinstance(obj, (xr.Dataset, xr.DataArray)):
        raise TypeError("obj must be an xarray Dataset or DataArray")
    if isinstance(variables, str):
        raise TypeError("variables must be a collection of variable names")
    selected = set(variables)
    available = set(obj.data_vars) if isinstance(obj, xr.Dataset) else {obj.name}
    missing = selected - available
    if missing:
        raise ValueError(f"Unknown log1p variables: {sorted(missing)}")

    def transform(array):
        if array.name not in selected:
            return array.copy(deep=False)
        _require_data_array(array, name=str(array.name))
        if array.dtype.kind == "c":
            raise TypeError("apply_log1p requires real-valued data")
        if bool((array <= -1).any().compute().item()):
            raise ValueError(f"log1p requires values > -1: {array.name}")
        result = np.log1p(array)
        result.attrs = dict(array.attrs)
        return result

    return obj.map(transform, keep_attrs=True) if isinstance(obj, xr.Dataset) else transform(obj)


def rescale_units(
    obj: XarrayObject,
    *,
    scales: Mapping[str, float] = UNIT_SCALE,
) -> XarrayObject:
    """Apply per-variable scale factors without nonlinear transformations."""

    if isinstance(obj, xr.Dataset):
        return obj.map(lambda array: rescale_units(array, scales=scales))
    if obj.name not in scales:
        return obj.copy()
    result = obj * scales[obj.name]
    result.attrs = dict(obj.attrs)
    return result


def log1p_variables(
    obj: XarrayObject,
    *,
    variables: Collection[str] = WATER_VARIABLES,
    minimum: float = 0.0,
) -> XarrayObject:
    """Apply ``log1p`` only to selected named variables."""

    if isinstance(obj, xr.Dataset):
        return obj.map(
            lambda array: log1p_variables(
                array,
                variables=variables,
                minimum=minimum,
            )
        )
    if obj.name not in variables:
        return obj.copy()
    result = np.log1p(obj.clip(min=minimum))
    result.attrs = dict(obj.attrs)
    return result


def preprocess_channel_units(
    obj: xr.DataArray,
    *,
    channel_dim: str = "channel",
    humidity_prefix: str = "q",
    humidity_scale: float = UNIT_SCALE["q"],
    radiation_vars: Collection[str] = ERA5_RADIATION_VARS,
    accumulation_hours: float = 1.0,
    log1p_vars: Collection[str] = ("tp",),
    log1p_minimum: float = 0.0,
) -> xr.DataArray:
    """Preprocess channel-labeled meteorological data using datakit transforms.

    Steps:
    1. Convert ``DataArray(channel, ...)`` to temporary ``Dataset`` variables.
    2. Apply unit scaling via ``rescale_units``.
    3. Apply ``log1p`` to selected variables via ``log1p_variables``.
    4. Restore original channel order and dimensions.
    """

    source = _require_data_array(obj, name="obj")
    if channel_dim not in source.dims:
        raise ValueError(f"Expected '{channel_dim}' dimension for preprocessing")
    if accumulation_hours <= 0:
        raise ValueError("accumulation_hours must be positive")

    channel_names = [str(item) for item in source[channel_dim].values.tolist()]
    dataset = source.to_dataset(dim=channel_dim)

    scales: dict[str, float] = {}
    q_scale = float(humidity_scale)
    radiation_set = {str(item) for item in radiation_vars}

    for name in channel_names:
        if name.startswith(humidity_prefix):
            scales[name] = q_scale
        elif name in radiation_set and name in UNIT_SCALE:
            scales[name] = float(UNIT_SCALE[name]) / float(accumulation_hours)
        elif name in UNIT_SCALE:
            scales[name] = float(UNIT_SCALE[name])

    if scales:
        dataset = rescale_units(dataset, scales=scales)

    log_targets = tuple(str(item) for item in log1p_vars)
    if log_targets:
        dataset = log1p_variables(
            dataset,
            variables=log_targets,
            minimum=log1p_minimum,
        )

    dataset = dataset[channel_names]
    result = dataset.to_array(dim=channel_dim)
    result = result.transpose(*source.dims)
    result = result.assign_coords({channel_dim: source[channel_dim]})
    result.attrs = dict(source.attrs)
    result.name = source.name
    return result
