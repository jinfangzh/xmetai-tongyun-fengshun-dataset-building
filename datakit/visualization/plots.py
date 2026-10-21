"""Lightweight Matplotlib plots for in-memory xarray data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr


XarrayObject = xr.Dataset | xr.DataArray


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib is required for plotting; install "
            "xmetai-datakit[visualization]"
        ) from exc
    return plt


def _select_array(
    obj: XarrayObject,
    *,
    variable: str | None,
    selection: Mapping[str, Any] | None,
    indexers: Mapping[str, int] | None,
) -> xr.DataArray:
    if isinstance(obj, xr.Dataset):
        if variable is None:
            if len(obj.data_vars) != 1:
                raise ValueError(
                    "variable is required when plotting a Dataset with multiple "
                    "data variables"
                )
            variable = next(iter(obj.data_vars))
        if variable not in obj.data_vars:
            raise KeyError(
                f"Variable {variable!r} not found; available variables: "
                f"{list(obj.data_vars)!r}"
            )
        array = obj[variable]
    elif isinstance(obj, xr.DataArray):
        if variable is not None and variable != obj.name:
            raise ValueError("variable cannot select another value from a DataArray")
        array = obj
    else:
        raise TypeError("obj must be an xarray Dataset or DataArray")

    if selection:
        array = array.sel(dict(selection))
    if indexers:
        array = array.isel(dict(indexers))
    return array.squeeze(drop=True)


def _require_dimensions(array: xr.DataArray, dimensions: Sequence[str]) -> None:
    expected = tuple(dimensions)
    if tuple(array.dims) != expected and set(array.dims) != set(expected):
        raise ValueError(
            f"Expected dimensions {expected!r} after selection, got {array.dims!r}; "
            "use selection or indexers to select remaining dimensions"
        )


def _coordinate_name(
    array: xr.DataArray,
    requested: str | None,
    candidates: Sequence[str],
    *,
    role: str,
) -> str:
    if requested is not None:
        if requested not in array.coords:
            raise ValueError(f"{role} coordinate {requested!r} not found")
        return requested
    for name in candidates:
        if name in array.coords:
            return name
    raise ValueError(
        f"Could not infer the {role} coordinate; pass {role}= explicitly"
    )


def _label(value: xr.DataArray) -> str:
    name = value.attrs.get("long_name") or value.attrs.get("standard_name") or value.name
    units = value.attrs.get("units")
    if name and units:
        return f"{name} [{units}]"
    return str(name or units or "value")


def _axes(ax: Any) -> tuple[Any, Any]:
    if ax is not None:
        return ax.figure, ax
    figure, axes = _pyplot().subplots()
    return figure, axes


def _field_array(
    obj: XarrayObject,
    *,
    variable: str | None,
    selection: Mapping[str, Any] | None,
    indexers: Mapping[str, int] | None,
    x: str | None,
    y: str | None,
) -> tuple[xr.DataArray, str, str]:
    array = _select_array(
        obj,
        variable=variable,
        selection=selection,
        indexers=indexers,
    )
    x_name = _coordinate_name(
        array,
        x,
        ("lon", "longitude", "x"),
        role="x",
    )
    y_name = _coordinate_name(
        array,
        y,
        ("lat", "latitude", "y"),
        role="y",
    )
    spatial_dims = tuple(dict.fromkeys((*array[y_name].dims, *array[x_name].dims)))
    _require_dimensions(array, spatial_dims)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional field after selection, got {array.dims!r}"
        )
    return array.transpose(*spatial_dims), x_name, y_name


def plot_field(
    obj: XarrayObject,
    *,
    variable: str | None = None,
    selection: Mapping[str, Any] | None = None,
    indexers: Mapping[str, int] | None = None,
    x: str | None = None,
    y: str | None = None,
    ax: Any = None,
    cmap: Any = None,
    norm: Any = None,
    colorbar: bool = True,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot one selected two-dimensional field with ``pcolormesh``.

    The function accepts both regular one-dimensional latitude/longitude
    coordinates and two-dimensional curvilinear coordinates. It returns the
    Matplotlib axes so callers can add coastlines or other custom elements.
    """

    array, x_name, y_name = _field_array(
        obj,
        variable=variable,
        selection=selection,
        indexers=indexers,
        x=x,
        y=y,
    )
    figure, axes = _axes(ax)
    options = {"shading": "auto", **kwargs}
    artist = axes.pcolormesh(
        array[x_name].values,
        array[y_name].values,
        np.asarray(array.values),
        cmap=cmap,
        norm=norm,
        **options,
    )
    axes.set_xlabel(_label(array[x_name]))
    axes.set_ylabel(_label(array[y_name]))
    axes.set_title(title if title is not None else str(array.name or ""))
    if colorbar:
        figure.colorbar(artist, ax=axes, label=_label(array))
    return axes


def plot_contour(
    obj: XarrayObject,
    *,
    variable: str | None = None,
    selection: Mapping[str, Any] | None = None,
    indexers: Mapping[str, int] | None = None,
    x: str | None = None,
    y: str | None = None,
    ax: Any = None,
    levels: int | Sequence[float] = 10,
    filled: bool = True,
    cmap: Any = None,
    norm: Any = None,
    colorbar: bool = True,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot filled or line contours for one selected two-dimensional field."""

    array, x_name, y_name = _field_array(
        obj,
        variable=variable,
        selection=selection,
        indexers=indexers,
        x=x,
        y=y,
    )
    figure, axes = _axes(ax)
    method = axes.contourf if filled else axes.contour
    artist = method(
        array[x_name].values,
        array[y_name].values,
        np.asarray(array.values),
        levels=levels,
        cmap=cmap,
        norm=norm,
        **kwargs,
    )
    axes.set_xlabel(_label(array[x_name]))
    axes.set_ylabel(_label(array[y_name]))
    axes.set_title(title if title is not None else str(array.name or ""))
    if colorbar:
        figure.colorbar(artist, ax=axes, label=_label(array))
    return axes


def plot_series(
    obj: XarrayObject,
    *,
    variable: str | None = None,
    dim: str = "time",
    selection: Mapping[str, Any] | None = None,
    indexers: Mapping[str, int] | None = None,
    ax: Any = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot a selected one-dimensional series, usually along time."""

    array = _select_array(
        obj,
        variable=variable,
        selection=selection,
        indexers=indexers,
    )
    if dim not in array.dims:
        raise ValueError(f"Dimension {dim!r} not found in {array.dims!r}")
    _require_dimensions(array, (dim,))
    _, axes = _axes(ax)
    axes.plot(array[dim].values, np.asarray(array.values), **kwargs)
    axes.set_xlabel(_label(array[dim]))
    axes.set_ylabel(_label(array))
    axes.set_title(title if title is not None else str(array.name or ""))
    return axes


def plot_profile(
    obj: XarrayObject,
    *,
    variable: str | None = None,
    vertical: str | None = None,
    selection: Mapping[str, Any] | None = None,
    indexers: Mapping[str, int] | None = None,
    ax: Any = None,
    invert_yaxis: bool | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot a selected vertical profile against pressure, level or height."""

    array = _select_array(
        obj,
        variable=variable,
        selection=selection,
        indexers=indexers,
    )
    vertical_name = _coordinate_name(
        array,
        vertical,
        ("level", "pressure", "isobaricInhPa", "height", "altitude", "z"),
        role="vertical",
    )
    if array[vertical_name].ndim != 1:
        raise ValueError("The vertical coordinate must be one-dimensional")
    vertical_dim = array[vertical_name].dims[0]
    _require_dimensions(array, (vertical_dim,))
    _, axes = _axes(ax)
    axes.plot(np.asarray(array.values), array[vertical_name].values, **kwargs)
    axes.set_xlabel(_label(array))
    axes.set_ylabel(_label(array[vertical_name]))
    axes.set_title(title if title is not None else str(array.name or ""))

    if invert_yaxis is None:
        units = str(array[vertical_name].attrs.get("units", "")).lower()
        positive = str(array[vertical_name].attrs.get("positive", "")).lower()
        invert_yaxis = units in {"pa", "hpa", "mbar", "millibar"} or positive == "down"
    if invert_yaxis:
        axes.invert_yaxis()
    return axes
