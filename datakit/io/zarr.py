from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

from ._common import PathLike, as_dataset, local_path, path_exists, select_variable


def _default_blosc_encoding() -> dict[str, Any] | None:
    """Build default Zarr compression encoding compatible with v3 and v2.

    Preference order:
    1) Zarr v3 codec: ``compressors=[BloscCodec(...)]``
    2) Zarr v2 codec: ``compressor=Blosc(...)``
    """

    try:
        from zarr.codecs import BloscCodec

        return {
            "compressors": [
                BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
            ]
        }
    except Exception:
        pass

    try:
        from numcodecs import Blosc

        return {
            "compressor": Blosc(
                cname="zstd",
                clevel=5,
                shuffle=Blosc.BITSHUFFLE,
            )
        }
    except Exception:  # pragma: no cover - optional dependency
        return None


def _with_default_zarr_compressor(
    data: xr.Dataset | xr.DataArray,
    *,
    encoding: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Inject default Blosc compressor for all variables unless explicitly set."""

    default_encoding = _default_blosc_encoding()
    if default_encoding is None:
        return dict(encoding) if encoding is not None else None

    merged: dict[str, Any] = dict(encoding or {})
    dataset = as_dataset(data)
    for variable in dataset.variables:
        current = merged.get(variable)
        if current is None:
            merged[variable] = dict(default_encoding)
            continue
        if isinstance(current, Mapping):
            current_options = dict(current)
            for key, value in default_encoding.items():
                current_options.setdefault(key, value)
            merged[variable] = current_options
    return merged if merged else None


def _coerce_zarr_sidecar_array(name: str, value: Any) -> np.ndarray:
    """Convert sidecar payloads to NumPy arrays before Zarr writing."""

    if isinstance(value, xr.Dataset):
        if len(value.data_vars) != 1:
            raise ValueError(f"Sidecar {name!r} dataset must contain exactly one variable")
        value = value[next(iter(value.data_vars))]
    if isinstance(value, xr.DataArray):
        array = np.asarray(value.values)
    else:
        array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"Sidecar {name!r} cannot be empty")
    return array


def write_zarr_sidecars(
    path: PathLike,
    sidecars: Mapping[str, Any],
) -> None:
    """Write sidecar arrays into top-level Zarr arrays ``<zarr>/<name>``."""

    if not sidecars:
        return

    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("zarr is required to write sidecar arrays") from exc

    root = zarr.open_group(str(local_path(path)), mode="r+")
    default_encoding = _default_blosc_encoding() or {}
    for name, raw in sidecars.items():
        values = _coerce_zarr_sidecar_array(name, raw)
        if name in root:
            del root[name]
        try:
            root.create_array(
                name=name,
                data=values,
                overwrite=True,
                **default_encoding,
            )
        except (TypeError, ValueError):
            root.create_array(name=name, data=values, overwrite=True)
        except Exception:
            try:
                root.create_dataset(
                    name=name,
                    data=values,
                    overwrite=True,
                    **default_encoding,
                )
            except TypeError:
                root.create_dataset(name=name, data=values, overwrite=True)


def read_zarr_array(
    path: PathLike,
    name: str,
    *,
    flatten: bool = False,
) -> np.ndarray | None:
    """Read one top-level Zarr array from ``<zarr>/<name>``."""

    store = Path(local_path(path))
    if not store.exists() or not store.is_dir():
        return None
    if not ((store / ".zgroup").exists() or (store / "zarr.json").exists()):
        return None

    try:
        import zarr
    except ImportError:
        return None

    candidate = store / name
    if not candidate.exists():
        return None

    try:
        node = zarr.open_array(str(candidate), mode="r")
    except Exception:
        try:
            group = zarr.open_group(str(candidate), mode="r")
            keys = list(group.array_keys())
            if not keys:
                return None
            node = group[keys[0]]
        except Exception:
            return None

    values = np.asarray(node[:])
    return values.reshape(-1) if flatten else values


def update_zarr_coordinate(
    path: PathLike,
    coordinate: str,
    values: np.ndarray,
) -> None:
    """Update one existing Zarr coordinate after validating its shape."""

    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("zarr is required to update Zarr coordinates") from exc

    group = zarr.open_group(str(local_path(path)), mode="r+")
    if coordinate not in group:
        raise KeyError(f"Coordinate '{coordinate}' not found in dataset")
    array = group[coordinate]
    values = np.asarray(values)
    if array.shape != values.shape:
        raise ValueError(
            f"Shape mismatch for '{coordinate}': "
            f"existing={array.shape}, incoming={values.shape}"
        )
    array[:] = values


def _validate_zarr_append_compat(
    current: xr.Dataset,
    incoming: xr.Dataset,
    *,
    append_dim: str,
) -> None:
    if append_dim not in current.dims:
        raise ValueError(
            f"append_dim '{append_dim}' is not in existing target dims: "
            f"{tuple(current.dims)}"
        )
    if append_dim not in incoming.dims:
        raise ValueError(
            f"append_dim '{append_dim}' is not in incoming data dims: "
            f"{tuple(incoming.dims)}"
        )

    current_vars = set(current.data_vars)
    incoming_vars = set(incoming.data_vars)
    if current_vars != incoming_vars:
        raise ValueError(
            "Variable mismatch for append. "
            f"target={sorted(current_vars)}, incoming={sorted(incoming_vars)}"
        )

    for variable in sorted(current_vars):
        current_dims = tuple(dim for dim in current[variable].dims if dim != append_dim)
        incoming_dims = tuple(dim for dim in incoming[variable].dims if dim != append_dim)
        if current_dims != incoming_dims:
            raise ValueError(
                f"Variable '{variable}' dimension mismatch for append: "
                f"target={current_dims}, incoming={incoming_dims}"
            )

    for dim, size in current.sizes.items():
        if dim == append_dim:
            continue
        if dim in incoming.sizes and incoming.sizes[dim] != size:
            raise ValueError(
                f"Dimension '{dim}' size mismatch for append: "
                f"target={size}, incoming={incoming.sizes[dim]}"
            )

    current_fixed_coords = {
        name for name, coord in current.coords.items() if append_dim not in coord.dims
    }
    incoming_fixed_coords = {
        name for name, coord in incoming.coords.items() if append_dim not in coord.dims
    }
    if current_fixed_coords != incoming_fixed_coords:
        raise ValueError(
            "Fixed-coordinate mismatch for append. "
            f"target={sorted(current_fixed_coords)}, "
            f"incoming={sorted(incoming_fixed_coords)}"
        )
    for coord_name in sorted(current_fixed_coords):
        if not current.coords[coord_name].equals(incoming.coords[coord_name]):
            raise ValueError(f"Coordinate '{coord_name}' mismatch for append")


def read_zarr(
    path: PathLike,
    *,
    variable: str | None = None,
    consolidated: bool | None = None,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Open a local Zarr dataset lazily."""
    if not path_exists(path):
        raise FileNotFoundError(f"Zarr path not found: {path}")

    options = dict(kwargs)
    options.setdefault("chunks", {})
    dataset = xr.open_zarr(str(local_path(path)), consolidated=consolidated, **options)
    return select_variable(dataset, variable, source=path)


def write_zarr(
    data: xr.Dataset | xr.DataArray,
    path: PathLike,
    *,
    mode: str = "w",
    append_dim: str | None = None,
    consolidated: bool | None = None,
    chunks: Mapping[str, int] | None = None,
    check_append_compat: bool = True,
    # zarr_format: int = 3,
    **kwargs: Any,
) -> str:
    """Write an xarray object to a local Zarr store."""
    options = dict(kwargs)
    options["encoding"] = _with_default_zarr_compressor(
        data,
        encoding=options.get("encoding"),
    )
    if options["encoding"] is None:
        del options["encoding"]

    if chunks is not None:
        data = data.chunk(chunks)

    if mode == "a" and append_dim and check_append_compat and path_exists(path):
        current = read_zarr(path)
        try:
            _validate_zarr_append_compat(current, as_dataset(data), append_dim=append_dim)
        finally:
            current.close()

    store = str(local_path(path, create_parent=True))
    data.to_zarr(
        store,
        mode=mode,
        append_dim=append_dim,
        consolidated=consolidated,
        # zarr_format=zarr_format,
        **options,
    )
    return str(path)


def append_zarr(
    data: xr.Dataset | xr.DataArray,
    path: PathLike,
    *,
    append_dim: str = "time",
    **kwargs: Any,
) -> str:
    """Append data after validating variables, dimensions and fixed coordinates."""

    return write_zarr(
        data,
        path,
        mode="a",
        append_dim=append_dim,
        **kwargs,
    )


def merge_zarr(
    sources: list[PathLike] | tuple[PathLike, ...],
    destination: PathLike,
    *,
    concat_dim: str = "time",
    **kwargs: Any,
) -> str:
    """Concatenate Zarr datasets exactly along one dimension and write the result."""

    if not sources:
        raise ValueError("sources cannot be empty")
    datasets = [read_zarr(path) for path in sources]
    try:
        merged = xr.concat(datasets, concat_dim, join="exact", combine_attrs="drop")
        if concat_dim in merged.coords:
            merged = merged.sortby(concat_dim)
        return write_zarr(merged, destination, **kwargs)
    finally:
        for dataset in datasets:
            dataset.close()
