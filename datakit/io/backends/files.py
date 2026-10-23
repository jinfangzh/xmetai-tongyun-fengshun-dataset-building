"""路径与格式识别、单文件或 Store 的打开、写出和句柄管理。"""

from __future__ import annotations

import os
import tempfile
import threading
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import xarray as xr

PathLike = str | Path

_FORMAT_BY_SUFFIX = {
    ".cdf": "netcdf",
    ".grb": "grib",
    ".grb2": "grib",
    ".grib": "grib",
    ".grib2": "grib",
    ".nc": "netcdf",
    ".netcdf": "netcdf",
    ".netcdf4": "netcdf",
    ".npy": "numpy",
    ".npz": "numpy",
    ".zarr": "zarr",
    ".zarr-v2": "zarr",
    ".zarr-v3": "zarr",
}

_FORMAT_ALIASES = {
    "cdf": "netcdf",
    "grb": "grib",
    "grb2": "grib",
    "grib2": "grib",
    "nc": "netcdf",
    "netcdf4": "netcdf",
    "npy": "numpy",
    "npz": "numpy",
    "zarr-v2": "zarr",
    "zarr-v3": "zarr",
}

SUPPORTED_FORMATS = frozenset({"grib", "netcdf", "numpy", "zarr"})


def normalize_format(format_name: str) -> str:
    normalized = _FORMAT_ALIASES.get(
        format_name.strip().lower(), format_name.strip().lower()
    )
    if normalized not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported data format: {format_name!r}; supported: {sorted(SUPPORTED_FORMATS)}"
        )
    return normalized


def infer_format(path: PathLike) -> str:
    """Infer a supported data format from a path suffix."""
    suffix = local_path(path).suffix.lower()
    try:
        return normalize_format(_FORMAT_BY_SUFFIX[suffix])
    except KeyError as exc:
        raise ValueError(f"Unsupported data suffix: {suffix or '<none>'}") from exc


def local_path(path: PathLike, *, create_parent: bool = False) -> Path:
    """Normalize a local path and reject URL/object-storage paths."""

    raw_path = os.fspath(path)
    if "://" in raw_path:
        raise ValueError(f"Only local filesystem paths are supported: {raw_path}")
    result = Path(raw_path).expanduser()
    if create_parent:
        result.parent.mkdir(parents=True, exist_ok=True)
    return result


def as_dataset(data: xr.Dataset | xr.DataArray, *, name: str = "data") -> xr.Dataset:
    return (
        data
        if isinstance(data, xr.Dataset)
        else data.to_dataset(name=data.name or name)
    )


# Shared by eager NetCDF/HDF5 consumers to serialize open/read/close.
NETCDF_READ_LOCK = threading.Lock()


def write_file(dataset, path, options):
    options = dict(options)
    if options.get("compute") is False:
        raise ValueError("write() executes the write; compute=False is not supported")
    mode = options.pop("mode", None)
    if path.exists() and mode not in {"w", "a", "r+"}:
        raise FileExistsError(f"Refusing to overwrite an existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix.startswith(".zarr"):
        chunks = options.pop("chunks", None)
        if chunks is not None:
            dataset = dataset.chunk(chunks)
        dataset.to_zarr(str(path), mode=mode or "w-", **options)
    elif suffix in {".nc", ".cdf", ".netcdf", ".netcdf4"}:
        dataset.to_netcdf(path, mode=mode or "w", **options)
    elif suffix in {".grib", ".grib2", ".grb", ".grb2"}:
        from cfgrib.xarray_to_grib import to_grib

        to_grib(dataset, path, **options)
    elif suffix == ".npz":
        with path.open("wb") as stream:
            np.savez(
                stream,
                **{name: value.values for name, value in dataset.data_vars.items()},
            )
    elif suffix == ".npy":
        if len(dataset.data_vars) != 1:
            raise ValueError("NPY output requires exactly one variable")
        with path.open("wb") as stream:
            np.save(stream, dataset[next(iter(dataset.data_vars))].values)
    else:
        raise ValueError(f"No lossless Xarray writer configured for {path.suffix!r}")
    return str(path)


def _detect_format(path) -> str:
    """Detect local file signatures or store metadata, then fall back to suffix."""
    local = local_path(path)
    suffix = local.suffix.lower()
    if local.is_dir() and any(
        (local / marker).is_file() for marker in (".zgroup", "zarr.json")
    ):
        return "zarr"
    if local.is_file():
        with local.open("rb") as stream:
            signature = stream.read(8)
        if signature.startswith(b"GRIB"):
            return "grib"
        if signature.startswith(
            (b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n")
        ):
            return "netcdf"
        if signature.startswith(b"\x93NUMPY"):
            return "numpy"
        if signature.startswith(b"PK\x03\x04"):
            return "numpy" if suffix == ".npz" else "netcdf"
    return infer_format(path)


@contextmanager
def open_xarray(path, **options):
    """Resource owner for public Xarray/NumPy backends; no product rules."""
    format_name = options.pop("format", None) or _detect_format(path)
    fallback = options.pop("engine_fallback", None)
    variable = options.pop("variable", None)
    variable_filters = options.pop("variable_filters", None)
    prepare = options.pop("prepare", None)
    if variable_filters:
        from contextlib import ExitStack

        with ExitStack() as stack:
            frames = []
            for name, filters in variable_filters.items():
                field_options = dict(options)
                field_options["backend_kwargs"] = {
                    **field_options.get("backend_kwargs", {}),
                    "filter_by_keys": filters,
                }
                dataset = stack.enter_context(open_xarray(path, **field_options))
                if prepare is not None:
                    dataset = prepare(dataset, [name])
                if dataset is not None:
                    frames.append(dataset)
            if not frames:
                raise ValueError("No requested GRIB variables could be decoded")
            yield frames[0] if len(frames) == 1 else xr.merge(frames)
        return
    if format_name == "numpy":
        key = options.pop("key", variable)
        options.setdefault("name", key or "data")
        yield read_numpy(path, key=key, **options).to_dataset()
        return
    if format_name == "zarr":
        options.setdefault("chunks", None)
        context = xr.open_zarr(str(path), **options)
    elif format_name == "netcdf":
        context = open_netcdf_dataset(path, engine_fallback=fallback, **options)
    elif format_name == "grib":
        options.setdefault("engine", "cfgrib")
        context = xr.open_dataset(path, **options)
    else:
        raise ValueError(f"Unsupported format: {format_name}")
    with context as dataset:
        if prepare is not None:
            dataset = prepare(dataset, None)
            if dataset is None:
                raise ValueError("No requested variables found in file")
        if variable is not None:
            dataset = dataset[[variable]]
        yield dataset


@contextmanager
def open_netcdf_dataset(
    path,
    **kwargs,
):
    """Open one local NetCDF dataset, including zip-wrapped `.nc` payloads."""

    fallback = kwargs.pop("engine_fallback", None)
    local = local_path(path)
    if not local.exists():
        raise FileNotFoundError(f"NetCDF path not found: {path}")

    with local.open("rb") as stream:
        zipped = stream.read(4) == b"PK\x03\x04"
    if not zipped:
        try:
            dataset = xr.open_dataset(local, **kwargs)
        except OSError:
            if not fallback:
                raise
            dataset = xr.open_dataset(local, **{**kwargs, "engine": fallback})
        with dataset:
            yield dataset
        return

    with tempfile.TemporaryDirectory(prefix="datakit_netcdf_") as temporary:
        with zipfile.ZipFile(local) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ValueError(
                    f"Zip-wrapped NetCDF must contain exactly one file: {local}"
                )
            archive.extract(members[0], temporary)
            extracted = Path(temporary) / members[0]
        with xr.open_dataset(extracted, **kwargs) as dataset:
            yield dataset


def read_numpy(
    path,
    *,
    key: str | None = None,
    dims: tuple[str, ...] | None = None,
    coords: dict | None = None,
    name: str = "data",
    allow_pickle: bool = False,
) -> xr.DataArray:
    """Read a .npy file or one array from a .npz archive."""
    if not local_path(path).exists():
        raise FileNotFoundError(f"NumPy path not found: {path}")

    with local_path(path).open("rb") as stream:
        loaded = np.load(stream, allow_pickle=allow_pickle)
        try:
            if isinstance(loaded, np.lib.npyio.NpzFile):
                if key is None:
                    if len(loaded.files) != 1:
                        raise ValueError(
                            "NPZ archive contains multiple arrays; "
                            f"specify key from: {loaded.files}"
                        )
                    key = loaded.files[0]
                if key not in loaded.files:
                    raise KeyError(
                        f"Array {key!r} not found; available: {sorted(loaded.files)}"
                    )
                values = np.asarray(loaded[key])
            else:
                values = np.asarray(loaded)
        finally:
            if hasattr(loaded, "close"):
                loaded.close()

    resolved_dims = dims or tuple(f"dim_{index}" for index in range(values.ndim))
    if len(resolved_dims) != values.ndim:
        raise ValueError(
            f"dims length ({len(resolved_dims)}) must match array ndim ({values.ndim})"
        )
    return xr.DataArray(values, dims=resolved_dims, coords=coords, name=name)


def read_records(records, description, **options):
    """读取已发现的文件记录，供统一入口和逐文件处理复用。"""
    if len(records) != 1:
        raise ValueError(
            "A single-dataset description must locate exactly one file or store"
        )
    stack = ExitStack()
    try:
        dataset = stack.enter_context(open_xarray(records[0].path, **options))
        dataset = dataset.copy(deep=False)
        dataset.set_close(stack.close)
        # open_xarray owns the original; closing this view releases its context.
        return dataset
    except BaseException:
        stack.close()
        raise


def write(dataset, root, description, **options):
    """按当前布局写出 Dataset，返回输出文件路径列表。"""
    from .._description import templates

    patterns = templates(description)
    if patterns:
        if len(patterns) != 1 or patterns[0][1].fields:
            raise ValueError("Single-dataset output requires a fixed filename")
        root = root / patterns[0][1].render({})
    return [write_file(dataset, root, options)]
