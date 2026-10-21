from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr

from ._common import PathLike, local_path, path_exists


def read_numpy(
    path: PathLike,
    *,
    key: str | None = None,
    dims: tuple[str, ...] | None = None,
    coords: Mapping[str, Any] | None = None,
    name: str = "data",
    allow_pickle: bool = False,
) -> xr.DataArray:
    """Read a .npy file or one array from a .npz archive."""
    if not path_exists(path):
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
                    raise KeyError(f"Array {key!r} not found; available: {sorted(loaded.files)}")
                values = np.asarray(loaded[key])
            else:
                values = np.asarray(loaded)
        finally:
            if hasattr(loaded, "close"):
                loaded.close()

    resolved_dims = dims or tuple(f"dim_{index}" for index in range(values.ndim))
    if len(resolved_dims) != values.ndim:
        raise ValueError(
            f"dims length ({len(resolved_dims)}) must match "
            f"array ndim ({values.ndim})"
        )
    return xr.DataArray(values, dims=resolved_dims, coords=coords, name=name)


def write_numpy(
    data: xr.Dataset | xr.DataArray | np.ndarray,
    path: PathLike,
) -> str:
    """Write an array to .npy, or dataset variables to .npz."""
    suffix = Path(str(path)).suffix.lower()
    if isinstance(data, xr.Dataset) and suffix != ".npz":
        raise ValueError("Writing an xarray Dataset requires a .npz path")
    if not isinstance(data, xr.Dataset) and suffix != ".npy":
        raise ValueError("Writing an array requires a .npy path")

    destination = local_path(path, create_parent=True)
    with destination.open("wb") as stream:
        if isinstance(data, xr.Dataset):
            np.savez(stream, **{name: variable.values for name, variable in data.data_vars.items()})
        else:
            values = data.values if isinstance(data, xr.DataArray) else data
            np.save(stream, np.asarray(values))
    return str(path)
