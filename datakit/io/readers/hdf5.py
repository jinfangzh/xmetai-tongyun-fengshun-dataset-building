"""Serialize access to HDF5-backed readers within a process."""

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

_HDF5_ENGINES = frozenset({"netcdf", "h5netcdf"})
_HDF5_LOCK = threading.Lock()


def uses_hdf5(engine: Optional[str], engine_fallback: Optional[str] = None) -> bool:
    return (
        str(engine or "").strip() in _HDF5_ENGINES
        or str(engine_fallback or "").strip() in _HDF5_ENGINES
    )


@contextmanager
def hdf5_guard(
    engine: Optional[str], engine_fallback: Optional[str] = None
) -> Iterator[None]:
    """Lock the open-read-close window for HDF5-based backends."""
    if not uses_hdf5(engine, engine_fallback):
        yield
        return
    with _HDF5_LOCK:
        yield
