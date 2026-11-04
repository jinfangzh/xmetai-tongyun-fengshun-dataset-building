"""Local output staging and protection of sources owned by IO contexts."""

from collections import Counter
from contextlib import contextmanager
import os
import shutil
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


_READERS = Counter()
_LOCK = threading.Lock()


def check_source_coordinates(dataset, sources, *, scalar_coordinates=()):
    """Reject new labels rather than silently dropping them during projection."""
    indexes = {}
    for source in sources:
        for name, labels in source.items():
            index = pd.Index(np.atleast_1d(labels))
            indexes[name] = indexes[name].union(index, sort=False) if name in indexes else index
    for name, index in indexes.items():
        if name not in dataset.coords:
            continue
        coordinate = dataset[name]
        regular = coordinate.ndim == 0 or coordinate.dims == (name,)
        if not regular and name not in scalar_coordinates:
            raise ValueError(f"Cannot assign transformed coordinate {name!r} to source records")
        current = pd.Index(np.asarray(coordinate.values).reshape(-1))
        if coordinate.dims == (name,) and not current.is_unique:
            raise ValueError(f"Duplicate output coordinate labels: {name!r}")
        missing = current[~current.isin(index)]
        if len(missing):
            raise ValueError(f"New coordinate labels require an explicit output layout: {name!r}: {list(missing)!r}")


def project_coordinates(dataset, selectors, *, scalar_coordinates=()):
    """Intersect source membership with current coordinates, preserving current order."""
    part = dataset
    for name, labels in selectors.items():
        if name not in part.coords:
            raise ValueError(f"Source coordinate {name!r} was removed; cannot determine output membership")
        coordinate = part[name]
        index = pd.Index(np.asarray(coordinate.values).reshape(-1))
        positions = np.flatnonzero(index.isin(np.atleast_1d(labels)))
        if not len(positions):
            if name in scalar_coordinates:
                raise ValueError(f"Selected source coordinate {name!r} conflicts with the file record")
            return None
        if coordinate.ndim == 1 and coordinate.dims == (name,):
            part = part.isel({name: positions})
        elif coordinate.ndim != 0 and not (name in scalar_coordinates and coordinate.size == 1):
            raise ValueError(f"Cannot project coordinate {name!r}")
    if not part.data_vars or any(size == 0 for size in part.sizes.values()):
        return None
    return part


def check_source_variables(dataset, variables):
    unknown = set(dataset.data_vars) - set(variables)
    if unknown:
        raise ValueError(f"Variables have no source-file assignment: {sorted(unknown)}; use an explicit output layout")


@contextmanager
def reading(paths):
    paths = [Path(path).resolve() for path in paths]
    with _LOCK:
        _READERS.update(paths)
    try:
        yield
    finally:
        with _LOCK:
            _READERS.subtract(paths)
            for path in paths:
                if _READERS[path] == 0:
                    del _READERS[path]


def write_mode(options):
    mode = options.pop("mode", None)
    if mode not in (None, "w", "w-", "x"):
        raise ValueError("IO write supports creation or explicit replacement with mode='w'")
    compute = options.pop("compute", True)
    if not isinstance(compute, bool):
        raise TypeError("compute must be a boolean")
    if not compute:
        raise ValueError("write() executes the write; compute=False is not supported")
    return mode


def check_output(path, mode=None):
    path = Path(path)
    resolved = path.resolve()
    with _LOCK:
        sources = list(_READERS)
    for source in sources:
        overlaps = (
            resolved == source
            or resolved in source.parents
            or source in resolved.parents
        )
        if not overlaps and path.exists() and source.exists():
            overlaps = os.path.samefile(path, source)
        if overlaps:
            raise ValueError(f"Cannot write over an active IO source: {path}")
    if (path.exists() or path.is_symlink()) and mode != "w":
        raise FileExistsError(f"Refusing to overwrite an existing output: {path}")


def check_outputs(paths, mode=None):
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Output paths collide")
    for path in paths:
        check_output(path, mode)


@contextmanager
def staged_output(path, mode=None):
    """Publish a completed local output; replacement is per file or store.

    Directory replacement uses a recoverable backup, not an atomic directory
    exchange. Concurrent access and multi-file transactions are not supported.
    """
    path = Path(path)
    check_output(path, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as directory:
        staged = Path(directory) / path.name
        yield staged
        check_output(path, mode)
        if not staged.exists():
            raise FileNotFoundError(f"Writer did not produce output: {path}")
        if path.is_dir() and not path.is_symlink():
            backup = path.with_name(f".{path.name}.{uuid4().hex}.backup")
            path.replace(backup)
            try:
                staged.replace(path)
            except BaseException:
                # The backup stays outside the temporary directory if restore fails.
                backup.replace(path)
                raise
            shutil.rmtree(backup)
        elif mode == "w" or staged.is_dir():
            staged.replace(path)
        else:
            # Exclusive publication for files without an overwrite race.
            os.link(staged, path)
