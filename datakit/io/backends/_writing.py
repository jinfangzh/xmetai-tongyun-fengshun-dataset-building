"""Local output staging and protection of sources owned by IO contexts."""

from collections import Counter
from contextlib import contextmanager
import os
import shutil
import tempfile
import threading
from pathlib import Path
from uuid import uuid4


_READERS = Counter()
_LOCK = threading.Lock()


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
    if options.get("compute") is False:
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
