"""Shared source loading and ownership handling for dataset operations."""

from contextlib import contextmanager

import xarray as xr

from datakit import io


@contextmanager
def _source(source, source_config=None, selection=None, read_options=None):
    options = dict(read_options or {})
    if isinstance(source, xr.Dataset):
        if source_config is not None or options:
            raise ValueError(
                "In-memory sources do not accept source_config/read_options"
            )
        opened = source
        owned = False
    else:
        access = (
            io.open_dataset(source, source_config)
            if source_config is not None
            else io.open(source)
        )
        options.setdefault("all_files", True)
        options.setdefault("selection", selection)
        opened = access.read(**options)
        owned = True
    try:
        dataset = opened.sel(selection) if not owned and selection else opened
        dataset = dataset.copy(deep=False)
        yield dataset
    finally:
        if owned:
            opened.close()
