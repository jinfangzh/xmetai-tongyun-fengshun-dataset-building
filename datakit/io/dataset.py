"""Uniform raw dataset access through explicit file layouts and shared backends."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import xarray as xr

from ._layout import get_layout, load_layout, profile_layout, templates
from .backends import file_collection, files
from .backends.file_collection import discover
from .backends.files import as_dataset, local_path


class _DatasetAccess:
    """Bind a root and structural layout; construction performs no data IO.

    ``read`` returns a closeable Dataset. Multi-file arrays default to Dask-backed
    access (install the ``parallel`` extra); text tables are eager. Call close()
    or use ``with source.read() as ds``. No physical transforms are performed.
    """

    def __init__(
        self,
        path: str | Path,
        layout: str | Path | Mapping[str, Any] | None = None,
        *,
        selection: Mapping[str, Any] | None = None,
        backend=None,
    ):
        self.path = local_path(path)
        self.selection = dict(selection or {})
        self.layout = (
            load_layout(layout)
            if layout is not None
            else {
                "name": "xarray_dataset",
                "layout": {},
            }
        )
        self._backend = (
            backend
            if backend is not None
            else (file_collection if self.layout["layout"] else files)
        )

    def _records(self, selection, all_files):
        selection = {**self.selection, **(selection or {})}
        if self.layout["layout"] and not selection and not all_files:
            raise ValueError(
                "Provide selection for a described dataset, or explicitly use all_files=True"
            )
        # Path fields narrow discovery; remaining coordinates are selected by Xarray.
        path_fields = {}
        for _, template in templates(self.layout):
            for field, spec, _ in template.fields:
                value = selection.get(field)
                if value is None or isinstance(value, slice):
                    continue
                values = value if isinstance(value, (list, tuple)) else [value]
                if "%" in spec:
                    values = [
                        (
                            datetime.strptime("2000" + item, "%Y%m%d%H")  # noqa: DTZ007 -- source cycle label
                            if field == "cycle_time" and isinstance(item, str)
                            else datetime.fromisoformat(item)
                            if isinstance(item, str)
                            else item
                        )
                        for item in values
                    ]
                path_fields[field] = values
        if "lead_time" in selection and "lead_index" not in path_fields:
            step = self.layout.get("lead_time", {}).get("hours_per_index")
            value = selection["lead_time"]
            if step and not isinstance(value, slice):
                values = value if isinstance(value, (list, tuple)) else [value]
                if any(
                    float(item) / step != int(float(item) / step) for item in values
                ):
                    raise ValueError(
                        "lead_time must match the layout's hours_per_index"
                    )
                path_fields["lead_index"] = [int(float(item) / step) for item in values]
        return discover(self.path, self.layout, path_fields=path_fields), selection

    def discover(self, *, selection=None, all_files=False):
        """Return matching file records without opening data files."""
        records, _ = self._records(selection, all_files)
        return records

    def read(
        self, *, selection=None, all_files=False, load=False, **open_options: Any
    ) -> xr.Dataset:
        """Locate selected files and return a closeable Dataset, lazy where supported.

        Coordinate selectors use Xarray names, e.g. time/init_time, lead_time and
        member. Datetimes may be datetime objects or ISO strings. Described
        collections require selection or explicit all_files=True.
        """
        records, selection = self._records(selection, all_files)
        dataset = self._backend.read_records(records, self.layout, **open_options)
        try:
            selected = dataset.sel(selection) if selection else dataset
            if selected is not dataset:
                selected.set_close(dataset.close)
            if load:
                selected.load()
                dataset.close()
            return selected
        except BaseException:
            dataset.close()
            raise

    def write(self, dataset: xr.Dataset, **write_options: Any) -> list[str]:
        if not isinstance(dataset, (xr.Dataset, xr.DataArray)):
            raise TypeError("write requires an xarray.Dataset or DataArray")
        dataset = as_dataset(dataset)
        return self._backend.write(dataset, self.path, self.layout, **write_options)

    def iter_datasets(self, *, selection=None, all_files=False, **open_options):
        """Visit one physical file at a time; each context closes before advancing.

        These are file partitions, not complete forecast/member groups. A yielded
        lazy Dataset must be consumed or loaded before advancing the iterator.
        """
        records, selection = self._records(selection, all_files)
        for record in records:
            with self._backend.read_records(
                [record], self.layout, **open_options
            ) as ds:
                # A partition may contain no requested internal labels.
                indexers = {}
                for name, value in selection.items():
                    if name not in ds.indexes or isinstance(value, slice):
                        indexers[name] = value
                        continue
                    values = value if isinstance(value, (list, tuple)) else [value]
                    index = ds.indexes[name]
                    positions = index.get_indexer(values)
                    indexers[name] = index[positions[positions >= 0]]
                selected = ds.sel(indexers) if indexers else ds
                if all(selected.sizes.values()):
                    yield selected


def open(
    path: str | Path,
    layout: str | Path | Mapping[str, Any] | None = None,
    *,
    selection: Mapping[str, Any] | None = None,
    backend=None,
) -> _DatasetAccess:
    """Configure dataset access without discovering or opening data files.

    Call read() or iter_datasets() on the returned object to access data.
    ``layout`` is an explicit mapping or YAML path, never a dataset name.
    Specialized callers may supply a backend with read_records() and write().
    """
    return _DatasetAccess(path, layout, selection=selection, backend=backend)


__all__ = ["open", "open_dataset"]


def open_dataset(
    path, description, *, selection=None, profile=None, layout=None, backend=None
):
    if layout is not None and profile is not None:
        raise ValueError("Pass layout or profile, not both")
    if layout is None:
        layout = get_layout(description) if profile is None else profile_layout(profile)
    return open(path, layout, selection=selection, backend=backend)
