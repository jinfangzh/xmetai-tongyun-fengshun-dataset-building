"""Uniform raw dataset access through explicit file layouts and shared backends."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import xarray as xr

from ._layout import get_layout, templates
from .backends import file_collection, files
from .backends._writing import reading
from .backends.file_collection import discover
from .backends.files import as_dataset, local_path
from .records import FileRecord


class _DatasetAccess:
    """Bind a root and structural layout; construction performs no data IO.

    ``read`` returns a closeable Dataset. Multi-file arrays default to Dask-backed
    access (install the ``parallel`` extra); text tables are eager. Call close()
    or use ``with source.read() as ds``. No physical transforms are performed.
    """

    def __init__(
        self,
        path: str | Path,
        description: str | Path | Mapping[str, Any] | None = None,
        *,
        selection: Mapping[str, Any] | None = None,
        backend=None,
    ):
        self.path = local_path(path)
        self.selection = dict(selection or {})
        self.layout = (
            get_layout(description)
            if description is not None
            else {
                "name": "xarray_dataset",
                "layout": {},
                "example": self.path.name,
            }
        )
        if backend is not None:
            self._backend = backend
        elif self.layout.get("format", {}).get("name") in {
            "diamond_station",
            "station_probability",
        }:
            from .backends import station_tables

            self._backend = station_tables
        elif self.layout.get("format", {}).get("name") == "babj":
            from .backends import babj

            self._backend = babj
        else:
            self._backend = file_collection if self.layout["layout"] else files

    def _records(self, selection, all_files, records=None):
        selection = {**self.selection, **(selection or {})}
        if records is not None:
            records = list(records)
            if not records:
                raise ValueError("records must not be empty")
            paths = set()
            for record in records:
                if not isinstance(record, FileRecord):
                    raise TypeError("records must contain FileRecord objects")
                if not record.path.exists():
                    raise FileNotFoundError(record.path)
                path = record.path.resolve()
                if path in paths:
                    raise ValueError(f"Duplicate file record: {record.path}")
                paths.add(path)
            return records, selection
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
        self, *, records=None, selection=None, all_files=False, load=False, **open_options: Any
    ) -> xr.Dataset:
        """Locate selected files and return a closeable Dataset, lazy where supported.

        Coordinate selectors use Xarray names, e.g. time/init_time, lead_time and
        member. Datetimes may be datetime objects or ISO strings. Described
        collections require selection or explicit all_files=True.
        """
        explicit = records is not None
        records, selection = self._records(selection, all_files, records)
        backend = self._record_backend(records)
        stack = ExitStack()
        try:
            stack.enter_context(reading(record.path for record in records))
            dataset = backend.read_records(records, self.layout, **open_options)
            stack.callback(dataset.close)
            selected = dataset.sel(selection) if selection else dataset.copy(deep=False)
            if explicit and any(size == 0 for size in selected.sizes.values()):
                raise KeyError("selection matches no data in the supplied records")
            selected.set_close(stack.close)
            if load:
                selected.load()
                selected.close()
            return selected
        except BaseException:
            stack.close()
            raise

    def write(self, dataset: xr.Dataset, **write_options: Any) -> list[str]:
        if not isinstance(dataset, (xr.Dataset, xr.DataArray)):
            raise TypeError("write requires an xarray.Dataset or DataArray")
        dataset = as_dataset(dataset)
        return self._backend.write(dataset, self.path, self.layout, **write_options)

    def _record_backend(self, records):
        if self._backend is files and (len(records) > 1 or any(record.fields for record in records)):
            return file_collection
        return self._backend

    def iter_datasets(self, *, records=None, selection=None, all_files=False, **open_options):
        """Visit one physical file at a time; each context closes before advancing.

        These are file partitions, not complete forecast/member groups. A yielded
        lazy Dataset must be consumed or loaded before advancing the iterator.
        """
        explicit = records is not None
        records, selection = self._records(selection, all_files, records)
        backend = self._record_backend(records)
        matched = {
            name: set() for name, value in selection.items() if not isinstance(value, slice)
        }
        yielded = False
        for record in records:
            with (
                reading([record.path]),
                backend.read_records([record], self.layout, **open_options) as ds,
            ):
                # A partition may contain no requested internal labels.
                indexers = {}
                for name, value in selection.items():
                    if name not in ds.indexes or isinstance(value, slice):
                        indexers[name] = value
                        continue
                    values = value if isinstance(value, (list, tuple)) else [value]
                    index = ds.indexes[name]
                    positions = index.get_indexer(values)
                    matched[name].update(i for i, position in enumerate(positions) if position >= 0)
                    indexers[name] = index[positions[positions >= 0]]
                selected = ds.sel(indexers) if indexers else ds
                if all(selected.sizes.values()):
                    for name in matched.keys() - ds.indexes.keys():
                        value = selection[name]
                        count = len(value) if isinstance(value, (list, tuple)) else 1
                        matched[name].update(range(count))
                    yielded = True
                    yield selected
        if explicit:
            for name, positions in matched.items():
                value = selection[name]
                count = len(value) if isinstance(value, (list, tuple)) else 1
                if len(positions) != count:
                    raise KeyError(f"selection for {name!r} is not covered by the supplied records")
            if not yielded:
                raise KeyError("selection matches no data in the supplied records")


def open(
    path: str | Path,
    description: str | Path | Mapping[str, Any] | None = None,
    *,
    selection: Mapping[str, Any] | None = None,
    backend=None,
) -> _DatasetAccess:
    """Configure dataset access without discovering or opening data files.

    Call read() or iter_datasets() on the returned object to access data.
    ``description`` is a mapping or YAML path; only file rules are applied.
    Specialized callers may supply a backend with read_records() and write().
    """
    return _DatasetAccess(path, description, selection=selection, backend=backend)


__all__ = ["open"]
