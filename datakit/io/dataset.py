"""Uniform raw dataset access through explicit file layouts and shared backends."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ._layout import get_layout, templates
from .backends import file_collection
from .backends._writing import reading
from .backends.file_collection import discover
from .backends.files import as_dataset, local_path
from .records import FileRecord


def _normalize_selection(selection):
    """Use lists for exact labels and preserve explicit range slices."""
    normalized = {}
    for name, value in selection.items():
        if isinstance(value, slice):
            normalized[name] = value
        elif np.ndim(value) == 0:
            normalized[name] = [value[()] if isinstance(value, np.ndarray) else value]
        elif np.ndim(value) == 1:
            normalized[name] = list(value)
        else:
            raise ValueError(f"Selection for {name!r} must be scalar, one-dimensional or a slice")
        if not isinstance(value, slice) and not normalized[name]:
            raise KeyError(f"Selection for {name!r} contains no labels")
    return normalized


def _select_partition(dataset, selection, *, partial=False):
    """Select by exact index positions; track requested labels without reading arrays."""
    ranges = {name: value for name, value in selection.items() if isinstance(value, slice)}
    selected = dataset.sel(ranges) if ranges else dataset
    indexers, matched = {}, {}
    for name, values in selection.items():
        if isinstance(values, slice):
            continue
        index = selected.get_index(name)
        lookup = values
        if isinstance(index, xr.CFTimeIndex):
            lookup = [
                xr.date_range(start=value, periods=1, calendar=index.calendar, use_cftime=True)[0]
                if isinstance(value, str) else value
                for value in values
            ]
        positions = index.get_indexer(lookup)
        found = np.flatnonzero(positions >= 0)
        if not partial and len(found) != len(values):
            missing = [values[i] for i in np.flatnonzero(positions < 0)]
            raise KeyError(f"Selection labels not found for {name!r}: {missing!r}")
        indexers[name] = positions[found]
        matched[name] = set(found)
    selected = selected.isel(indexers) if indexers else selected.copy(deep=False)
    return selected, matched


def _require_selection_coverage(selection, matched):
    for name, values in selection.items():
        if not isinstance(values, slice) and len(matched[name]) != len(values):
            missing = [value for i, value in enumerate(values) if i not in matched[name]]
            raise KeyError(f"Selection labels not found for {name!r}: {missing!r}")


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
        self._backend = backend if backend is not None else file_collection

    def _records(self, selection, all_files, records=None):
        selection = _normalize_selection({**self.selection, **(selection or {})})
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
                values = value
                if "%" in spec:
                    values = [
                        (
                            datetime.strptime("2000" + item, "%Y%m%d%H")  # noqa: DTZ007 -- source cycle label
                            if field == "cycle_time" and isinstance(item, str)
                            else datetime.fromisoformat(item)
                            if isinstance(item, str)
                            else item.astype("datetime64[us]").item()
                            if isinstance(item, np.datetime64)
                            else item
                        )
                        for item in values
                    ]
                path_fields[field] = values
        if "lead_time" in selection and "lead_index" not in path_fields:
            step = self.layout.get("lead_time", {}).get("hours_per_index")
            value = selection["lead_time"]
            if step and not isinstance(value, slice):
                values = value
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
        member. Scalar labels preserve dimensions, like single-element lists.
        Time labels match exactly; use slices for ranges. Empty results and
        missing labels raise. Described collections require selection or explicit
        all_files=True, unless records are supplied.
        """
        records, selection = self._records(selection, all_files, records)
        backend = self._backend
        stack = ExitStack()
        try:
            stack.enter_context(reading(record.path for record in records))
            dataset = backend.read_records(records, self.layout, **open_options)
            stack.callback(dataset.close)
            selected, _ = _select_partition(dataset, selection)
            if not selected.variables or any(size == 0 for size in selected.sizes.values()):
                raise KeyError("Selection matches no data")
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

    def iter_datasets(self, *, records=None, selection=None, all_files=False, **open_options):
        """Visit one physical file at a time; each context closes before advancing.

        These are file partitions, not complete forecast/member groups. A yielded
        lazy Dataset must be consumed or loaded before advancing the iterator.
        Full consumption checks all requested labels across non-empty partitions;
        missing labels or no matching data raise, with or without explicit records.
        """
        records, selection = self._records(selection, all_files, records)
        backend = self._backend
        matched = {
            name: set() for name, value in selection.items() if not isinstance(value, slice)
        }
        yielded = False
        for record in records:
            with (
                reading([record.path]),
                backend.read_records([record], self.layout, **open_options) as ds,
            ):
                selected, found = _select_partition(ds, selection, partial=True)
                if selected.variables and all(selected.sizes.values()):
                    for name, positions in found.items():
                        matched[name].update(positions)
                    yielded = True
                    yield selected
        _require_selection_coverage(selection, matched)
        if not yielded:
            raise KeyError("Selection matches no data")


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
