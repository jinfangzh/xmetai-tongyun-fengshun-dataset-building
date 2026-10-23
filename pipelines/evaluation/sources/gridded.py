"""Adapt described gridded collections to evaluation requests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from datakit import io
from datakit.evaluation.contracts import DataBundle, DataIndex, DataRequest
from datakit.evaluation.errors import DecodeError, DiscoveryError
from datakit.io import profile_layout
from datakit.io.backends.file_collection import Record as GriddedRecord
from datakit.io.backends.file_collection import read_records
from datakit.schema.mapping import apply_dataset_mapping, profile_mapping

from ._bundle import build_bundle
from ._profile import DatasetProfile


class GriddedCatalog:
    def __init__(self, root_dir: Path, layout: DatasetProfile):
        self.root_dir = Path(root_dir)
        self.layout = layout
        self.source = io.open(root_dir, profile_layout(layout))

    def discover(self, request: DataRequest) -> DataIndex:
        if not request.init_times:
            raise DiscoveryError(
                "Gridded discovery requires init_times", source_id=request.source_id
            )
        axis = "init_time" if self.layout.time_kind == "forecast" else "time"
        selection = {axis: request.init_times}
        if request.lead_times:
            selection["lead_time"] = request.lead_times
        try:
            records = self.source.discover(selection=selection)
        except (FileNotFoundError, ValueError) as exc:
            raise DiscoveryError(str(exc), source_id=request.source_id) from exc
        groups = self.layout.groups_for(request.variables)
        # CRA's public layout uses atmosphere/land rather than legacy group labels.
        groups = {
            "atmosphere"
            if name == "ATM"
            else "land"
            if name == "SURFACE" and self.layout.name == "cra"
            else name
            for name in groups
        }
        if groups:
            records = [
                record for record in records if record.fields.get("group") in groups
            ]
        if not records:
            raise DiscoveryError(
                "No files for requested groups", source_id=request.source_id
            )
        return DataIndex(source_id=request.source_id, available=records)


class GriddedReader:
    def __init__(self, source_id: str, layout: DatasetProfile, version: str = "2.0.0"):
        self.source_id = source_id
        self.version = version
        self.layout = layout
        self.step_hours = layout.step_hours
        self.mapping = profile_mapping(layout)

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        return self._read_records(request, self.records_from_index(index))

    def read_one(
        self, request: DataRequest, index: DataIndex, init_time: datetime
    ) -> DataBundle:
        records = [
            record
            for record in self.records_from_index(index)
            if record.init_time == init_time
        ]
        if not records:
            raise DecodeError(
                f"No {self.layout.name} files for init time {init_time}",
                source_id=request.source_id,
            )
        return self._read_records(request, records)

    def _read_records(
        self, request: DataRequest, records: Sequence[GriddedRecord]
    ) -> DataBundle:
        description = profile_layout(self.layout)
        description[
            "structure"
        ] = {}  # Schema conversion already handles singleton axes.
        if self.layout.time_kind != "forecast":
            records = [
                GriddedRecord(record.path, {"valid_time": record.init_time})
                for record in records
            ]

        def prepare(dataset, names):
            return apply_dataset_mapping(
                dataset, self.mapping, names or request.variables
            )

        options = {
            "prepare": prepare,
            "load_partitions": True,
            "engine_fallback": self.layout.engine_fallback,
        }
        if self.layout.engine == "cfgrib":
            options.update(
                engine="cfgrib",
                variable_filters={
                    name: self.layout.grib_filters[name]
                    for name in request.variables
                    if name in self.layout.grib_filters
                },
            )
        try:
            combined = read_records(records, description, **options)
        except Exception as exc:
            raise DecodeError(str(exc), source_id=request.source_id, cause=exc) from exc
        return build_bundle(
            request,
            combined,
            [str(record.path) for record in records],
            self.source_id,
            self.version,
            kind=self.layout.kind,
            mapping=self.mapping,
        )

    def records_from_index(self, index: DataIndex) -> list[GriddedRecord]:
        if not index.available:
            raise DecodeError("No files in index", source_id=self.source_id)
        return list(index.available)

    def max_lead_hours(self, index: DataIndex) -> float:
        leads = [
            record.lead_time
            for record in self.records_from_index(index)
            if record.lead_time is not None
        ]
        return float(max(leads)) if leads else 0.0

    def available_init_times(self, index: DataIndex) -> list[datetime]:
        return sorted(
            {
                record.init_time
                for record in self.records_from_index(index)
                if record.init_time is not None
            }
        )

    def input_files(self, index: DataIndex) -> list[str]:
        return list(
            dict.fromkeys(str(record.path) for record in self.records_from_index(index))
        )
