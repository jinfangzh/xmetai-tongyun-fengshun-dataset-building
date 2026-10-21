"""Execute source discovery/reading and assemble evaluation bundles from rules."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import xarray as xr

from datakit.evaluation.adapters.climatology import climatology_rules
from datakit.evaluation.adapters.gridded import rules_for_layout
from datakit.evaluation.adapters.rules import AdaptationRules
from datakit.evaluation.adapters.station import DIAMOND_STATION_RULES
from datakit.evaluation.contracts import (
    DataBundle,
    DataIndex,
    DataRequest,
    Provenance,
    SemanticMetadata,
)
from datakit.evaluation.errors import DecodeError, DiscoveryError
from datakit.io.readers.climatology import ClimatologyDatasetReader, ClimatologyFiles
from datakit.io.readers.diamond import (
    DiamondStationDatasetReader,
    DiamondStationFiles,
    parse_station_file_time,
)
from datakit.io.readers.errors import SourceDecodeError, SourceDiscoveryError
from datakit.io.readers.gridded import (
    GriddedDatasetReader,
    GriddedFileCatalog,
    GriddedRecord,
)
from datakit.io.readers.layouts import GriddedLayout


def build_bundle(
    request: DataRequest,
    dataset: xr.Dataset,
    rules: AdaptationRules,
    input_files: Sequence[str],
    reader_id: str,
    reader_version: str,
) -> DataBundle:
    """Apply declared metadata to an already decoded dataset."""
    renames = {
        source: target
        for source, target in rules.coordinate_names.items()
        if source in dataset.coords or source in dataset.dims
    }
    if renames:
        dataset = dataset.rename(renames)
    units = {}
    temporal = {}
    for name in dataset.data_vars:
        key = str(name)
        declared = rules.units.get(key, "unknown")
        units[key] = str(
            (dataset[name].attrs.get("units") or declared)
            if rules.prefer_dataset_units
            else declared
        )
        temporal[key] = rules.temporal_kind.get(key, rules.default_temporal_kind)
    member_count = None
    if rules.member_dim and rules.member_dim in dataset.sizes:
        member_count = int(dataset.sizes[rules.member_dim])
    return DataBundle(
        payload=dataset,
        kind=rules.kind,
        source_id=request.source_id,
        standard_vars={
            str(name): rules.variable_names.get(str(name), str(name))
            for name in dataset.data_vars
        },
        semantic=SemanticMetadata(
            units=units,
            temporal_kind=(
                temporal if rules.temporal_per_variable else rules.default_temporal_kind
            ),
            grid_type=rules.grid_type,
            member_count=member_count,
        ),
        provenance=Provenance(
            input_files=list(dict.fromkeys(input_files)),
            reader_id=reader_id,
            reader_version=reader_version,
        ),
    )


class GriddedCatalog:
    def __init__(self, root_dir: Path, layout: GriddedLayout):
        self.source = GriddedFileCatalog(root_dir, layout)
        self.root_dir = self.source.root_dir
        self.layout = layout

    def discover(self, request: DataRequest) -> DataIndex:
        try:
            records = self.source.discover_records(request)
        except SourceDiscoveryError as exc:
            raise DiscoveryError(str(exc), source_id=request.source_id) from exc
        return DataIndex(source_id=request.source_id, available=records)


class GriddedReader:
    def __init__(self, source_id: str, layout: GriddedLayout, version: str = "2.0.0"):
        self.source_id = source_id
        self.version = version
        self.layout = layout
        self.step_hours = layout.step_hours
        self.source = GriddedDatasetReader(source_id, layout)
        self.rules = rules_for_layout(layout)

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
        try:
            combined, frames = self.source.read_records(records, request.variables)
        except SourceDecodeError as exc:
            raise DecodeError(
                str(exc),
                source_id=request.source_id,
                path=exc.path or None,
                cause=exc.cause,
            ) from exc
        return build_bundle(
            request,
            combined,
            self.rules,
            [str(record.path) for record, _ in frames],
            self.source_id,
            self.version,
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


class DiamondStationCatalog:
    def __init__(self, station_dir: Path, station_whitelist: list[int] | None = None):
        self.source = DiamondStationFiles(station_dir, station_whitelist)
        self.station_dir = self.source.station_dir
        self.station_whitelist = self.source.station_whitelist

    def file_time(self, path) -> datetime:
        return parse_station_file_time(path)

    def discover(self, request: DataRequest) -> DataIndex:
        try:
            files = self.source.discover_files(request.source_id)
        except SourceDiscoveryError as exc:
            raise DiscoveryError(str(exc), source_id=request.source_id) from exc
        return DataIndex(source_id=request.source_id, available=files)


class DiamondStationReader:
    def __init__(
        self,
        source_id: str = "diamond_station",
        version: str = "1.0.0",
        station_whitelist: list[int] | None = None,
    ):
        self.source_id = source_id
        self.version = version
        self.source = DiamondStationDatasetReader(source_id, version, station_whitelist)
        self.station_whitelist = self.source.station_whitelist

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        try:
            dataset, files = self.source.read_files(index.available)
        except SourceDecodeError as exc:
            raise DecodeError(
                str(exc),
                source_id=request.source_id,
                path=exc.path or None,
                cause=exc.cause,
            ) from exc
        return build_bundle(
            request, dataset, DIAMOND_STATION_RULES, files, self.source_id, self.version
        )


class ClimatologyCatalog:
    def __init__(
        self,
        root_dir: Path,
        template: str = "ART_ATM_GLB_0P25_CLI_ANAL_{mmdd}{hh}.grib2",
    ):
        self.source = ClimatologyFiles(root_dir, template)
        self.root_dir = self.source.root_dir
        self.template = template

    def file_name(self, moment: datetime) -> str:
        return self.source.file_name(moment)

    def discover(self, request: DataRequest) -> DataIndex:
        try:
            mapping = self.source.discover_files(request.init_times or [])
        except SourceDiscoveryError as exc:
            raise DiscoveryError(str(exc), source_id=request.source_id) from exc
        return DataIndex(source_id=request.source_id, available=[mapping])


class ClimatologyReader:
    def __init__(
        self,
        source_id: str = "climatology",
        version: str = "1.0.0",
        engine: str = "cfgrib",
        grib_filters: dict[str, dict[str, Any]] | None = None,
        units: dict[str, str] | None = None,
    ):
        self.source_id = source_id
        self.version = version
        self.source = ClimatologyDatasetReader(engine, grib_filters)
        self.engine = engine
        self.grib_filters = self.source.grib_filters
        self.rules = climatology_rules(units or None)
        self.units = self.rules.units

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        if not index.available:
            raise DecodeError("No files in index", source_id=request.source_id)
        try:
            dataset, files = self.source.read_files(
                index.available[0], request.variables
            )
        except SourceDecodeError as exc:
            raise DecodeError(
                str(exc),
                source_id=request.source_id,
                path=exc.path or None,
                cause=exc.cause,
            ) from exc
        return build_bundle(
            request, dataset, self.rules, files, self.source_id, self.version
        )


def load_station_whitelist(value: Any | None) -> list[int] | None:
    """Load station IDs from a sequence or a whitespace-delimited file."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [int(station_id) for station_id in value]
    path = Path(value)
    if not path.exists():
        raise ValueError(f"站点白名单文件不存在: {path}")
    station_ids = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            try:
                station_ids.add(int(float(fields[0])))
            except (TypeError, ValueError):
                continue
    if not station_ids:
        raise ValueError(f"站点白名单为空或格式无效: {path}")
    return sorted(station_ids)
