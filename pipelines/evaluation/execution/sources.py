"""Execute source discovery/reading and assemble evaluation bundles from rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from datakit import io
from datakit.evaluation.contracts import (
    DataBundle,
    DataIndex,
    DataRequest,
    Provenance,
    SemanticMetadata,
)
from datakit.evaluation.errors import DecodeError, DiscoveryError
from datakit.io._description import load_description
from datakit.io.backends.file_collection import Record as GriddedRecord
from datakit.io.backends.file_collection import read_records
from datakit.io.backends.station_tables import parse_station_file_time
from datakit.schema.dataset_profiles import DEFAULT_GRIB_FILTERS, DatasetProfile
from datakit.schema.mapping import (
    CLIMATOLOGY_MAPPING,
    DIAMOND_STATION_MAPPING,
    DatasetMapping,
    VarSpec,
    apply_dataset_mapping,
)
from datakit.schema.semantics import DataKind, TemporalKind


def build_bundle(
    request: DataRequest,
    dataset,
    input_files: Sequence[str],
    reader_id: str,
    reader_version: str,
    *,
    kind: DataKind,
    mapping: DatasetMapping | None = None,
    grid_type: str = "regular_latlon",
    temporal_kind=None,
    prefer_dataset_units: bool = True,
    timezone: str = "UTC",
    standard_vars=None,
    source_variables=None,
) -> DataBundle:
    """Wrap prepared data with schema metadata and the caller's evaluation role."""
    units, temporal, names = {}, {}, {}
    for name, variable in dataset.data_vars.items():
        name = str(name)
        canonical = name
        if mapping is not None and mapping.spec_for(name) is None:
            canonical = mapping.standard_name_for(name) or name
        spec = mapping.spec_for(canonical) if mapping is not None else None
        declared = spec.unit if spec is not None else "unknown"
        units[name] = str(
            (variable.attrs.get("units") or declared)
            if prefer_dataset_units
            else declared
        )
        temporal[name] = (
            spec.temporal_kind if spec is not None else TemporalKind.INSTANTANEOUS
        )
        names[name] = canonical
    if isinstance(temporal_kind, Mapping):
        temporal = {
            name: temporal_kind.get(name, TemporalKind.INSTANTANEOUS) for name in names
        }
    elif temporal_kind is not None:
        temporal = temporal_kind
    elif mapping is None:
        temporal = TemporalKind.INSTANTANEOUS
    member_dim = mapping.member_dim if mapping is not None else None
    return DataBundle(
        payload=dataset,
        kind=kind,
        source_id=request.source_id,
        standard_vars=dict(standard_vars) if standard_vars is not None else names,
        semantic=SemanticMetadata(
            units=units,
            temporal_kind=temporal,
            grid_type=grid_type,
            member_count=int(dataset.sizes[member_dim])
            if member_dim in dataset.sizes
            else None,
            timezone=timezone,
        ),
        provenance=Provenance(
            input_files=list(dict.fromkeys(input_files)),
            reader_id=reader_id,
            reader_version=reader_version,
            source_variables=dict(source_variables)
            if source_variables is not None
            else None,
        ),
    )


def _gridded_description(layout):
    names = {
        "fuxi": "fuxi",
        "fuxi_phys": "fuxi",
        "fuxi_ens": "fuxi_ensemble",
        "fuxi_ens_phys": "fuxi_ensemble",
        "fengqing": "fengqing_ensemble",
        "fengqing_phys": "fengqing_ensemble",
        "cra": "cra_analysis",
    }
    description = load_description(names[layout.name])
    if "lead_time" in description and "hours_per_index" in description["lead_time"]:
        description["lead_time"]["hours_per_index"] = layout.step_hours
    return description


class GriddedCatalog:
    def __init__(self, root_dir: Path, layout: DatasetProfile):
        self.root_dir = Path(root_dir)
        self.layout = layout
        self.source = io.open(root_dir, _gridded_description(layout))

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
        mapping = DatasetMapping(
            variables=layout.variables,
            squeeze_dims=layout.squeeze_dims,
            transpose_dims=layout.transpose_dims,
            member_dim=layout.member_dim,
            ensure_member_dim=layout.ensure_member_dim,
            case_insensitive=layout.case_insensitive,
        )
        self.mapping = mapping

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
        description = _gridded_description(self.layout)
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


class DiamondStationCatalog:
    def __init__(self, station_dir: Path, station_whitelist: list[int] | None = None):
        self.station_dir = Path(station_dir)
        self.station_whitelist = station_whitelist

    def file_time(self, path) -> datetime:
        return parse_station_file_time(path)

    def discover(self, request: DataRequest) -> DataIndex:
        for description in ("diamond_station_flat", "diamond_station"):
            try:
                records = io.open(self.station_dir, description).discover(
                    all_files=True
                )
                return DataIndex(
                    source_id=request.source_id,
                    available=[record.path for record in records],
                )
            except FileNotFoundError:
                continue
        raise DiscoveryError("No station files found", source_id=request.source_id)


class DiamondStationReader:
    def __init__(
        self, source_id="diamond_station", version="1.0.0", station_whitelist=None
    ):
        self.source_id = source_id
        self.version = version
        self.station_whitelist = station_whitelist

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        from datakit.io.backends.station_tables import read_records as read_tables

        if not index.available:
            raise DecodeError("No files in index", source_id=request.source_id)
        records = []
        for path in index.available:
            try:
                records.append(
                    GriddedRecord(Path(path), {"time": parse_station_file_time(path)})
                )
            except ValueError:
                continue
        if not records:
            raise DecodeError(
                "No valid station data parsed", source_id=request.source_id
            )
        try:
            dataset = read_tables(
                records,
                {"implementation": "diamond_station"},
                stations=self.station_whitelist,
                fixed_station_metadata=True,
            )
        except Exception as exc:
            raise DecodeError(str(exc), source_id=request.source_id, cause=exc) from exc
        dataset = dataset.assign_coords(station_id=("station", dataset.station.values))
        dataset = dataset[["precipitation"]]
        dataset.precipitation.attrs.update(units="mm", long_name="Hourly precipitation")
        return build_bundle(
            request,
            dataset,
            [str(path) for path in index.available],
            self.source_id,
            self.version,
            kind=DataKind.STATION_OBSERVATION,
            mapping=DIAMOND_STATION_MAPPING,
            grid_type="stations",
            temporal_kind=TemporalKind.INTERVAL_ACCUMULATION,
        )


class ClimatologyCatalog:
    def __init__(
        self, root_dir: Path, template="ART_ATM_GLB_0P25_CLI_ANAL_{mmdd}{hh}.grib2"
    ):
        self.root_dir = Path(root_dir)
        self.template = template

    def file_name(self, moment: datetime) -> str:
        return self.template.format(
            mmdd=moment.strftime("%m%d"), hh=moment.strftime("%H")
        )

    def discover(self, request: DataRequest) -> DataIndex:
        mapping = {}
        for moment in request.init_times or []:
            filename = self.template.replace("{mmdd}", "{cycle_time:%m%d}").replace(
                "{hh}", "{cycle_time:%H}"
            )
            for suffix in (".grib2", ".grib", ".nc"):
                description = load_description("cra_climatology")
                description["layout"]["filename"] = str(
                    Path(filename).with_suffix(suffix)
                )
                description["example"] = str(
                    Path(self.file_name(datetime(2000, 1, 1))).with_suffix(suffix)
                )
                try:
                    records = io.open(self.root_dir, description).discover(
                        selection={"cycle_time": [moment.strftime("%m%d%H")]}
                    )
                    mapping[moment] = records[0].path
                    break
                except FileNotFoundError:
                    continue
        if not mapping:
            raise DiscoveryError(
                "No climatology files for requested times", source_id=request.source_id
            )
        return DataIndex(source_id=request.source_id, available=[mapping])


class ClimatologyReader:
    def __init__(
        self,
        source_id="climatology",
        version="1.0.0",
        engine="cfgrib",
        grib_filters=None,
        units=None,
    ):
        self.source_id = source_id
        self.version = version
        self.engine = engine
        self.grib_filters = grib_filters or DEFAULT_GRIB_FILTERS
        self.mapping = (
            DatasetMapping(
                variables={name: VarSpec(unit=unit) for name, unit in units.items()}
            )
            if units
            else CLIMATOLOGY_MAPPING
        )
        self.units = {name: spec.unit for name, spec in self.mapping.variables.items()}

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        if not index.available:
            raise DecodeError("No files in index", source_id=request.source_id)
        mapping = DatasetMapping(
            variables={name: VarSpec() for name in request.variables},
        )

        def prepare(dataset, names):
            return apply_dataset_mapping(dataset, mapping, names or request.variables)

        records = [
            GriddedRecord(path, {"valid_time": moment})
            for moment, path in index.available[0].items()
        ]
        options = {"load_partitions": True, "prepare": prepare}
        if self.engine == "cfgrib":
            options.update(
                engine="cfgrib",
                variable_filters={
                    name: self.grib_filters[name]
                    for name in request.variables
                    if name in self.grib_filters
                },
            )
        try:
            dataset = read_records(records, {"structure": {}}, **options)
        except Exception as exc:
            raise DecodeError(str(exc), source_id=request.source_id, cause=exc) from exc
        return build_bundle(
            request,
            dataset,
            [str(record.path) for record in records],
            self.source_id,
            self.version,
            kind=DataKind.REFERENCE,
            mapping=self.mapping,
            temporal_kind=TemporalKind.INSTANTANEOUS,
            prefer_dataset_units=False,
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
