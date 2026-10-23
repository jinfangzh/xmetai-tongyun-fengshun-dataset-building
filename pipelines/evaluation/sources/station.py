"""Adapt station observations to evaluation requests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from datakit import io
from datakit.evaluation.contracts import DataBundle, DataIndex, DataRequest
from datakit.evaluation.errors import DecodeError, DiscoveryError
from datakit.io import get_layout
from datakit.io.backends.file_collection import Record as GriddedRecord
from datakit.io.backends.station_tables import parse_station_file_time
from datakit.schema.semantics import DataKind, TemporalKind
from examples.dataset_io import observation_diamond

from ._bundle import build_bundle
from .dataset_profiles import DIAMOND_STATION_MAPPING


class DiamondStationCatalog:
    def __init__(self, station_dir: Path, station_whitelist: list[int] | None = None):
        self.station_dir = Path(station_dir)
        self.station_whitelist = station_whitelist

    def file_time(self, path) -> datetime:
        return parse_station_file_time(path)

    def discover(self, request: DataRequest) -> DataIndex:
        for description in (
            observation_diamond.FLAT_DESCRIPTION,
            observation_diamond.DESCRIPTION,
        ):
            try:
                records = io.open(self.station_dir, get_layout(description)).discover(
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
                get_layout(observation_diamond.DESCRIPTION),
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
