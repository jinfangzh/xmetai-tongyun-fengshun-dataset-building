"""Adapt station observations to evaluation requests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from datakit import io
from datakit.evaluation.contracts import DataBundle, DataIndex, DataRequest
from datakit.evaluation.errors import DecodeError, DiscoveryError
from datakit.io import get_layout, source_config
from datakit.schema.semantics import DataKind, TemporalKind

from ._bundle import build_bundle
from .dataset_profiles import DIAMOND_STATION_MAPPING


def _station_record(path, descriptions):
    if isinstance(path, io.FileRecord):
        return path
    for description in descriptions:
        try:
            return io.open(path, description).discover(all_files=True)[0]
        except FileNotFoundError:
            continue
    raise ValueError(f"无法解析观测文件时间: {path}")


class DiamondStationCatalog:
    def __init__(self, station_dir: Path, station_whitelist: list[int] | None = None):
        self.station_dir = Path(station_dir)
        self.station_whitelist = station_whitelist
        self._layouts = (
            get_layout(source_config("observation_diamond_flat")),
            get_layout(source_config("observation_diamond")),
        )

    def file_time(self, path) -> datetime:
        return _station_record(path, self._layouts).fields["time"]

    def discover(self, request: DataRequest) -> DataIndex:
        for description in self._layouts:
            try:
                records = io.open(self.station_dir, get_layout(description)).discover(
                    all_files=True
                )
                return DataIndex(
                    source_id=request.source_id,
                    available=records,
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
        self._layout = get_layout(source_config("observation_diamond"))
        self._layouts = (get_layout(source_config("observation_diamond_flat")), self._layout)

    def read(self, request: DataRequest, index: DataIndex) -> DataBundle:
        if not index.available:
            raise DecodeError("No files in index", source_id=request.source_id)
        records = []
        for path in index.available:
            try:
                records.append(_station_record(path, self._layouts))
            except ValueError:
                continue
        if not records:
            raise DecodeError(
                "No valid station data parsed", source_id=request.source_id
            )
        try:
            with io.open(records[0].path.parent, self._layout).read(
                records=records,
                stations=self.station_whitelist,
                fixed_station_metadata=True,
            ) as raw:
                dataset = raw.load()
        except Exception as exc:
            raise DecodeError(str(exc), source_id=request.source_id, cause=exc) from exc
        dataset = dataset.assign_coords(station_id=("station", dataset.station.values))
        dataset = dataset[["precipitation"]]
        dataset.precipitation.attrs.update(units="mm", long_name="Hourly precipitation")
        return build_bundle(
            request,
            dataset,
            [str(record.path) for record in records],
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
