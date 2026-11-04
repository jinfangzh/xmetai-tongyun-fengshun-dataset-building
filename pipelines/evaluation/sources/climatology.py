"""Adapt calendar-indexed reference fields to evaluation requests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from datakit import io
from datakit.evaluation.contracts import DataBundle, DataIndex, DataRequest
from datakit.evaluation.errors import DecodeError, DiscoveryError
from datakit.io import get_layout, source_config
from datakit.schema.mapping import DatasetMapping, VarSpec
from datakit.schema.semantics import DataKind, TemporalKind

from ._bundle import build_bundle
from .dataset_profiles import CLIMATOLOGY_MAPPING, DEFAULT_GRIB_FILTERS
from .gridded import read_mapped_records


class ClimatologyCatalog:
    def __init__(
        self, root_dir: Path, template="ART_ATM_GLB_0P25_CLI_ANAL_{mmdd}{hh}.grib2"
    ):
        self.root_dir = Path(root_dir)
        self.template = template
        self._layout = get_layout(source_config("climatology_cra"))

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
                description = get_layout(self._layout)
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

        records = [
            io.FileRecord(path, {"valid_time": moment})
            for moment, path in index.available[0].items()
        ]
        options = {"engine": self.engine}
        if self.engine == "cfgrib":
            options.update(
                variable_filters={
                    name: self.grib_filters[name]
                    for name in request.variables
                    if name in self.grib_filters
                },
            )
        try:
            dataset = read_mapped_records(
                records, None, mapping, request.variables, **options
            )
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
