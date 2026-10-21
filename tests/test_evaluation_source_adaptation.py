from datetime import datetime

import numpy as np
import xarray as xr

from datakit.evaluation.adapters.gridded import rules_for_layout
from datakit.evaluation.contracts import DataRequest
from datakit.io.readers.climatology import ClimatologyDatasetReader, ClimatologyFiles
from datakit.io.readers.layouts import FUXI_LAYOUT
from datakit.schema.semantics import DataKind, TemporalKind
from pipelines.evaluation.execution.sources import (
    ClimatologyCatalog,
    ClimatologyReader,
    build_bundle,
)


def test_climatology_io_and_evaluation_rules(tmp_path):
    times = [
        datetime.fromisoformat("2026-08-20T00:00:00"),
        datetime.fromisoformat("2026-08-20T06:00:00"),
    ]
    source_catalog = ClimatologyFiles(tmp_path)
    for i, moment in enumerate(times):
        xr.Dataset(
            {
                "t2m": (
                    ("latitude", "longitude"),
                    np.full((2, 2), i + 1.0),
                    {"units": "raw"},
                )
            },
            coords={"latitude": [10.0, 20.0], "longitude": [100.0, 110.0]},
        ).to_netcdf(tmp_path / source_catalog.file_name(moment).replace(".grib2", ".nc"))

    mapping = source_catalog.discover_files(times)
    raw, paths = ClimatologyDatasetReader(engine="netcdf").read_files(mapping, ["t2m"])
    assert raw.sizes["valid_time"] == 2
    assert "latitude" in raw.dims
    assert len(paths) == 2

    request = DataRequest(source_id="climatology", variables=["t2m"], init_times=times)
    bundle = ClimatologyReader(engine="netcdf").read(
        request, ClimatologyCatalog(tmp_path).discover(request)
    )
    assert bundle.kind == DataKind.REFERENCE
    assert bundle.payload["t2m"].dims == ("valid_time", "lat", "lon")
    assert bundle.semantic.units == {"t2m": "K"}
    assert bundle.semantic.temporal_kind == TemporalKind.INSTANTANEOUS
    assert bundle.provenance.input_files == paths


def test_gridded_rules_preserve_variable_metadata():
    request = DataRequest(source_id="fuxi", variables=["tp"])
    dataset = xr.Dataset({"tp": (("member", "lat", "lon"), np.ones((2, 1, 1)))})
    rules = rules_for_layout(FUXI_LAYOUT)
    bundle = build_bundle(request, dataset, rules, ["a.nc", "a.nc"], "fuxi", "2.0.0")
    assert bundle.semantic.units["tp"] == "mm"
    assert bundle.semantic.temporal_kind["tp"] == TemporalKind.INTERVAL_ACCUMULATION
    assert bundle.provenance.input_files == ["a.nc"]
