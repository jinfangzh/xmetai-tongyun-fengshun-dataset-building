from datetime import datetime

import numpy as np
import xarray as xr

from datakit import io
from datakit.evaluation.contracts import DataRequest
from datakit.schema.dataset_profiles import FUXI_LAYOUT
from datakit.schema.mapping import DatasetMapping
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
    source_catalog = ClimatologyCatalog(tmp_path)
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

    mapping = source_catalog.discover(
        DataRequest(source_id="climatology", variables=["t2m"], init_times=times)
    ).available[0]
    frames = []
    paths = []
    for moment, path in mapping.items():
        with io.open(path).read() as dataset:
            frames.append(dataset.load().expand_dims(valid_time=[moment]))
        paths.append(str(path))
    raw = xr.concat(frames, dim="valid_time")
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
    mapping = DatasetMapping(variables=FUXI_LAYOUT.variables)
    bundle = build_bundle(
        request, dataset, ["a.nc", "a.nc"], "fuxi", "2.0.0", kind=FUXI_LAYOUT.kind, mapping=mapping
    )
    assert bundle.semantic.units["tp"] == "mm"
    assert bundle.semantic.temporal_kind["tp"] == TemporalKind.INTERVAL_ACCUMULATION
    assert bundle.provenance.input_files == ["a.nc"]
