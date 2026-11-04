from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from datakit import io
from datakit.evaluation.contracts import DataRequest
from datakit.schema.mapping import DatasetMapping, VarSpec
from datakit.schema.semantics import DataKind, TemporalKind
from pipelines.evaluation.sources import (
    ClimatologyCatalog,
    ClimatologyReader,
)
from pipelines.evaluation.sources._bundle import build_bundle
from pipelines.evaluation.sources.dataset_profiles import FUXI_LAYOUT


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


def test_grib_fields_are_mapped_before_merge_and_handles_close(tmp_path, monkeypatch):
    from pipelines.evaluation.sources import gridded

    closed = []

    @contextmanager
    def decode(**options):
        level = options["backend_kwargs"]["filter_by_keys"]["level"]
        try:
            # Both GRIB groups decode to the same raw variable name.
            yield xr.Dataset({"t": ("latitude", [float(level)])}, coords={"latitude": [30.0]})
        finally:
            closed.append(level)

    class Access:
        read = staticmethod(decode)

    monkeypatch.setattr(gridded.io, "open", lambda path: Access())
    moment = datetime(2025, 1, 1)
    records = [io.FileRecord(tmp_path / "input.grib", {"init_time": moment})]
    mapping = DatasetMapping(variables={
        "t2m": VarSpec(offset=273.15, unit="K"),
        "t850": VarSpec(scale=2.0, unit="K"),
    })
    result = gridded.read_mapped_records(
        records, None, mapping, ["t2m", "t850"], engine="cfgrib",
        variable_filters={"t2m": {"level": 2}, "t850": {"level": 850}},
    )
    np.testing.assert_allclose(result.t2m, 275.15)
    np.testing.assert_allclose(result.t850, 1700.0)
    assert "lat" in result.dims
    assert closed == [2, 850]


def test_real_grib_fields_decode_map_and_combine(tmp_path):
    ec = pytest.importorskip("eccodes")
    pytest.importorskip("cfgrib")
    from pipelines.evaluation.sources.gridded import read_mapped_records

    path = tmp_path / "fields.grib2"
    with path.open("wb") as stream:
        for short_name, value in (("t", 260.0), ("u", 280.0)):
            handle = ec.codes_grib_new_from_samples("regular_ll_pl_grib2")
            try:
                for key, item in {
                    "Ni": 2, "Nj": 2,
                    "latitudeOfFirstGridPointInDegrees": 31.0,
                    "latitudeOfLastGridPointInDegrees": 30.0,
                    "longitudeOfFirstGridPointInDegrees": 120.0,
                    "longitudeOfLastGridPointInDegrees": 121.0,
                    "iDirectionIncrementInDegrees": 1.0,
                    "jDirectionIncrementInDegrees": 1.0,
                    "shortName": short_name, "level": 500,
                    "dataDate": 20250101, "dataTime": 0,
                }.items():
                    ec.codes_set(handle, key, item)
                ec.codes_set_values(handle, np.full(4, value))
                ec.codes_write(handle, stream)
            finally:
                ec.codes_release(handle)
    mapping = DatasetMapping(
        variables={"t500": VarSpec(unit="K"), "u500": VarSpec(scale=2.0, unit="m/s")},
    )
    record = io.FileRecord(path, {"valid_time": datetime(2025, 1, 1)})
    result = read_mapped_records(
        [record], None, mapping, ["t500", "u500"], engine="cfgrib",
        variable_filters={
            "t500": {"shortName": "t", "typeOfLevel": "isobaricInhPa", "level": 500},
            "u500": {"shortName": "u", "typeOfLevel": "isobaricInhPa", "level": 500},
        },
    )
    np.testing.assert_allclose(result.t500, 260.0)
    np.testing.assert_allclose(result.u500, 560.0)
    assert result.sizes["valid_time"] == 1
    assert result.sizes["lat"] == result.sizes["lon"] == 2
