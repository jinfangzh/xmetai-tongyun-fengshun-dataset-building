from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from datakit import io
from datakit.evaluation.contracts import DataRequest
from datakit.io import get_description, get_layout
from examples.dataset_io import (
    analysis_cra,
    climatology_cra,
    climatology_daily,
    climatology_station_probability,
    forecast_fengqing,
    forecast_fuxi,
    observation_babj,
    observation_diamond,
)
from pipelines.evaluation.components import (
    _fengqing_source,
    _fuxi_ensemble_source,
    _phys_source,
    _ref_probability_source,
)
from pipelines.evaluation.sources import GriddedCatalog, GriddedReader
from pipelines.evaluation.sources._profile import build_profile
from pipelines.evaluation.sources.dataset_profiles import (
    FENGQING_PHYS_LAYOUT,
    FUXI_ENS_LAYOUT,
    FUXI_ENS_PHYS_LAYOUT,
)
from pipelines.evaluation.sources.reference_probability import RefProbabilitySource


def test_fuxi_raw_read_and_explicit_normalization(tmp_path):
    for member in (0, 1):
        directory = tmp_path / "20250101" / f"member_{member}"
        directory.mkdir(parents=True)
        for index in (1, 2):
            xr.Dataset(
                {
                    "unlisted": (("latitude", "longitude"), [[42.0]]),
                    "Z500": (
                        ("latitude", "longitude"),
                        np.full((1, 1), 9.80665 * (member + index)),
                        {"units": "m^2/s^2"},
                    ),
                },
                coords={"latitude": [30.0], "longitude": [120.0]},
            ).to_netcdf(directory / f"{index:03d}.nc")

    with forecast_fuxi.open(
        tmp_path,
        ensemble=True,
        selection={"init_time": ["2025-01-01"], "lead_time": [12], "member": [1]},
    ).read() as raw:
        assert "unlisted" in raw
        np.testing.assert_allclose(raw.unlisted, 42.0)
        np.testing.assert_allclose(raw.Z500, 3.0 * 9.80665)

    profile = FUXI_ENS_LAYOUT.with_step_hours(3)
    source = forecast_fuxi.open(
        tmp_path,
        profile=profile,
        selection={"init_time": ["2025-01-01"], "lead_time": [6], "member": [1]},
    )
    records = source.discover()
    assert [record.path.relative_to(tmp_path).as_posix() for record in records] == [
        "20250101/member_1/002.nc"
    ]
    with source.read() as raw:
        assert "Z500" in raw and "latitude" in raw.dims
        assert raw.Z500.attrs["units"] == "m^2/s^2"
        normalized = forecast_fuxi.normalize(raw, ["z500"], profile=profile)
        assert "lat" in normalized.dims
        assert normalized.z500.attrs["units"] == "m"
        np.testing.assert_allclose(normalized.z500, 3.0)
        physical = forecast_fuxi.normalize(raw, ["z500"], profile=FUXI_ENS_PHYS_LAYOUT)
        np.testing.assert_allclose(physical.z500, 3.0 * 9.80665)
        assert "Z500" in raw and "latitude" in raw.dims

    request = DataRequest(
        source_id="fuxi_ens",
        variables=["z500"],
        init_times=[datetime(2025, 1, 1)],
        lead_times=[6],
    )
    for factory, expected in (
        (_fuxi_ensemble_source, [2.0, 3.0]),
        (_phys_source(FUXI_ENS_PHYS_LAYOUT, "fuxi_ens"), [2.0 * 9.80665, 3.0 * 9.80665]),
    ):
        source = factory(root_dir=tmp_path, step_hours=3)
        bundle = source.reader.read(request, source.catalog.discover(request))
        np.testing.assert_allclose(bundle.payload.z500.values.reshape(-1), expected)


def test_layout_customization_is_local_and_open_does_not_scan(tmp_path):
    first = get_layout(forecast_fuxi.ENSEMBLE_DESCRIPTION)
    first["layout"]["directory"] = "{init_time:%Y%m%d}/member_{member:03d}"
    assert (
        "member_{member:d}" in get_layout(forecast_fuxi.ENSEMBLE_DESCRIPTION)["layout"]["directory"]
    )
    source = forecast_fuxi.open(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        source.discover(all_files=True)
    with pytest.raises(ValueError, match="examples.dataset_io"):
        io.open(tmp_path, "fuxi")


def test_diamond_raw_columns_normalization_and_writeback(tmp_path):
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    (source_dir / "2025010106.000").write_text(
        "diamond 3 station\n2025 01 01 06\n54511 116.4 39.9 20 5.5\n",
        encoding="gbk",
    )
    with observation_diamond.open(source_dir, flat=True).read(all_files=True) as raw:
        assert "precipitation" in raw and "tp" not in raw
        normalized = observation_diamond.normalize(raw, ["tp"])
        np.testing.assert_allclose(normalized.tp, [[5.5]])
        assert normalized.tp.attrs["units"] == "mm"
        assert "units" not in raw.precipitation.attrs
        assert "precipitation" in raw
        outputs = observation_diamond.open(tmp_path / "output", flat=True).write(raw)
    with observation_diamond.open(tmp_path / "output", flat=True).read(all_files=True) as copied:
        np.testing.assert_allclose(copied.precipitation, [[5.5]])
    assert len(outputs) == 1


def test_probability_contract_used_by_io_and_evaluation(tmp_path):
    (tmp_path / "010106.000").write_text("54511 0.9 0.6 0.3 0.1\n")
    with climatology_station_probability.open(tmp_path).read(all_files=True) as raw:
        assert raw.cycle_time.values.tolist() == ["010106"]
        np.testing.assert_allclose(raw.threshold, [0.1, 4.0, 13.0, 25.0])
        np.testing.assert_allclose(raw.probability, [[[0.9, 0.6, 0.3, 0.1]]])
    source = RefProbabilitySource(tmp_path)
    np.testing.assert_allclose(
        source.probabilities(datetime(2025, 1, 1, 6), [54511, 99999]),
        [[0.9, 0.6, 0.3, 0.1], [np.nan, np.nan, np.nan, np.nan]],
    )
    for options in ({}, {"description": "station_probability"}):
        source = _ref_probability_source(root_dir=tmp_path, **options).reader
        np.testing.assert_allclose(
            source.probabilities(datetime(2025, 1, 1, 6), [54511]), [[0.9, 0.6, 0.3, 0.1]]
        )


def test_babj_specialized_backend_roundtrip_preserves_all_leads(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "babj2501.dat").write_text(
        "diamond 7\nTEST 2501\nheader\n2025 1 1 0 0 120 20 990 30\n2025 1 1 0 6 121 21 985 35\n",
        encoding="gbk",
    )
    with observation_babj.open(source).read(all_files=True) as raw:
        assert raw.lead_time.values.tolist() == [0, 6]
        observation_babj.open(tmp_path / "output").write(raw)
    with observation_babj.open(tmp_path / "output").read(all_files=True) as copied:
        xr.testing.assert_equal(raw, copied)


def test_description_controls_mapping_and_evaluation_contract(tmp_path):
    description = get_description(forecast_fuxi.DESCRIPTION)
    description["mapping"]["coordinate_aliases"] = {"y": "lat", "x": "lon"}
    description["mapping"]["variables"] = {
        "t2m": {"source": "temperature", "source_unit": "degC", "unit": "K", "offset": 273.15}
    }
    profile = build_profile(description)
    directory = tmp_path / "20250101"
    directory.mkdir()
    xr.Dataset(
        {"temperature": (("y", "x"), [[20.0]], {"units": "degC"})},
        coords={"y": [30.0], "x": [120.0]},
    ).to_netcdf(directory / "001.nc")
    with forecast_fuxi.open(tmp_path, profile=profile).read(all_files=True) as raw:
        assert raw.temperature.attrs["units"] == "degC"
        normalized = forecast_fuxi.normalize(raw, ["t2m"], profile=profile)
        assert "lat" in normalized.dims
        assert normalized.t2m.attrs["units"] == "K"
        np.testing.assert_allclose(normalized.t2m, 293.15)
    request = DataRequest(source_id="custom", variables=["t2m"], init_times=[datetime(2025, 1, 1)])
    index = GriddedCatalog(tmp_path, profile).discover(request)
    bundle = GriddedReader("custom", profile).read(request, index)
    np.testing.assert_allclose(bundle.payload.t2m, 293.15)
    assert "lat" in bundle.payload.dims
    assert (
        get_description(forecast_fuxi.DESCRIPTION)["mapping"]["variables"]["t2m"]["source"] == "t2m"
    )


def test_text_format_declaration_controls_column_order(tmp_path):
    (tmp_path / "2025010106.000").write_text(
        "diamond 3 station\n2025 01 01 06\n54511 5.5 39.9 116.4 20\n", encoding="gbk"
    )
    layout = get_layout(observation_diamond.FLAT_DESCRIPTION)
    layout["format"]["columns"] = ["station", "precipitation", "lat", "lon", "altitude"]
    with observation_diamond.open(tmp_path, layout=layout).read(all_files=True) as raw:
        np.testing.assert_allclose(raw.precipitation, [[5.5]])
        np.testing.assert_allclose(raw.lon, [[116.4]])
    with observation_diamond.open(tmp_path, layout=layout).read(
        all_files=True, fixed_station_metadata=True
    ) as raw:
        np.testing.assert_allclose(raw.precipitation, [[5.5]])
        np.testing.assert_allclose(raw.lon, [116.4])


def test_fengqing_file_groups_and_evaluation_units(tmp_path):
    directory = tmp_path / "20250101"
    directory.mkdir()
    coords = {"member": [0, 1], "time": [datetime(2025, 1, 1)], "lat": [30.0], "lon": [120.0]}
    dimensions = ("member", "time", "lat", "lon")
    for group, variables in (
        ("PLEVELS", {"Z500": np.array([100.0, 101.0]) * 9.80665}),
        ("SURFACE", {"T2M": np.array([280.0, 281.0]), "UNLISTED": np.array([7.0, 8.0])}),
    ):
        xr.Dataset(
            {name: (dimensions, values.reshape(2, 1, 1, 1)) for name, values in variables.items()},
            coords=coords,
        ).to_netcdf(directory / f"Fengqing_1.0_GLB_{group}_OP25_6HOR_ENS_FCST_2025010100_006.nc")
    with forecast_fengqing.open(tmp_path).read(all_files=True) as raw:
        assert set(raw.data_vars) == {"Z500", "T2M", "UNLISTED"}
        assert raw.sizes["member"] == 2
        normalized = forecast_fengqing.normalize(raw, ["z500", "t2m"])
        np.testing.assert_allclose(normalized.z500.values.reshape(-1), [100.0, 101.0])
        assert normalized.z500.attrs["units"] == "m"
    request = DataRequest(
        source_id="fengqing", variables=["z500"], init_times=[datetime(2025, 1, 1)], lead_times=[6]
    )
    for factory, factor in (
        (_fengqing_source, 1.0),
        (_phys_source(FENGQING_PHYS_LAYOUT, "fengqing"), 9.80665),
    ):
        source = factory(root_dir=tmp_path)
        bundle = source.reader.read(request, source.catalog.discover(request))
        np.testing.assert_allclose(
            bundle.payload.z500.values.reshape(-1), np.array([100, 101]) * factor
        )


def test_cra_raw_archive_reads_variables_without_declarations(tmp_path):
    """NetCDF payloads exercise the GRIB archive layout without a GRIB decoder."""
    analysis_dir = tmp_path / "20250101"
    analysis_dir.mkdir()
    for variable, filename in (
        ("temperature_source", "ART_ATM_GLB_0P25_6HOR_ANAL_2025010100.grib2"),
        ("land_source", "CRA40LAND_SURFACE_2025010100_GLB_0P25_HOUR_V1_0_0.grib"),
    ):
        xr.Dataset(
            {variable: (("lat", "lon"), [[12.0]])}, coords={"lat": [30.0], "lon": [120.0]}
        ).to_netcdf(analysis_dir / filename)
    with analysis_cra.open(tmp_path, selection={"time": ["2025-01-01"]}).read(
        engine="netcdf4"
    ) as raw:
        assert set(raw.data_vars) == {"temperature_source", "land_source"}
        np.testing.assert_allclose(raw.land_source, 12.0)

    xr.Dataset({"unlisted_climate": ("lat", [3.0])}, coords={"lat": [30.0]}).to_netcdf(
        tmp_path / "ART_ATM_GLB_0P25_CLI_ANAL_010106.grib2"
    )
    with climatology_cra.open(tmp_path, selection={"cycle_time": ["010106"]}).read(
        engine="netcdf4"
    ) as raw:
        assert raw.cycle_time.values.tolist() == ["010106"]
        np.testing.assert_allclose(raw.unlisted_climate, 3.0)


def test_daily_climatology_unpacks_only_when_requested(tmp_path):
    path = tmp_path / "daily.nc"
    xr.Dataset(
        {"data": (("time", "level", "lat"), [[[280.0], [8.0]]])},
        coords={"time": [datetime(2025, 1, 1)], "level": ["T2M", "UNLISTED"], "lat": [30.0]},
    ).to_netcdf(path)
    with climatology_daily.open(path).read() as raw:
        assert raw.level.values.tolist() == ["T2M", "UNLISTED"]
        normalized = climatology_daily.normalize(raw, ["t2m"])
        np.testing.assert_allclose(normalized.t2m, 280.0)
        assert "data" in raw


@pytest.mark.parametrize("suffix", [".nc", ".npy", ".npz"])
def test_plain_file_roundtrip_and_overwrite_protection(tmp_path, suffix):
    data = xr.Dataset({"data": (("row", "column"), np.arange(6).reshape(2, 3))})
    path = tmp_path / ("array" + suffix)
    assert io.open(path).write(data) == [str(path)]
    options = {} if suffix == ".nc" else {"dims": ("row", "column")}
    with io.open(path).read(format=suffix[1:], **options) as restored:
        xr.testing.assert_equal(restored, data)
    with pytest.raises(FileExistsError):
        io.open(path).write(data)


def test_fixed_filename_layout_requires_no_backend_registration(tmp_path):
    description = {"name": "fixed", "layout": {"filename": "field.nc"}, "example": "field.nc"}
    data = xr.Dataset({"temperature": ("lat", [280.0, 281.0])}, coords={"lat": [30.0, 31.0]})
    source = io.open_dataset(tmp_path, description)
    assert source.write(data) == [str(tmp_path / "field.nc")]
    with source.read(all_files=True) as restored:
        xr.testing.assert_equal(restored, data)


@pytest.mark.parametrize("year_days", [365, 366])
def test_daily_climatology_evaluation_sampling(tmp_path, year_days):
    from pipelines.evaluation.components import _daily_climatology_source

    path = tmp_path / "daily_climatology.nc"
    xr.Dataset(
        {"data": (("time", "level", "lat", "lon"), np.arange(year_days).reshape(-1, 1, 1, 1))},
        coords={
            "time": (
                "time",
                np.arange(year_days),
                {"units": f"days since {2000 if year_days == 366 else 2001}-01-01"},
            ),
            "level": ["t2m"],
            "lat": [30.0],
            "lon": [120.0],
        },
    ).to_netcdf(path)
    times = [datetime(2024, 1, 1, 12), datetime(2024, 2, 29, 12), datetime(2024, 12, 31, 12)]
    request = DataRequest(source_id="daily_climatology", variables=["t2m"], init_times=times)
    source = _daily_climatology_source(
        root_dir=path, smooth_days=0, scales={"t2m": 2.0}, units={"t2m": "K"}
    )
    result = source.reader.read(request, source.catalog.discover(request))
    np.testing.assert_allclose(
        result.payload.t2m.values.ravel(),
        2 * np.array([0.5, 59.5 if year_days == 366 else 58.5, (year_days - 1) / 2]),
    )
    assert result.semantic.units["t2m"] == "K"

    source = _daily_climatology_source(root_dir=path, smooth_days=3)
    request.init_times = [datetime(2024, 1, 1)]
    result = source.reader.read(request, source.catalog.discover(request))
    np.testing.assert_allclose(result.payload.t2m, year_days / 3)
