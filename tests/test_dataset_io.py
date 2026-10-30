import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from datakit import io
from datakit.evaluation.contracts import DataRequest
from datakit.io import get_description, get_layout, open_dataset, source_config
from datakit.schema.mapping import DatasetMapping, apply_dataset_mapping, normalize_dataset
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


def test_all_source_declarations_load_from_an_external_yaml_directory(tmp_path):
    import shutil

    directory = source_config("forecast_fuxi").parent
    assert len(list(directory.iterdir())) == 9
    for original in directory.iterdir():
        assert original.suffix == ".yaml"
        copied = tmp_path / original.name
        shutil.copyfile(original, copied)
        declaration = get_description(copied)
        assert get_description(source_config(original.stem, directory=tmp_path)) == declaration
        source = open_dataset(tmp_path / "missing-data", copied)
        assert source.layout == get_layout(declaration)


def test_source_yaml_rejects_executable_or_unknown_format_selection(tmp_path):
    declaration = get_description(source_config("forecast_fuxi"))
    declaration["format"] = "examples.dataset_io.custom_backend"
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        open_dataset(tmp_path, declaration)
    with pytest.raises(ValueError, match="simple filename"):
        source_config("../forecast_fuxi")


@pytest.mark.parametrize("format_value", ["netcdf", {"name": "netcdf"}])
def test_declared_format_reads_and_writes_without_a_recognized_suffix(tmp_path, format_value):
    description = {
        "name": "netcdf_payload",
        "layout": {},
        "example": "data.payload",
        "format": format_value,
    }
    expected = xr.Dataset({"temperature": ("row", [280.0, 281.0])})
    source = open_dataset(tmp_path / "data.payload", description)
    source.write(expected)
    with source.read() as result:
        xr.testing.assert_identical(result, expected)
    assert description["format"] == format_value


def test_evaluation_sources_do_not_import_examples():
    code = """
import importlib.abc
import sys
class RejectExamples(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'examples' or fullname.startswith('examples.'):
            raise AssertionError('Evaluation imported example code: ' + fullname)
        return None
sys.meta_path.insert(0, RejectExamples())
import pipelines.evaluation.sources.dataset_profiles
import pipelines.evaluation.sources.babj
import pipelines.evaluation.sources.reference_probability
import pipelines.evaluation.sources.daily_climatology
assert not any(n == 'examples' or n.startswith('examples.') for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_io_example_cli_reads_and_exports_fuxi(tmp_path, normalize):
    source = tmp_path / "input" / "20250101"
    source.mkdir(parents=True)
    xr.Dataset(
        {
            "Z500": (("latitude", "longitude"), [[9.80665 * 3]], {"units": "m^2/s^2"}),
            "unlisted": (("latitude", "longitude"), [[42.0]]),
        },
        coords={"latitude": [30.0], "longitude": [120.0]},
    ).to_netcdf(source / "001.nc")
    output = tmp_path / "result.nc"
    command = [
        sys.executable,
        "-m",
        "examples.dataset_io.io_example",
        "--config",
        str(source_config("forecast_fuxi")),
        "--path",
        str(source.parent),
        "--init-time",
        "2025-01-01",
        "--lead-time",
        "6",
        "--output",
        str(output),
    ]
    if normalize:
        command += ["--variables", "z500"]
    subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    with xr.open_dataset(output) as result:
        if normalize:
            assert set(result.data_vars) == {"z500"}
            assert result.z500.attrs["units"] == "m"
            np.testing.assert_allclose(result.z500, 3.0)
        else:
            assert set(result.data_vars) == {"Z500", "unlisted"}
            assert result.Z500.attrs["units"] == "m^2/s^2"
            np.testing.assert_allclose(result.Z500, 9.80665 * 3)
            np.testing.assert_allclose(result.unlisted, 42.0)


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

    with open_dataset(
        tmp_path,
        source_config("forecast_fuxi_ensemble"),
        selection={"init_time": ["2025-01-01"], "lead_time": [12], "member": [1]},
    ).read() as raw:
        assert "unlisted" in raw
        np.testing.assert_allclose(raw.unlisted, 42.0)
        np.testing.assert_allclose(raw.Z500, 3.0 * 9.80665)

    profile = FUXI_ENS_LAYOUT.with_step_hours(3)
    source = open_dataset(
        tmp_path,
        source_config("forecast_fuxi"),
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
        normalized = normalize_dataset(
            raw, ["z500"], source_config("forecast_fuxi"), profile=profile
        )
        assert "lat" in normalized.dims
        assert normalized.z500.attrs["units"] == "m"
        np.testing.assert_allclose(normalized.z500, 3.0)
        physical = normalize_dataset(
            raw, ["z500"], source_config("forecast_fuxi"), profile=FUXI_ENS_PHYS_LAYOUT
        )
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
    first = get_layout(get_description(source_config("forecast_fuxi_ensemble")))
    first["layout"]["directory"] = "{init_time:%Y%m%d}/member_{member:03d}"
    assert (
        "member_{member:d}"
        in get_layout(get_description(source_config("forecast_fuxi_ensemble")))["layout"][
            "directory"
        ]
    )
    source = open_dataset(tmp_path / "missing", source_config("forecast_fuxi"))
    with pytest.raises(FileNotFoundError):
        source.discover(all_files=True)
    with pytest.raises(ValueError, match="config/data_sources"):
        io.open(tmp_path, "fuxi")


def test_diamond_raw_columns_normalization_and_writeback(tmp_path):
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    (source_dir / "2025010106.000").write_text(
        "diamond 3 station\n2025 01 01 06\n54511 116.4 39.9 20 5.5\n",
        encoding="gbk",
    )
    with open_dataset(source_dir, source_config("observation_diamond_flat")).read(
        all_files=True
    ) as raw:
        assert "precipitation" in raw and "tp" not in raw
        normalized = normalize_dataset(raw, ["tp"], source_config("observation_diamond"))
        np.testing.assert_allclose(normalized.tp, [[5.5]])
        assert normalized.tp.attrs["units"] == "mm"
        assert "units" not in raw.precipitation.attrs
        assert "precipitation" in raw
        outputs = open_dataset(
            tmp_path / "output", source_config("observation_diamond_flat")
        ).write(raw)
    with open_dataset(tmp_path / "output", source_config("observation_diamond_flat")).read(
        all_files=True
    ) as copied:
        np.testing.assert_allclose(copied.precipitation, [[5.5]])
    assert len(outputs) == 1


def test_probability_contract_used_by_io_and_evaluation(tmp_path):
    (tmp_path / "010106.000").write_text("54511 0.9 0.6 0.3 0.1\n")
    with open_dataset(tmp_path, source_config("climatology_station_probability")).read(
        all_files=True
    ) as raw:
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
    with open_dataset(source, source_config("observation_babj")).read(all_files=True) as raw:
        assert raw.lead_time.values.tolist() == [0, 6]
        open_dataset(tmp_path / "output", source_config("observation_babj")).write(raw)
    with open_dataset(tmp_path / "output", source_config("observation_babj")).read(
        all_files=True
    ) as copied:
        xr.testing.assert_equal(raw, copied)


def test_description_controls_mapping_and_evaluation_contract(tmp_path):
    description = get_description(source_config("forecast_fuxi"))
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
    with open_dataset(tmp_path, source_config("forecast_fuxi"), profile=profile).read(
        all_files=True
    ) as raw:
        assert raw.temperature.attrs["units"] == "degC"
        normalized = normalize_dataset(
            raw, ["t2m"], source_config("forecast_fuxi"), profile=profile
        )
        assert "lat" in normalized.dims
        assert normalized.t2m.attrs["units"] == "K"
        np.testing.assert_allclose(normalized.t2m, 293.15)
    request = DataRequest(source_id="custom", variables=["t2m"], init_times=[datetime(2025, 1, 1)])
    index = GriddedCatalog(tmp_path, profile).discover(request)
    bundle = GriddedReader("custom", profile).read(request, index)
    np.testing.assert_allclose(bundle.payload.t2m, 293.15)
    assert "lat" in bundle.payload.dims
    assert (
        get_description(source_config("forecast_fuxi"))["mapping"]["variables"]["t2m"]["source"]
        == "t2m"
    )


def test_text_format_declaration_controls_column_order(tmp_path):
    (tmp_path / "2025010106.000").write_text(
        "diamond 3 station\n2025 01 01 06\n54511 5.5 39.9 116.4 20\n", encoding="gbk"
    )
    layout = get_layout(get_description(source_config("observation_diamond_flat")))
    layout["format"]["columns"] = ["station", "precipitation", "lat", "lon", "altitude"]
    with open_dataset(tmp_path, source_config("observation_diamond"), layout=layout).read(
        all_files=True
    ) as raw:
        np.testing.assert_allclose(raw.precipitation, [[5.5]])
        np.testing.assert_allclose(raw.lon, [[116.4]])
    with open_dataset(tmp_path, source_config("observation_diamond"), layout=layout).read(
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
    with open_dataset(tmp_path, source_config("forecast_fengqing")).read(all_files=True) as raw:
        assert set(raw.data_vars) == {"Z500", "T2M", "UNLISTED"}
        assert raw.sizes["member"] == 2
        normalized = normalize_dataset(raw, ["z500", "t2m"], source_config("forecast_fengqing"))
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
    with open_dataset(
        tmp_path, source_config("analysis_cra"), selection={"time": ["2025-01-01"]}
    ).read(format="netcdf", engine="netcdf4") as raw:
        assert set(raw.data_vars) == {"temperature_source", "land_source"}
        np.testing.assert_allclose(raw.land_source, 12.0)

    xr.Dataset({"unlisted_climate": ("lat", [3.0])}, coords={"lat": [30.0]}).to_netcdf(
        tmp_path / "ART_ATM_GLB_0P25_CLI_ANAL_010106.grib2"
    )
    with open_dataset(
        tmp_path, source_config("climatology_cra"), selection={"cycle_time": ["010106"]}
    ).read(format="netcdf", engine="netcdf4") as raw:
        assert raw.cycle_time.values.tolist() == ["010106"]
        np.testing.assert_allclose(raw.unlisted_climate, 3.0)


def test_daily_climatology_unpacks_only_when_requested(tmp_path):
    path = tmp_path / "daily.nc"
    xr.Dataset(
        {"data": (("time", "level", "lat"), [[[280.0], [8.0]]])},
        coords={"time": [datetime(2025, 1, 1)], "level": ["T2M", "UNLISTED"], "lat": [30.0]},
    ).to_netcdf(path)
    with io.open(path).read() as raw:
        assert raw.level.values.tolist() == ["T2M", "UNLISTED"]
        normalized = apply_dataset_mapping(
            raw, DatasetMapping(packed_variable="data", packed_dimension="level"), ["t2m"]
        )
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
