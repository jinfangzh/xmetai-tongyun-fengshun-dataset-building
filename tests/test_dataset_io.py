import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from datakit import io
from datakit.evaluation.contracts import DataRequest
from datakit.io import get_description, get_layout, source_config
from datakit.schema.mapping import (
    DatasetMapping,
    apply_dataset_mapping,
    load_mapping,
    profile_mapping,
)
from pipelines.evaluation.components import (
    _fengqing_source,
    _fuxi_ensemble_source,
    _phys_source,
    _ref_probability_source,
)
from pipelines.evaluation.sources import GriddedCatalog, GriddedReader
from pipelines.evaluation.sources._profile import build_profile, profile_layout
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
        source = io.open(tmp_path / "missing-data", copied)
        assert source.layout == get_layout(declaration)


def test_source_yaml_rejects_executable_or_unknown_format_selection(tmp_path):
    declaration = get_description(source_config("forecast_fuxi"))
    declaration["format"] = "examples.dataset_io.custom_backend"
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        io.open(tmp_path, declaration)
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
    source = io.open(tmp_path / "data.payload", description)
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

    with io.open(
        tmp_path,
        source_config("forecast_fuxi_ensemble"),
        selection={"init_time": ["2025-01-01"], "lead_time": [12], "member": [1]},
    ).read() as raw:
        assert "unlisted" in raw
        np.testing.assert_allclose(raw.unlisted, 42.0)
        np.testing.assert_allclose(raw.Z500, 3.0 * 9.80665)

    profile = FUXI_ENS_LAYOUT.with_step_hours(3)
    source = io.open(
        tmp_path,
        profile_layout(profile),
        selection={"init_time": ["2025-01-01"], "lead_time": [6], "member": [1]},
    )
    records = source.discover()
    assert [record.path.relative_to(tmp_path).as_posix() for record in records] == [
        "20250101/member_1/002.nc"
    ]
    with source.read() as raw:
        assert "Z500" in raw and "latitude" in raw.dims
        assert raw.Z500.attrs["units"] == "m^2/s^2"
        normalized = apply_dataset_mapping(raw, profile_mapping(profile), ["z500"])
        assert "lat" in normalized.dims
        assert normalized.z500.attrs["units"] == "m"
        np.testing.assert_allclose(normalized.z500, 3.0)
        physical = apply_dataset_mapping(raw, profile_mapping(FUXI_ENS_PHYS_LAYOUT), ["z500"])
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
    source = io.open(tmp_path / "missing", source_config("forecast_fuxi"))
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
    with io.open(source_dir, source_config("observation_diamond_flat")).read(
        all_files=True
    ) as raw:
        assert "precipitation" in raw and "tp" not in raw
        normalized = apply_dataset_mapping(
            raw, load_mapping(source_config("observation_diamond")), ["tp"]
        )
        np.testing.assert_allclose(normalized.tp, [[5.5]])
        assert normalized.tp.attrs["units"] == "mm"
        assert "units" not in raw.precipitation.attrs
        assert "precipitation" in raw
        outputs = io.open(
            tmp_path / "output", source_config("observation_diamond_flat")
        ).write(raw)
    with io.open(tmp_path / "output", source_config("observation_diamond_flat")).read(
        all_files=True
    ) as copied:
        np.testing.assert_allclose(copied.precipitation, [[5.5]])
    assert len(outputs) == 1


def test_probability_contract_used_by_io_and_evaluation(tmp_path):
    (tmp_path / "010106.000").write_text("54511 0.9 0.6 0.3 0.1\n")
    with io.open(tmp_path, source_config("climatology_station_probability")).read(
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
    with io.open(source, source_config("observation_babj")).read(all_files=True) as raw:
        assert raw.lead_time.values.tolist() == [0, 6]
        io.open(tmp_path / "output", source_config("observation_babj")).write(raw)
    with io.open(tmp_path / "output", source_config("observation_babj")).read(
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
    with io.open(tmp_path, profile_layout(profile)).read(
        all_files=True
    ) as raw:
        assert raw.temperature.attrs["units"] == "degC"
        normalized = apply_dataset_mapping(raw, profile_mapping(profile), ["t2m"])
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
    with io.open(tmp_path, layout).read(
        all_files=True
    ) as raw:
        np.testing.assert_allclose(raw.precipitation, [[5.5]])
        np.testing.assert_allclose(raw.lon, [[116.4]])
    with io.open(tmp_path, layout).read(
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
    with io.open(tmp_path, source_config("forecast_fengqing")).read(all_files=True) as raw:
        assert set(raw.data_vars) == {"Z500", "T2M", "UNLISTED"}
        assert raw.sizes["member"] == 2
        normalized = apply_dataset_mapping(
            raw, load_mapping(source_config("forecast_fengqing")), ["z500", "t2m"]
        )
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
    with io.open(
        tmp_path, source_config("analysis_cra"), selection={"time": ["2025-01-01"]}
    ).read(format="netcdf", engine="netcdf4") as raw:
        assert set(raw.data_vars) == {"temperature_source", "land_source"}
        np.testing.assert_allclose(raw.land_source, 12.0)

    xr.Dataset({"unlisted_climate": ("lat", [3.0])}, coords={"lat": [30.0]}).to_netcdf(
        tmp_path / "ART_ATM_GLB_0P25_CLI_ANAL_010106.grib2"
    )
    with io.open(
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


@pytest.mark.parametrize("suffix", [".nc", ".npy", ".npz", ".zarr"])
def test_plain_file_roundtrip_and_overwrite_protection(tmp_path, suffix):
    data = xr.Dataset({"data": (("row", "column"), np.arange(6).reshape(2, 3))})
    path = tmp_path / ("array" + suffix)
    assert io.open(path).write(data) == [str(path)]
    options = {"dims": ("row", "column")} if suffix in (".npy", ".npz") else {}
    with io.open(path).read(format=suffix[1:], **options) as restored:
        xr.testing.assert_equal(restored, data)
    with pytest.raises(FileExistsError):
        io.open(path).write(data)
    replacement = data + 10
    io.open(path).write(replacement, mode="w")
    with io.open(path).read(**options) as restored:
        xr.testing.assert_equal(restored, replacement)


@pytest.mark.parametrize("suffix", [".nc", ".npy", ".npz", ".zarr"])
def test_failed_write_preserves_old_output_and_removes_staging(tmp_path, monkeypatch, suffix):
    from datakit.io.backends import files

    original = xr.Dataset({"data": ("row", [1.0, 2.0])})
    target = tmp_path / ("output" + suffix)
    io.open(target).write(original)
    writer = files._write_file

    def fail_after_write(*args, **kwargs):
        writer(*args, **kwargs)
        raise RuntimeError("write interrupted")

    monkeypatch.setattr(files, "_write_file", fail_after_write)
    with pytest.raises(RuntimeError, match="write interrupted"):
        io.open(target).write(original + 10, mode="w")
    options = {"dims": ("row",)} if suffix in (".npy", ".npz") else {}
    with io.open(target).read(**options) as restored:
        xr.testing.assert_equal(restored, original)
    fresh = tmp_path / ("fresh" + suffix)
    with pytest.raises(RuntimeError, match="write interrupted"):
        io.open(fresh).write(original)
    assert not fresh.exists()
    assert list(tmp_path.iterdir()) == [target]


def test_directory_publication_failure_restores_backup(tmp_path, monkeypatch):
    path = tmp_path / "data.zarr"
    original = xr.Dataset({"data": ("row", [1, 2])})
    io.open(path).write(original)
    replace = Path.replace

    def fail_publish(source, target):
        if source.name == path.name and source.parent != tmp_path:
            raise OSError("cannot publish directory")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_publish)
    with pytest.raises(OSError, match="cannot publish directory"):
        io.open(path).write(original + 10, mode="w")
    with io.open(path).read() as restored:
        xr.testing.assert_equal(restored, original)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_live_source_and_aliases_are_protected_until_close(tmp_path, suffix):
    path = tmp_path / ("input" + suffix)
    original = xr.Dataset({"data": ("row", [1, 2])})
    io.open(path).write(original)
    alias = tmp_path / ("alias" + suffix)
    alias.symlink_to(path, target_is_directory=path.is_dir())
    destinations = [path, alias]
    if path.is_file():
        hardlink = tmp_path / ("hardlink" + suffix)
        hardlink.hardlink_to(path)
        destinations.append(hardlink)
    with io.open(path).read(selection={"row": [0]}) as source:
        for destination in destinations:
            with pytest.raises(ValueError, match="active IO source"):
                io.open(destination).write(source, mode="w")
        overlapping = path / "nested.nc" if path.is_dir() else path.parent
        with pytest.raises(ValueError, match="active IO source"):
            io.open(overlapping).write(original, mode="w", format="netcdf")
        assert source.data.values.tolist() == [1]
    loaded = io.open(path).read(load=True)
    io.open(path).write(loaded + 10, mode="w")
    with io.open(path).read() as restored:
        xr.testing.assert_equal(restored, original + 10)


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_write_rejects_in_place_modes(tmp_path, mode):
    with pytest.raises(ValueError, match="mode='w'"):
        io.open(tmp_path / "field.nc").write(xr.Dataset(), mode=mode)


def test_layout_write_preflights_all_targets_and_allows_explicit_replacement(tmp_path):
    root = tmp_path / "source" / "20250101"
    root.mkdir(parents=True)
    for index in (1, 2):
        xr.Dataset({"temperature": ("lat", [float(index)])}).to_netcdf(root / f"{index:03d}.nc")
    description = source_config("forecast_fuxi")
    output = io.open(tmp_path / "output", description)
    with io.open(root.parent, description).read(all_files=True) as data:
        paths = [Path(path) for path in output.write(data)]
        paths[0].unlink()
        with pytest.raises(FileExistsError):
            output.write(data)
        assert not paths[0].exists()
        output.write(data, mode="w")
        with pytest.raises(ValueError, match="active IO source"):
            io.open(root.parent, description).write(data, mode="w")
    with output.read(all_files=True) as restored:
        assert restored.temperature.values.ravel().tolist() == [1, 2]


@pytest.mark.parametrize(
    "declaration, filename, content",
    [
        ("observation_diamond_flat", "2025010106.000",
         "diamond 3 station\n2025 01 01 06\n54511 116.4 39.9 20 5.5\n"),
        ("climatology_station_probability", "010106.000", "54511 0.9 0.6 0.3 0.1\n"),
        ("observation_babj", "babj2501.dat",
         "diamond 7\nTEST 2501\nheader\n2025 1 1 0 0 120 20 990 30\n"),
    ],
)
def test_text_write_replacement_and_failure_preserve_output(
    tmp_path, monkeypatch, declaration, filename, content
):
    root = tmp_path / "input"
    root.mkdir()
    (root / filename).write_text(content, encoding="gbk")
    description = source_config(declaration)
    output = io.open(tmp_path / "output", description)
    with io.open(root, description).read(all_files=True) as data:
        paths = [Path(path) for path in output.write(data)]
        original = {path: path.read_bytes() for path in paths}
        with pytest.raises(FileExistsError):
            output.write(data)
        output.write(data, mode="w")
        assert {path: path.read_bytes() for path in paths} == original
        with pytest.raises(ValueError, match="active IO source"):
            io.open(root, description).write(data, mode="w")

        def fail_write(*args, **kwargs):
            raise OSError("text write interrupted")

        monkeypatch.setattr(np, "savetxt", fail_write)
        with pytest.raises(OSError, match="text write interrupted"):
            output.write(data, mode="w")
        assert {path: path.read_bytes() for path in paths} == original
        assert sorted(path.name for path in (tmp_path / "output").iterdir()) == [filename]


def test_reader_protection_releases_on_failure_and_iterator_close(tmp_path):
    path = tmp_path / "field.nc"
    data = xr.Dataset({"data": ("row", [1, 2])}, coords={"row": [0, 1]})
    io.open(path).write(data)
    with pytest.raises(KeyError):
        io.open(path).read(selection={"row": [9]})
    io.open(path).write(data, mode="w")
    iterator = io.open(path).iter_datasets()
    next(iterator)
    with pytest.raises(ValueError, match="active IO source"):
        io.open(path).write(data, mode="w")
    iterator.close()
    io.open(path).write(data, mode="w")


def test_fixed_filename_layout_requires_no_backend_registration(tmp_path):
    description = {"name": "fixed", "layout": {"filename": "field.nc"}, "example": "field.nc"}
    data = xr.Dataset({"temperature": ("lat", [280.0, 281.0])}, coords={"lat": [30.0, 31.0]})
    source = io.open(tmp_path, description)
    assert source.write(data) == [str(tmp_path / "field.nc")]
    with source.read(all_files=True) as restored:
        xr.testing.assert_equal(restored, data)


def test_records_freeze_discovery_and_apply_selection_without_scanning(tmp_path, monkeypatch):
    root = tmp_path / "20250101"
    root.mkdir()
    source = io.open(tmp_path, source_config("forecast_fuxi"))
    for index in (1, 2):
        xr.Dataset({"temperature": ("lat", [float(index)])}).to_netcdf(root / f"{index:03d}.nc")
    records = source.discover(all_files=True)
    assert all(isinstance(record, io.FileRecord) for record in records)
    xr.Dataset({"temperature": ("lat", [99.0])}).to_netcdf(root / "003.nc")

    def no_scan(*args, **kwargs):
        pytest.fail("explicit records must not rescan")

    monkeypatch.setattr("datakit.io.dataset.discover", no_scan)
    with source.read(records=records, selection={"lead_time": [12]}) as data:
        assert data.temperature.values.ravel().tolist() == [2]
    parts = [part.load() for part in source.iter_datasets(records=records)]
    assert [part.temperature.values.item() for part in parts] == [1, 2]
    with pytest.raises(KeyError):
        source.read(records=records, selection={"lead_time": [18]})
    with pytest.raises(KeyError):
        list(source.iter_datasets(records=records, selection={"lead_time": [6, 18]}))
    records[0].path.unlink()
    with pytest.raises(FileNotFoundError):
        source.read(records=records)


def test_record_validation_and_partition_combination_do_not_own_or_load_inputs(tmp_path):
    import dask.array as da
    from dask import delayed

    @delayed
    def unexpected_load():
        raise AssertionError("combination must not load array values")

    closed = []
    records = [io.FileRecord(tmp_path / f"{i}.nc", {"time": datetime(2025, 1, i + 1)}) for i in (0, 1)]
    datasets = [
        xr.Dataset({"value": ("lat", da.from_delayed(unexpected_load(), shape=(1,), dtype="f8"))})
        for _ in records
    ]
    for dataset in datasets:
        dataset.set_close(lambda: closed.append(True))
    combined = io.combine_partitions(records, datasets)
    assert combined.sizes["time"] == 2
    assert combined.value.chunks is not None
    combined.close()
    assert not closed
    for dataset in datasets:
        dataset.close()
    assert len(closed) == 2
    with pytest.raises(ValueError, match="one Dataset"):
        io.combine_partitions(records, datasets[:1])
    with pytest.raises(ValueError):
        io.combine_partitions([], [])
    with pytest.raises(ValueError, match="Overlapping"):
        io.combine_partitions([records[0], records[0]], datasets)
    records[0].path.touch()
    with pytest.raises(ValueError, match="Duplicate file"):
        io.open(tmp_path).read(records=[records[0], records[0]])


@pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf", "scipy"])
def test_threaded_netcdf_reads_and_nested_lazy_handles(tmp_path, engine):
    from concurrent.futures import ThreadPoolExecutor
    import pickle

    pytest.importorskip("netCDF4" if engine == "netcdf4" else engine)

    paths = [tmp_path / f"{i}.nc" for i in range(4)]
    for i, path in enumerate(paths):
        xr.Dataset({"value": ("x", np.full(8, i))}).to_netcdf(path, engine=engine)

    def read_value(i):
        with io.open(paths[i % 4]).read(engine=engine) as dataset:
            return dataset.value.values.tolist()

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(read_value, range(40)))
    assert values == [[i % 4] * 8 for i in range(40)]
    with io.open(paths[0]).read(engine=engine) as first, io.open(paths[1]).read(engine=engine) as second:
        assert first.value.values[0] == 0
        assert second.value.values[0] == 1
        restored = pickle.loads(pickle.dumps(second.value.variable))
        np.testing.assert_array_equal(restored.values, np.ones(8))


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
