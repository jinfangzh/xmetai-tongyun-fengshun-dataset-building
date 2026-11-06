"""CDS-shaped monthly NetCDF/ZIP fixtures, including accumulation boundaries."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import xarray as xr

from datakit import io
from datakit.schema import load_mapping, load_schema
from datakit.schema.semantics import TemporalKind
from examples.dataset_building import example_build_era5 as era5
from examples.dataset_building import example_build_oras5 as oras5
from pipelines.dataset import build as pipeline
from pipelines.dataset import build_zarr
from pipelines.dataset.preparation import (
    assemble_standard_dataset,
    prepare_standard_arrays,
    prepare_statistics,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "config/data_sources/era5_cds_monthly_raw.yaml"
BASE_SPEC = ROOT / "config/dataset_specs/era5_c78_p25_h6.yaml"


@contextmanager
def preprocess_era5(source, description, *, spec):
    """Keep raw handles alive while testing the process-side preparation."""
    with era5.read_source(source, description, spec=spec) as raw:
        yield era5._prepare_raw(raw, description, spec, accumulation_hours=6)


def _write_partition(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        payload = dataset.to_netcdf(engine="scipy")
        with ZipFile(path, "w") as archive:
            archive.writestr("data.nc", payload)
    else:
        dataset.to_netcdf(path, engine="scipy")


def _write_raw(root, spec, description, *, missing_hour=None, truncate_boundary=False,
               missing_level=False, missing_cell=False):
    mapping = load_mapping(description)
    expected = spec.expected_times()
    hourly = np.arange(expected[0] - np.timedelta64(5, "h"),
                       expected[-1] + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    if truncate_boundary:
        hourly = hourly[hourly >= np.datetime64("2025-01-01T00")]
    pressure = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50])
    for variable, declaration in mapping.variables.items():
        accumulation = declaration.temporal_kind == TemporalKind.INTERVAL_ACCUMULATION
        times = hourly if accumulation else expected
        if missing_hour is not None and variable == "tp":
            times = times[times != np.datetime64(missing_hour)]
        is_pressure = variable in {"z", "t", "u", "v", "q"}
        if is_pressure:
            dims = ("valid_time", "pressure_level", "latitude", "longitude")
            data = np.broadcast_to(pressure[None, :, None, None], (len(times), 13, 2, 2)).astype("f4")
            if variable == "q":
                data = data * 1e-6
        else:
            dims = ("valid_time", "latitude", "longitude")
            if accumulation:
                values = times.astype("datetime64[h]").astype("int64") % 24 + 1
                factor = 1e-4 if variable == "tp" else (-100 if variable == "ttr" else 100)
                data = np.broadcast_to((values * factor)[:, None, None], (len(times), 2, 2)).astype("f4")
                if missing_cell and variable == "tp":
                    data[1, 0, 0] = np.nan
            else:
                data = np.ones((len(times), 2, 2), dtype="f4")
        coords = {"valid_time": times, "latitude": [89.75, 90.0], "longitude": [0.25, 0.0]}
        if is_pressure:
            coords["pressure_level"] = ("pressure_level", pressure, {"units": "hPa"})
        dataset = xr.DataArray(data, dims=dims, coords=coords, name=declaration.source,
                               attrs={"units": declaration.source_unit,
                                      "GRIB_stepType": "accum" if accumulation else "instant"}).to_dataset()
        # CDS auxiliary coordinates must not become channels or collide on merge.
        dataset = dataset.assign_coords(number=0, expver=("valid_time", np.ones(len(times), dtype="i4")))
        if missing_level and variable == "z":
            dataset = dataset.isel(pressure_level=slice(1, None))
        months = np.unique(times.astype("datetime64[M]"))
        for month in months:
            moment = datetime.fromisoformat(str(month) + "-01")
            selected = dataset.isel(valid_time=times.astype("datetime64[M]") == month)
            path = root / description["layout"]["files"][variable].format(month=moment)
            _write_partition(path, selected)


@pytest.fixture
def era5_raw(tmp_path):
    # Unit tests derive their tiny coverage from the stable product definition,
    # not a user-editable example run contract.
    spec, _ = load_schema(BASE_SPEC)
    variant = spec.variant
    grid = replace(variant.grid, latitude_count=2, longitude_count=2)
    product = replace(spec.product, variants=(replace(variant, grid=grid, auxiliaries=()),))
    spec = replace(spec, product=product, first_time="2025-01-01T06",
                   last_time="2025-01-01T12", schema_status="approved",
                   directory_name="result.zarr", identity_mapping_evidence="Synthetic test")
    description = io.get_description(SOURCE)
    return tmp_path, spec, description


def test_era5_monthly_nc_and_zip_are_packed_and_accumulated(era5_raw):
    root, spec, description = era5_raw
    _write_raw(root, spec, description)
    (root / "accum_h1_v1/total_precipitation/2025/incomplete.zip.part").touch()
    assert pipeline.validate_source(root, description)["files_checked"] == 18
    with era5.read_source(root, description, spec=spec) as raw:
        assert "data" not in raw
        assert raw.sizes["time_tp"] == 12
        assert raw.sizes["time_z"] == 2
        # Reading preserves the one-hour amounts; aggregation belongs to process.
        np.testing.assert_allclose(raw.tp.isel(time_tp=0).compute(), 0.0002)
    with preprocess_era5(root, description, spec=spec) as packed:
        assert packed.data.dims == ("time", "channel", "lat", "lon")
        assert packed.data.shape == (2, 78, 2, 2)
        assert tuple(packed.channel.values) == spec.product.channel_names
        assert hasattr(packed.data.data, "dask")
        np.testing.assert_allclose(packed.data.sel(channel="z50").compute(), 50)
        np.testing.assert_allclose(packed.data.sel(channel="tp").isel(lat=0, lon=0).compute(), [0.0027, 0.0063])
        np.testing.assert_allclose(packed.data.sel(channel="ssr").isel(lat=0, lon=0).compute(), [2700, 6300])
        np.testing.assert_allclose(packed.data.sel(channel="ttr").isel(lat=0, lon=0).compute(), [-2700, -6300])
    assert sorted(root.rglob("*.nc"))  # Original input files remain available.


@pytest.mark.parametrize("problem", ["missing_hour", "month_boundary", "missing_variable"])
def test_era5_incomplete_downloads_fail_before_writing(era5_raw, problem):
    root, spec, description = era5_raw
    if problem == "month_boundary":
        spec = replace(spec, first_time="2025-01-01T00", last_time="2025-01-01T00")
    _write_raw(root, spec, description,
               missing_hour="2025-01-01T03" if problem == "missing_hour" else None,
               truncate_boundary=problem == "month_boundary")
    if problem == "missing_variable":
        next(root.rglob("temperature_202501.nc")).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="missing|incomplete"):
        with preprocess_era5(root, description, spec=spec):
            pass


def test_era5_accumulation_spans_month_partitions(era5_raw):
    root, spec, description = era5_raw
    spec = replace(spec, first_time="2025-02-01T00", last_time="2025-02-01T00")
    _write_raw(root, spec, description)
    with preprocess_era5(root, description, spec=spec) as packed:
        # Jan31 19..23 -> 20+21+22+23+24, Feb1 00 -> 1.
        np.testing.assert_allclose(packed.data.sel(channel="ssr").compute(), 11100)


def test_era5_missing_pressure_level_is_rejected(era5_raw):
    root, spec, description = era5_raw
    _write_raw(root, spec, description, missing_level=True)
    with pytest.raises(ValueError, match="level|align"):
        with preprocess_era5(root, description, spec=spec):
            pass


def test_era5_missing_cells_stay_nan_during_accumulation(era5_raw):
    root, spec, description = era5_raw
    _write_raw(root, spec, description, missing_cell=True)
    with preprocess_era5(root, description, spec=spec) as packed:
        # Coordinates are sorted into north-to-south, west-to-east target order.
        value = packed.data.sel(channel="tp", lat=89.75, lon=0.25).compute().values
        assert np.isnan(value[0])
        np.testing.assert_allclose(value[1], 0.0063)


def test_era5_raw_to_normalized_product_dry_run(era5_raw):
    root, spec, description = era5_raw
    _write_raw(root, spec, description)
    channels = list(spec.product.channel_names)
    mean = xr.DataArray(np.zeros(78, dtype="f4"), dims="channel", coords={"channel": channels})
    lat, lon = spec.variant.grid.coordinates()
    mask = xr.DataArray(np.array([np.ones((2, 2)), np.zeros((2, 2))], dtype="u1"),
                        dims=("mask_channel", "lat", "lon"),
                        coords={"mask_channel": ["land_mask", "sea_mask"], "lat": lat, "lon": lon})
    paths = {}
    for name, array in {"mean": mean, "std": xr.ones_like(mean), "mask": mask}.items():
        paths[name] = root.parent / f"{name}.nc"
        array.rename(name).to_dataset().to_netcdf(paths[name], engine="scipy")
    output = root.parent / "result.zarr"
    report = build_zarr(
        description, spec, root, output,
        read=era5.read_source,
        process=partial(era5.process, mean_path=paths["mean"], std_path=paths["std"], mask_path=paths["mask"]),
        dry_run=True,
        build_overrides={"first_time": "2025-01-01T06", "last_time": "2025-01-01T12",
                         "dataset_id": "result.zarr"},
    )
    assert report.coverage["conformance_passed"], report.to_dict()
    assert not report.coverage["written"] and not output.exists()
    assert report.coverage["target_contract"]["effective"]["time_count"] == 2
    assert report.coverage["target_contract"]["effective"]["dataset_id"] == "result.zarr"
    assert report.coverage["target_contract"]["template_path"] == str(BASE_SPEC.resolve())
    with preprocess_era5(root, description, spec=spec) as packed:
        result = era5.process(packed, description, spec, mean_path=paths["mean"],
                              std_path=paths["std"], mask_path=paths["mask"])
        np.testing.assert_allclose(result.data.sel(channel="ssr").isel(lat=0, lon=0).compute(),
                                   np.array([2700, 6300]) / 21600, rtol=1e-3)
        np.testing.assert_allclose(result.data.sel(channel="tp").isel(lat=0, lon=0).compute(),
                                   np.log1p([2.7, 6.3]), rtol=1e-3)


def test_oras5_daily_raw_depth_grid_and_build_dry_run(tmp_path):
    spec, _ = load_schema(ROOT / "config/dataset_specs/oras5_c88_p25_h24.yaml")
    variant = spec.variant
    grid = replace(variant.grid, latitude_count=2, longitude_count=2, west=180.0)
    product = replace(spec.product, variants=(replace(variant, grid=grid, auxiliaries=()),))
    spec = replace(spec, product=product, first_time="2025-01-01T00", last_time="2025-01-02T00",
                   directory_name="result.zarr", identity_mapping_evidence="Synthetic test")
    description = io.get_description(ROOT / "config/data_sources/oras5_cmems_daily_raw.yaml")
    mapping = load_mapping(description)
    depths = sorted({c.level for c in spec.product.channels if c.variable == "so"})
    source = tmp_path / "downloads"
    (source / "2025").mkdir(parents=True)
    for day, time in enumerate(spec.expected_times(), 1):
        variables = {}
        for name, declaration in mapping.variables.items():
            if name in {"so", "to", "uo", "vo"}:
                array = xr.DataArray(np.broadcast_to(np.array(depths, dtype="f4")[None, :, None, None],
                                                     (1, len(depths), 1, 2)).copy(),
                                     dims=("time", "depth", "latitude", "longitude"),
                                     coords={"time": [time], "depth": ("depth", depths, {"units": "m"}),
                                             "latitude": [90.0], "longitude": [-179.75, -180.0]})
            else:
                array = xr.DataArray(np.full((1, 1, 2), 2, dtype="f4"),
                                     dims=("time", "latitude", "longitude"),
                                     coords={"time": [time], "latitude": [90.0],
                                             "longitude": [-179.75, -180.0]})
            array.attrs["units"] = declaration.source_unit
            variables[declaration.source] = array
        xr.Dataset(variables).to_netcdf(source / "2025" / f"202501{day:02d}_oras5.nc", engine="scipy")
    channels = list(spec.product.channel_names)
    mean = xr.DataArray(np.zeros(88, dtype="f4"), dims="channel", coords={"channel": channels})
    lat, lon = grid.coordinates()
    mask = xr.DataArray(np.array([np.zeros((2, 2)), np.ones((2, 2))], dtype="u1"),
                        dims=("mask_channel", "lat", "lon"),
                        coords={"mask_channel": ["land_mask", "sea_mask"], "lat": lat, "lon": lon})
    paths = {}
    for name, array in {"mean": mean, "std": xr.ones_like(mean), "mask": mask}.items():
        paths[name] = tmp_path / f"{name}.nc"
        array.rename(name).to_dataset().to_netcdf(paths[name], engine="scipy")
    processor = partial(oras5.process, mean_path=paths["mean"], std_path=paths["std"], mask_path=paths["mask"])
    with pipeline.read_source(source, description) as raw:
        result = processor(raw, description, spec)
        assert result.data.shape == (2, 88, 2, 2)
        np.testing.assert_allclose(result.data.sel(channel="so19979", lat=90).compute(),
                                   np.float16(199.79))
        assert result.data.sel(lat=89.75).isnull().all().compute()
        np.testing.assert_array_equal(result.lon, [180.0, 180.25])
    report = build_zarr(description, spec, source, tmp_path / "result.zarr", process=processor, dry_run=True)
    assert report.coverage["conformance_passed"], report.to_dict()
    assert not report.coverage["written"] and not (tmp_path / "result.zarr").exists()


def _prepared_arrays(spec):
    channels = list(spec.product.channel_names)
    lat, lon = spec.variant.grid.coordinates()
    data = xr.DataArray(np.arange(2 * len(channels) * 4, dtype="f4").reshape(2, len(channels), 2, 2),
                        dims=("time", "channel", "lat", "lon"),
                        coords={"time": spec.expected_times(), "channel": channels, "lat": lat, "lon": lon})
    mean = xr.DataArray(np.zeros(len(channels), dtype="f4"), dims="channel", coords={"channel": channels})
    mask = xr.DataArray(np.array([np.ones((2, 2)), np.zeros((2, 2))], dtype="u1"),
                        dims=("mask_channel", "lat", "lon"),
                        coords={"mask_channel": ["land_mask", "sea_mask"], "lat": lat, "lon": lon})
    return data, mean, xr.ones_like(mean), mask


@pytest.mark.parametrize("external_mean", [False, True])
def test_statistics_preserve_supplied_mean_and_compute_missing_values(era5_raw, external_mean):
    root, spec, _ = era5_raw
    data, mean, _, _ = _prepared_arrays(spec)
    path = None
    if external_mean:
        path = root / "mean.nc"
        supplied = xr.concat([mean + 5, xr.DataArray([99.0], dims="channel", coords={"channel": ["extra"]})],
                             dim="channel").isel(channel=slice(None, None, -1))
        supplied.rename("mean").to_dataset().to_netcdf(path, engine="scipy")
    actual_mean, actual_std = prepare_statistics(data, spec, mean_path=path)
    expected_mean = np.full(len(mean), 5) if external_mean else np.mean(data.values, axis=(0, 2, 3))
    np.testing.assert_allclose(actual_mean, expected_mean)
    np.testing.assert_allclose(actual_std, np.std(data.values, axis=(0, 2, 3)), rtol=1e-6)
    assert actual_mean.dtype == actual_std.dtype == np.dtype("float32")
    assert tuple(actual_mean.channel.values) == spec.product.channel_names


def test_computed_statistics_reject_incomplete_time_coverage(era5_raw):
    _, spec, _ = era5_raw
    data, _, _, _ = _prepared_arrays(spec)
    with pytest.raises(ValueError, match="time coverage"):
        prepare_statistics(data.isel(time=[0]), spec)


@pytest.mark.parametrize("problem", ["overflow", "float_mask", "grid"])
def test_standard_arrays_reject_invalid_mask_before_casting(era5_raw, problem):
    _, spec, _ = era5_raw
    data, mean, std, mask = _prepared_arrays(spec)
    if problem == "overflow":
        mask = mask.astype("uint16")
        mask.values[0, 0, 0] = 257
    elif problem == "float_mask":
        mask = mask.astype("float32")
    else:
        mask = mask.assign_coords(lat=mask.lat + 0.01)
    with pytest.raises(ValueError):
        prepare_standard_arrays(data, mean, std, mask, spec)


def test_standard_arrays_keep_values_lazy_and_align_storage_chunks(era5_raw):
    _, spec, _ = era5_raw
    data, mean, std, mask = _prepared_arrays(spec)
    lazy = data.chunk({"time": 2, "channel": 1, "lat": 1, "lon": 1})
    result = prepare_standard_arrays(lazy, mean, std, mask, spec)
    assert tuple(result.sizes) == ("time", "channel", "lat", "lon", "mask_channel")
    assert result.data.chunks == ((1, 1), (78,), (2,), (2,))
    assert result.data.dtype == np.dtype("float16")
    np.testing.assert_array_equal(result.data.compute(), data.astype("float16"))
    assert lazy.chunks == ((2,), (1,) * 78, (1, 1), (1, 1))
    assembled = assemble_standard_dataset(result, spec)
    assert tuple(assembled.sizes) == ("time", "channel", "lat", "lon", "mask_channel")
    assert assembled.data.dims == ("time", "channel", "lat", "lon")
    assert assembled['mask'].dims == ("mask_channel", "lat", "lon")
    assert assembled.data.data is result.data.data

