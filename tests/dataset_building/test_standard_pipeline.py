"""Contract-bound example processing without external archives or Zarr fixtures."""

from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from datakit.schema import load_schema
from examples.dataset_building import standard_pipeline as pipeline


@pytest.fixture(params=["era5_c78_p25_h6", "era5_c116_p25_h6", "oras5_c88_1p5_h24"])
def inputs(request):
    root = Path(__file__).resolve().parents[2]
    spec, _ = load_schema(root / "config/dataset_specs" / f"{request.param}.yaml")
    variant = spec.variant
    grid = replace(variant.grid, latitude_count=2, longitude_count=3, resolution=1.5)
    product = replace(spec.product, variants=(replace(variant, grid=grid),))
    spec = replace(
        spec, product=product, first_time="2024-01-01T00", last_time="2024-01-01T00",
        directory_name="result.zarr", identity_mapping_evidence="Explicit test destination",
    )
    lat, lon = grid.coordinates()
    channels = list(product.channel_names)
    raw = xr.Dataset({"data": xr.DataArray(
        np.ones(spec.expected_shape, dtype="f4"), dims=("time", "channel", "lat", "lon"),
        coords={"time": spec.expected_times(), "channel": channels, "lat": lat, "lon": lon},
    )})
    # Reversed statistic labels exercise alignment by name rather than position.
    mean = xr.DataArray(np.zeros(len(channels), dtype="f4"), dims="channel",
                        coords={"channel": channels[::-1]}, name="mean")
    std = xr.ones_like(mean).rename("std")
    mask = xr.DataArray(
        np.stack([np.ones(grid.shape, dtype="u1"), np.zeros(grid.shape, dtype="u1")]),
        dims=("mask_channel", "lat", "lon"),
        coords={"mask_channel": ["land_mask", "sea_mask"], "lat": lat, "lon": lon},
        name="mask",
    )
    return spec, raw, {"mean": mean, "std": std, "mask": mask}


def processor(spec, root=Path(".")):
    function = pipeline.process_era5 if spec.product.product_id.startswith("era5") else pipeline.process_oras5
    return partial(function, mean_path=root / "mean.nc", std_path=root / "std.nc", mask_path=root / "mask.nc")


def value_processor(spec):
    return pipeline._era5_values if spec.product.product_id.startswith("era5") else pipeline._oras5_values


def test_era5_and_oras5_share_contract_bound_pipeline(inputs, tmp_path, monkeypatch):
    spec, raw, auxiliaries = inputs
    closed = []
    raw.set_close(lambda: closed.append(True))
    before = raw.copy(deep=True)
    calls = []

    def reader(path, description, **options):
        calls.append((path, description, options))
        return raw

    monkeypatch.setattr(pipeline, "validate_source", lambda *args: None)
    monkeypatch.setattr(pipeline, "_load_array", lambda path, name: auxiliaries[name])
    monkeypatch.setattr(pipeline.io, "open_dataset", lambda path, description:
                        SimpleNamespace(read=lambda **options: reader(path, description, **options)))
    report = pipeline.build_dataset(
        {"name": "physical", "format": "netcdf", "layout": {}, "example": "data.nc"},
        spec, tmp_path / "input.nc", tmp_path / "result.zarr",
        process=processor(spec, tmp_path), dry_run=True,
    )
    assert report.coverage["conformance_passed"], report.to_dict()
    assert not report.coverage["written"]
    assert not (tmp_path / "result.zarr").exists()
    assert closed == [True]
    assert calls[0][0] == tmp_path / "input.nc"
    xr.testing.assert_identical(raw, before)
    mean = auxiliaries["mean"].sel(channel=list(spec.product.channel_names))
    std = auxiliaries["std"].sel(channel=mean.channel)
    result = value_processor(spec)(raw, spec, mean, std)
    if spec.product.product_id.startswith("era5"):
        np.testing.assert_allclose(result.sel(channel="q50"), 1000)
        np.testing.assert_allclose(result.sel(channel="tp"), np.log1p(1000))
        np.testing.assert_allclose(result.sel(channel="ssr"), 1 / 21600)
        if "ws10m" in result.channel:
            np.testing.assert_allclose(result.sel(channel="ws10m"), np.sqrt(2))
    else:
        np.testing.assert_allclose(result, 1)


def test_shifted_mask_is_rejected_instead_of_aligned_silently(inputs):
    spec, raw, aux = inputs
    channels = list(spec.product.channel_names)
    data = raw.data
    mask = aux["mask"].assign_coords(lat=aux["mask"].lat + 0.5)
    with pytest.raises(ValueError):
        pipeline.assemble_product(data, spec, aux["mean"].sel(channel=channels),
                                  aux["std"].sel(channel=channels), mask)


def test_missing_channel_is_not_renamed(inputs):
    spec, raw, aux = inputs
    channels = list(spec.product.channel_names)
    raw = raw.assign_coords(channel=["unrelated", *channels[1:]])
    with pytest.raises(KeyError):
        value_processor(spec)(raw, spec, aux["mean"], aux["std"])


def test_existing_destination_is_preserved(tmp_path):
    target = tmp_path / "result.zarr"
    target.mkdir()
    with pytest.raises(FileExistsError):
        pipeline.build_dataset({}, "unused.yaml", tmp_path / "input.nc", target,
                               process=pipeline.process_oras5)


def test_reader_is_closed_when_processing_fails(inputs, tmp_path, monkeypatch):
    spec, raw, aux = inputs
    closed = []
    raw.set_close(lambda: closed.append(True))
    monkeypatch.setattr(pipeline, "validate_source", lambda *args: None)
    monkeypatch.setattr(pipeline, "_load_array", lambda path, name: aux[name])
    monkeypatch.setattr(pipeline.io, "open_dataset", lambda *args:
                        SimpleNamespace(read=lambda **kwargs: raw))

    def fail(*args):
        raise ValueError("Explicit strategy failed")

    with pytest.raises(ValueError, match="strategy failed"):
        pipeline.build_dataset(
            {"name": "source", "layout": {}, "example": "data.nc"}, spec,
            tmp_path / "input.nc", tmp_path / "result.zarr",
            process=fail,
        )
    assert closed == [True]


def test_real_netcdf_reader_and_auxiliaries(inputs, tmp_path):
    spec, raw, aux = inputs
    source = tmp_path / "input.nc"
    raw.to_netcdf(source, engine="scipy")
    for name, array in aux.items():
        array.to_dataset().to_netcdf(tmp_path / f"{name}.nc", engine="scipy")
    report = pipeline.build_dataset(
        {"name": "packed_nc", "format": "netcdf", "layout": {}, "example": "input.nc"},
        spec, source, tmp_path / "result.zarr",
        process=processor(spec, tmp_path), dry_run=True,
    )
    assert report.coverage["conformance_passed"], report.to_dict()


def test_processor_rejects_a_different_target_recipe(inputs):
    spec, raw, aux = inputs
    first, *rest = spec.product.channels
    changed = replace(first, preprocessing=({"operation": "log1p"},))
    altered = replace(spec, product=replace(spec.product, channels=(changed, *rest)))
    with pytest.raises(ValueError, match="recipe differs|only unconverted zscore"):
        value_processor(spec)(raw, altered, aux["mean"], aux["std"])


def test_invalid_mask_cannot_wrap_into_valid_uint8(inputs):
    spec, raw, aux = inputs
    channels = list(spec.product.channel_names)
    mask = aux["mask"].astype("int16")
    mask.values[0, 0, 0] = 257
    with pytest.raises(ValueError, match="binary integer"):
        pipeline.assemble_product(raw.data, spec, aux["mean"].sel(channel=channels),
                                  aux["std"].sel(channel=channels), mask)


def test_pipeline_validates_written_output_after_closing_source(tmp_path, monkeypatch):
    events = []
    raw = xr.Dataset()
    raw.set_close(lambda: events.append("close_source"))
    spec = object()
    product = xr.Dataset({"prepared": ("sample", [1.0])})
    creation = SimpleNamespace(coverage={"written": True, "conformance_passed": True})
    # A successful write must not hide a failed final validation.
    validation = SimpleNamespace(coverage={}, quick_passed=False)

    def load_source(*args):
        events.append("load_source_contract")
        return {}

    def load_target(*args):
        events.append("load_target_contract")
        return spec

    def read(*args):
        events.append("read")
        return raw

    def process(dataset, target):
        assert dataset is raw and target is spec
        events.append("process")
        return product

    def write(dataset, destination, target, **options):
        assert dataset is product and target is spec
        events.append("write")
        return creation

    def validate(destination, target, *, mode):
        assert destination == tmp_path / "result.zarr" and target is spec
        assert mode == "quick"
        events.append("validate")
        return validation

    monkeypatch.setattr(pipeline, "validate_source", lambda *args: events.append("validate_source"))
    monkeypatch.setattr(pipeline, "load_source_contract", load_source)
    monkeypatch.setattr(pipeline, "load_target_contract", load_target)
    monkeypatch.setattr(pipeline, "read_source", read)
    monkeypatch.setattr(pipeline, "write_product", write)
    monkeypatch.setattr(pipeline, "validate_zarr", validate)
    result = pipeline.build_dataset({}, spec, tmp_path / "input.nc", tmp_path / "result.zarr",
                                    process=process)
    assert result is validation and not result.quick_passed
    assert events == ["load_source_contract", "load_target_contract", "validate_source", "read", "process", "write", "close_source", "validate"]


@pytest.mark.parametrize("dry_run", [False, True])
def test_unwritten_product_is_not_reopened(tmp_path, monkeypatch, dry_run):
    creation = SimpleNamespace(coverage={"written": False, "dry_run": dry_run})

    def unexpected(*args, **kwargs):
        pytest.fail("Unwritten product must not be opened")

    monkeypatch.setattr(pipeline, "validate_zarr", unexpected)
    assert pipeline.validate_product(tmp_path / "absent.zarr", object(), creation) is creation


@pytest.mark.parametrize("quick_passed", [False, True])
def test_cli_uses_final_validation_status(monkeypatch, quick_passed):
    report = SimpleNamespace(
        coverage={"written": True, "conformance_passed": True},
        quick_passed=quick_passed, to_dict=lambda: {},
    )
    monkeypatch.setattr(pipeline, "build_dataset", lambda *args, **kwargs: report)
    args = ["--processor", "oras5"]
    for option in ("source-config", "target-spec", "source", "destination", "mean-nc", "std-nc", "mask-nc"):
        args.extend([f"--{option}", f"/test/{option}"])
    assert pipeline.main(args) == (0 if quick_passed else 1)


@pytest.mark.parametrize("actual_units", ["K", "degC", None])
def test_source_validation_checks_actual_files_and_units(tmp_path, actual_units):
    path = tmp_path / "input.nc"
    field = xr.DataArray([280.0], dims="time", coords={"time": [0]}, name="temperature")
    if actual_units:
        field.attrs["units"] = actual_units
    field.to_dataset().to_netcdf(path, engine="scipy")
    description = {
        "name": "physical", "format": "netcdf", "layout": {}, "example": "input.nc",
        "mapping": {"variables": {"t2m": {"source": "temperature", "source_unit": "K"}}},
    }
    if actual_units == "K":
        summary = pipeline.validate_source(path, description)
        assert summary["files_checked"] == 1
    else:
        with pytest.raises(ValueError, match="Source contract mismatch.*expected source units"):
            pipeline.validate_source(path, description)


def test_source_validation_rejects_missing_packed_structure(tmp_path):
    path = tmp_path / "input.nc"
    xr.Dataset({"other": ("row", [1.0])}).to_netcdf(path, engine="scipy")
    description = {
        "name": "packed", "format": "netcdf", "layout": {}, "example": "input.nc",
        "mapping": {"packed_variable": "data", "packed_dimension": "channel"},
    }
    with pytest.raises(ValueError, match="Expected 'data' with dimension 'channel'"):
        pipeline.validate_source(path, description)


def test_source_validation_does_not_treat_optional_mapping_as_required(tmp_path):
    path = tmp_path / "input.nc"
    xr.Dataset({"other": ("row", [1.0])}).to_netcdf(path, engine="scipy")
    description = {
        "name": "partial", "format": "netcdf", "layout": {}, "example": "input.nc",
        "mapping": {"variables": {"t2m": {"source": "temperature", "source_unit": "K"}}},
    }
    assert pipeline.validate_source(path, description)["files_checked"] == 1


def test_invalid_source_stops_before_processing_or_writing(tmp_path, monkeypatch):
    description = {"name": "source", "format": "netcdf", "layout": {}, "example": "input.nc"}
    monkeypatch.setattr(pipeline, "load_source_contract", lambda *args: description)
    monkeypatch.setattr(pipeline, "load_target_contract", lambda *args: object())

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid input must stop before read/process/write")

    monkeypatch.setattr(pipeline, "read_source", unexpected)
    monkeypatch.setattr(pipeline, "write_product", unexpected)
    with pytest.raises(FileNotFoundError, match="Source path does not exist"):
        pipeline.build_dataset(description, object(), tmp_path / "missing", tmp_path / "output",
                               process=unexpected)
