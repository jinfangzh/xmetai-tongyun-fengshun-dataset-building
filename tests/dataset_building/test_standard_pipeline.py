"""Contract-bound example processing without external archives or Zarr fixtures."""

from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from datakit.schema import load_schema
from pipelines.dataset import build as pipeline, preparation
from datakit.transforms import preprocess_channel_units
from examples.dataset_building import example_build_era5 as era5
from examples.dataset_building import example_build_oras5 as oras5


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
    function = era5.process if spec.product.product_id.startswith("era5") else oras5.process
    return partial(function, mean_path=root / "mean.nc", std_path=root / "std.nc", mask_path=root / "mask.nc")


def auxiliary_access(arrays):
    """Fake only the IO boundary; production conversion code selects and loads variables."""
    return lambda path: SimpleNamespace(read=lambda **options: xr.Dataset(arrays))


def value_processor(spec):
    module = era5 if spec.product.product_id.startswith("era5") else oras5
    process = module.process

    def values(raw, target, mean, std):
        lat, lon = target.variant.grid.coordinates()
        mask = xr.DataArray(np.stack([np.ones((len(lat), len(lon)), dtype='u1'),
                                      np.zeros((len(lat), len(lon)), dtype='u1')]),
                            dims=('mask_channel', 'lat', 'lon'),
                            coords={'mask_channel': ['land_mask', 'sea_mask'], 'lat': lat, 'lon': lon})
        arrays = {'mean': mean, 'std': std, 'mask': mask}
        with patch.object(module.io, 'open', side_effect=auxiliary_access(arrays)):
            return process(raw, {}, target, mean_path='mean', std_path='std', mask_path='mask').data
    return values


def reference_physical(raw, spec):
    """Pre-refactor aggregate operator, used as an independent numerical reference."""
    data = raw.data.astype('float32')
    if not spec.product.product_id.startswith('era5'):
        return data.sel(channel=list(spec.product.channel_names))
    derived = []
    for name, height in (('ws10m', '10m'), ('ws100m', '100m')):
        if name in spec.product.channel_names:
            wind = np.hypot(data.sel(channel=f'u{height}', drop=True),
                            data.sel(channel=f'v{height}', drop=True))
            derived.append(wind.expand_dims(channel=[name]))
    if derived:
        names = [str(item.channel.values[0]) for item in derived]
        data = xr.concat([data.drop_sel(channel=[n for n in names if n in data.channel]),
                          *derived], dim='channel', join='exact')
    return preprocess_channel_units(data.sel(channel=list(spec.product.channel_names)),
                                    accumulation_hours=6.)


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
    monkeypatch.setattr(era5.io, 'open', auxiliary_access(auxiliaries))
    monkeypatch.setattr(pipeline.io, "open_dataset", lambda path, description:
                        SimpleNamespace(read=lambda **options: reader(path, description, **options)))
    report = pipeline.build_zarr(
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
        np.testing.assert_allclose(result.sel(channel="tp"), np.float16(np.log1p(1000)))
        np.testing.assert_allclose(result.sel(channel="ssr"), np.float16(1 / 21600))
        if "ws10m" in result.channel:
            np.testing.assert_allclose(result.sel(channel="ws10m"), np.float16(np.sqrt(2)))
    else:
        np.testing.assert_allclose(result, 1)


def test_shifted_mask_is_rejected_instead_of_aligned_silently(inputs, monkeypatch):
    spec, raw, aux = inputs
    aux = {**aux, 'mask': aux['mask'].assign_coords(lat=aux['mask'].lat + .5)}
    monkeypatch.setattr(era5.io, 'open', auxiliary_access(aux))
    with pytest.raises(ValueError):
        processor(spec)(raw, {}, spec)


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
        pipeline.build_zarr({}, "unused.yaml", tmp_path / "input.nc", target,
                               process=oras5.process)


def test_reader_is_closed_when_processing_fails(inputs, tmp_path, monkeypatch):
    spec, raw, aux = inputs
    closed = []
    raw.set_close(lambda: closed.append(True))
    monkeypatch.setattr(pipeline, "validate_source", lambda *args: None)
    monkeypatch.setattr(era5.io, 'open', auxiliary_access(aux))
    monkeypatch.setattr(pipeline.io, "open_dataset", lambda *args:
                        SimpleNamespace(read=lambda **kwargs: raw))

    def fail(*args):
        raise ValueError("Explicit strategy failed")

    with pytest.raises(ValueError, match="strategy failed"):
        pipeline.build_zarr(
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
    report = pipeline.build_zarr(
        {"name": "packed_nc", "format": "netcdf", "layout": {}, "example": "input.nc"},
        spec, source, tmp_path / "result.zarr",
        process=processor(spec, tmp_path), dry_run=True,
    )
    assert report.coverage["conformance_passed"], report.to_dict()


def test_processor_algorithm_does_not_interpret_target_recipe(inputs):
    spec, raw, aux = inputs
    first, *rest = spec.product.channels
    changed = replace(first, preprocessing=({"operation": "log1p"},))
    altered = replace(spec, product=replace(spec.product, channels=(changed, *rest)))
    # Processing is explicit Python; recipe metadata must not select new operations.
    before = value_processor(spec)(raw, spec, aux["mean"], aux["std"])
    after = value_processor(spec)(raw, altered, aux["mean"], aux["std"])
    np.testing.assert_array_equal(after.values, before.values)


def test_invalid_mask_cannot_wrap_into_valid_uint8(inputs, monkeypatch):
    spec, raw, aux = inputs
    mask = aux['mask'].astype('int16')
    mask.values[0, 0, 0] = 257
    aux = {**aux, 'mask': mask}
    monkeypatch.setattr(era5.io, 'open', auxiliary_access(aux))
    with pytest.raises(ValueError, match='binary integer'):
        processor(spec)(raw, {}, spec)


@pytest.mark.parametrize("dry_run", [False, True])
def test_unwritten_product_is_not_reopened(tmp_path, monkeypatch, dry_run):
    creation = SimpleNamespace(coverage={"written": False, "dry_run": dry_run,
                                         "conformance_passed": dry_run})
    raw = xr.Dataset()
    closed = []
    raw.set_close(lambda: closed.append(True))
    monkeypatch.setattr(pipeline, "load_source_contract", lambda *args: {})
    monkeypatch.setattr(pipeline, "load_target_contract", lambda *args: object())
    monkeypatch.setattr(pipeline, "validate_source", lambda *args: None)
    monkeypatch.setattr(pipeline, "read_source", lambda *args: raw)
    monkeypatch.setattr(pipeline, "assemble_standard_dataset", lambda product, spec: product)
    monkeypatch.setattr(pipeline, "write_product", lambda *args, **kwargs: creation)

    def unexpected(*args, **kwargs):
        pytest.fail("Unwritten product must not be opened")

    monkeypatch.setattr(pipeline, "validate_zarr", unexpected)
    result = pipeline.build_zarr({}, "target", tmp_path / "source", tmp_path / "result",
                                    process=lambda *args: xr.Dataset(), dry_run=dry_run)
    assert result.creation_report is creation
    assert not result.quick_passed and closed == [True]


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


def varying_input(inputs):
    spec, raw, aux = inputs
    raw = raw.copy(deep=True)
    raw.data.values[:] = np.arange(1, 7, dtype='f4').reshape(1, 1, 2, 3)
    raw['source_land'] = xr.DataArray(
        [[0, .4, .5], [.6, 1, 0]], dims=('lat', 'lon'),
        coords={'lat': raw.lat, 'lon': raw.lon})
    from pipelines.dataset.preparation import generate_mask
    aux = {**aux, 'mask': generate_mask(raw, spec, 'source_land', .5)}
    return spec, raw, aux


@pytest.mark.parametrize('provided', [(), ('mean',), ('std',), ('mean', 'std')])
def test_optional_statistics_in_preprocessed_domain(inputs, monkeypatch, provided):
    spec, raw, aux = varying_input(inputs)
    # This numerical recipe test has no fixed production statistics.
    spec = replace(spec, product=replace(spec.product, effective_statistics={}))
    monkeypatch.setattr(era5.io, 'open', auxiliary_access(aux))
    module = era5 if spec.product.product_id.startswith('era5') else oras5
    process = module.process
    channels = list(spec.product.channel_names)
    expected = reference_physical(raw, spec)
    result = process(raw, {}, spec,
                     mean_path='external' if 'mean' in provided else None,
                     std_path='external' if 'std' in provided else None, mask_path='external')
    channels = list(spec.product.channel_names)
    mean = (aux['mean'].sel(channel=channels) if 'mean' in provided else
            expected.astype('float64').mean(('time', 'lat', 'lon'), skipna=True).astype('float32'))
    std = (aux['std'].sel(channel=channels) if 'std' in provided else
           expected.astype('float64').std(('time', 'lat', 'lon'), skipna=True, ddof=0).astype('float32'))
    np.testing.assert_array_equal(result['mean'], mean)
    np.testing.assert_array_equal(result['std'], std)
    np.testing.assert_array_equal(result.data, ((expected - mean) / std).astype('float16'))
    np.testing.assert_array_equal(result['mask'].sel(mask_channel='land_mask'),
                                  [[0, 0, 1], [1, 1, 0]])
    np.testing.assert_array_equal(result['mask'].sum('mask_channel'), 1)


@pytest.mark.parametrize('fault', ['extra', 'missing', 'duplicate'])
def test_generated_statistics_reject_inexact_time_coverage(inputs, fault):
    spec, raw, aux = varying_input(inputs)
    if fault == 'missing':
        raw = raw.isel(time=slice(0, 0))
    else:
        other = raw if fault == 'duplicate' else raw.assign_coords(time=raw.time + np.timedelta64(6, 'h'))
        raw = xr.concat([raw, other], dim='time')
    process = era5.process if spec.product.product_id.startswith('era5') else oras5.process
    with pytest.raises(ValueError, match='time coverage'):
        process(raw, {}, spec)


@pytest.mark.parametrize('fault', ['constant', 'all_nan'])
def test_invalid_generated_statistics_fail(inputs, monkeypatch, fault):
    spec, raw, aux = inputs
    raw = raw.copy(deep=True)
    if fault == 'all_nan':
        raw.data.values[:] = np.nan
    process = era5.process if spec.product.product_id.startswith('era5') else oras5.process
    module = era5 if spec.product.product_id.startswith('era5') else oras5
    monkeypatch.setattr(module.io, 'open', auxiliary_access(aux))
    with pytest.raises(ValueError, match='finite|positive'):
        process(raw, {}, spec, mask_path='external')


@pytest.mark.parametrize('fault', ['unspecified', 'no_threshold', 'nan', 'out_of_range', 'bad_threshold'])
def test_generated_mask_rejects_ambiguous_or_invalid_input(inputs, fault):
    from pipelines.dataset.preparation import generate_mask
    spec, raw, aux = varying_input(inputs)
    variable, threshold = 'source_land', .5
    if fault == 'unspecified':
        variable = None
    elif fault == 'no_threshold':
        threshold = None
    elif fault == 'nan':
        raw.source_land.values[0, 0] = np.nan
    elif fault == 'out_of_range':
        raw.source_land.values[0, 0] = 2
    else:
        threshold = float('nan')
    with pytest.raises(ValueError):
        generate_mask(raw, spec, variable, threshold)


def test_binary_and_packed_source_mask(inputs):
    from pipelines.dataset.preparation import generate_mask
    spec, raw, aux = inputs
    land = aux['mask'].sel(mask_channel='land_mask', drop=True)
    raw = raw.assign(source_land=land)
    result = generate_mask(raw, spec, 'source_land', None)
    xr.testing.assert_equal(result, aux['mask'])
    channel = str(raw.channel.values[0])
    raw.data.loc[{'channel': channel}] = 0
    packed = generate_mask(raw, spec, channel, None)
    assert not packed.sel(mask_channel='land_mask').values.any()


def test_guard_rejects_generated_statistics_against_fixed_target(inputs, tmp_path, monkeypatch):
    spec, raw, aux = varying_input(inputs)
    first = spec.product.channels[0]
    product = replace(spec.product, effective_statistics={first.variable: (0., 1.)})
    spec = replace(spec, product=product)
    module = era5 if spec.product.product_id.startswith('era5') else oras5
    process = module.process
    monkeypatch.setattr(pipeline, 'validate_source', lambda *args: None)
    monkeypatch.setattr(pipeline, 'read_source', lambda *args: raw)
    monkeypatch.setattr(module.io, 'open', auxiliary_access(aux))
    report = pipeline.build_zarr(
        {'name': 'packed', 'layout': {}, 'example': 'input.nc'}, spec,
        tmp_path / 'input.nc', tmp_path / 'result.zarr', dry_run=True,
        process=partial(process, mask_path='external'))
    assert not report.coverage['conformance_passed'] and not report.coverage['written']
    assert report.failures[0]['stage'] == 'input_check'


def test_framework_passes_both_contracts_without_adapting_product(tmp_path, monkeypatch):
    import json
    from pipelines.dataset.reports import ValidationReport
    raw = xr.Dataset()
    closed = []
    raw.set_close(lambda: closed.append(True))
    target = object()
    description = {'name': 'input'}
    # Incomplete data must be rejected without repairing or mutating it.
    product = xr.Dataset({'data': ('sample', np.array([1.5], dtype='float64'))})
    before = product.copy(deep=True)
    monkeypatch.setattr(pipeline, 'load_source_contract', lambda *args: description)
    monkeypatch.setattr(pipeline, 'load_target_contract', lambda *args: target)
    monkeypatch.setattr(pipeline, 'validate_source', lambda *args: None)
    monkeypatch.setattr(pipeline, 'read_source', lambda *args: raw)
    def process(dataset, source_contract, target_contract):
        assert dataset is raw and source_contract is description and target_contract is target
        return product
    def create(dataset, *args, **kwargs):
        assert dataset is product
        report = ValidationReport('sample', 'create', {'checked_scope': 'input'},
                                  coverage={'conformance_passed': False, 'written': False})
        report.add('layout.data', 'fail', 'standard_conformance')
        return report
    monkeypatch.setattr(pipeline, 'create_zarr', create)
    result = pipeline.build_zarr({}, target, tmp_path/'input', tmp_path/'output', process=process)
    assert closed == [True] and result.failures[0]['stage'] == 'assemble'
    xr.testing.assert_identical(product, before)
    json.dumps(result.to_dict())


def test_build_writes_and_validates_real_product_after_closing_source(published, tmp_path, monkeypatch):
    import json
    path, spec = published
    events = []
    opened = xr.open_zarr(path, chunks=None, mask_and_scale=False)
    close = opened.close
    def close_source():
        close()
        events.append('closed')
    # Use a real lazy Dataset, while recording resource/validation ordering.
    from contextlib import contextmanager
    @contextmanager
    def reader(*args):
        try:
            yield opened
        finally:
            close_source()
    monkeypatch.setattr(pipeline, 'read_source', reader)
    validate = pipeline.validate_zarr
    def validate_after_close(*args, **kwargs):
        assert events == ['closed']
        return validate(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'validate_zarr', validate_after_close)
    target = tmp_path/'built.zarr'
    result = pipeline.build_zarr(
        {'name': 'published', 'format': 'zarr', 'layout': {}, 'example': path.name},
        spec, path, target, process=lambda raw, source, target: raw)
    assert result.coverage['written'] and result.quick_passed, result.to_dict()
    assert result.creation_report is not None and result.validation_report is not None
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize('fault', ['write', 'validate'])
def test_build_failure_preserves_partial_output_and_resources(tmp_path, monkeypatch, fault):
    from pipelines.dataset.reports import ValidationReport
    closed = []
    raw = xr.Dataset()
    raw.set_close(lambda: closed.append(True))
    monkeypatch.setattr(pipeline, 'load_source_contract', lambda *args: {})
    monkeypatch.setattr(pipeline, 'load_target_contract', lambda *args: object())
    monkeypatch.setattr(pipeline, 'validate_source', lambda *args: None)
    monkeypatch.setattr(pipeline, 'read_source', lambda *args: raw)
    target = tmp_path/'partial.zarr'
    def write(*args, **kwargs):
        target.mkdir()
        result = ValidationReport('sample', 'create', {'checked_scope': 'input'},
                                  coverage={'written': fault != 'write', 'conformance_passed': True})
        result.add('create.write', 'fail' if fault == 'write' else 'pass', 'data_integrity')
        return result
    def validate(*args, **kwargs):
        assert fault != 'write' and closed == [True]
        raise OSError('readback failed')
    monkeypatch.setattr(pipeline, 'assemble_standard_dataset', lambda product, spec: product)
    monkeypatch.setattr(pipeline, 'write_product', write)
    monkeypatch.setattr(pipeline, 'validate_zarr', validate)
    result = pipeline.build_zarr({}, object(), tmp_path/'input', target,
                                    process=lambda *args: xr.Dataset())
    assert target.is_dir() and closed == [True]
    assert not result.quick_passed and result.failures[0]['stage'] == fault


def test_build_import_does_not_load_examples():
    import subprocess
    import sys
    code = "from pipelines.dataset import build_zarr, BuildReport; import sys; assert not any(n == 'examples' or n.startswith('examples.') for n in sys.modules)"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_generated_statistics_ignore_partial_nan_with_known_values(inputs, monkeypatch):
    spec, raw, aux = varying_input(inputs)
    spec = replace(spec, product=replace(spec.product, effective_statistics={}))
    raw.data.values[:, :, 0, 0] = np.nan
    module = era5 if spec.product.product_id.startswith('era5') else oras5
    process = module.process
    monkeypatch.setattr(module.io, 'open', auxiliary_access(aux))
    result = process(raw, {}, spec, mask_path='external')
    # Every unchanged channel contains [NaN, 2, 3, 4, 5, 6].
    channel = spec.product.channels[0].name
    assert float(result['mean'].sel(channel=channel)) == 4.
    np.testing.assert_allclose(result['std'].sel(channel=channel), np.sqrt(2), rtol=1e-7)
    assert np.isnan(result.data.sel(channel=channel).values[0, 0, 0])


def test_concrete_recipe_requires_external_mask(inputs):
    spec, raw, _ = varying_input(inputs)
    process = era5.process if spec.product.product_id.startswith('era5') else oras5.process
    with pytest.raises(ValueError, match='requires an external mask_path'):
        process(raw, {}, spec)


def test_pipeline_adds_standard_metadata_without_mutating_processed_arrays(inputs, tmp_path, monkeypatch):
    spec, raw, aux = inputs
    monkeypatch.setattr(era5.io, 'open', auxiliary_access(aux))
    processed = processor(spec)(raw, {}, spec)
    before = processed.copy(deep=True)
    assert not processed.attrs and 'channel_info' not in processed.channel.attrs
    monkeypatch.setattr(pipeline, 'validate_source', lambda *args: None)
    monkeypatch.setattr(pipeline, 'read_source', lambda *args: raw)
    create = pipeline.create_zarr
    def check_assembled(product, *args, **kwargs):
        assert product.attrs['dataset_id'] == spec.dataset_id
        assert 'channel_info' in product.channel.attrs
        for name in ('data', 'mean', 'std', 'mask'):
            np.testing.assert_array_equal(product[name], before[name])
        return create(product, *args, **kwargs)
    monkeypatch.setattr(pipeline, 'create_zarr', check_assembled)
    result = pipeline.build_zarr(
        {'name': 'packed', 'layout': {}, 'example': 'input.nc'}, spec,
        tmp_path/'input.nc', tmp_path/'result.zarr',
        process=lambda *args: processed, dry_run=True)
    assert result.coverage['conformance_passed'], result.to_dict()
    assert result.stages['assemble'] == 'passed'
    xr.testing.assert_identical(processed, before)
