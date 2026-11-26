"""Source failures enqueue repairs before any Zarr write; READY stays immutable."""

import fcntl
import argparse
import json
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import xarray as xr

from datakit import io
from datakit.runtime.daily_integrity import LEVELS
from datakit.schema import load_mapping
from datakit.schema.semantics import TemporalKind
from examples.dataset_building import build_ready_tongyun_days as consumer
from examples.dataset_building import example_build_fengshun_daily as fengshun
from examples.dataset_building import example_build_tongyun_daily as tongyun
from examples.dataset_building import source_repairs


@pytest.fixture(params=[tongyun, fengshun], ids=['tongyun', 'fengshun'])
def sources(request, tmp_path, monkeypatch):
    builder = request.param
    description = io.get_description(builder.SOURCE_CONFIG)
    mapping = load_mapping(description)
    day = date(2026, 11, 17)
    raw = tmp_path / 'raw'
    root = raw / ('daily' if builder is tongyun else 'cds_daily_1p5')
    paths = {}
    for variable, template in description['layout']['files'].items():
        accumulated = mapping.variables[variable].temporal_kind == TemporalKind.INTERVAL_ACCUMULATION
        days = [day - timedelta(days=1), day] if builder is tongyun and accumulated else [day]
        for selected in days:
            hours = range(24) if builder is tongyun and accumulated else range(0, 24, 6) if builder is tongyun else [0]
            coords = {'valid_time': np.asarray([f'{selected}T{h:02d}:00' for h in hours],
                                               dtype='datetime64[ns]'),
                      'latitude': [90., 0., -90.], 'longitude': [0., 90., 180., 270.]}
            if variable in {'z', 't', 'u', 'v', 'q'}:
                coords['pressure_level'] = LEVELS
                coords = {k: coords[k] for k in ('valid_time', 'pressure_level', 'latitude', 'longitude')}
            dataset = xr.Dataset({mapping.variables[variable].source:
                                  (tuple(coords), np.ones(tuple(len(v) for v in coords.values()), dtype='f4'))},
                                 coords={k: list(v) for k, v in coords.items()})
            path = root / template.format(day=datetime.combine(selected, time()))
            path.parent.mkdir(parents=True, exist_ok=True)
            if builder is fengshun:
                with ZipFile(path, 'w') as archive:
                    archive.writestr('source.nc', dataset.to_netcdf(engine='scipy'))
            else:
                dataset.to_netcdf(path, engine='scipy')
            paths[(variable, selected)] = path
    records = raw / '_records' / f'{builder.DOWNLOAD_PRODUCT}_daily'
    records.mkdir(parents=True)
    watermark = records / 'watermark.json'
    watermark.write_text(json.dumps({'latest_complete_day': str(day)}))
    config = {'source': root, 'download_watermark': watermark, 'repair_queue': records / 'repairs',
              'output_root': tmp_path / 'products', 'report_dir': tmp_path / 'reports',
              **{name: tmp_path / name for name in ('mean_nc', 'std_nc', 'mask_nc', 'const_nc', 'weight_npy')}}
    grid = SimpleNamespace(coordinates=lambda: (np.array([90., 0., -90.]), np.arange(0., 360., 90.)))
    monkeypatch.setattr(source_repairs, 'load_schema', lambda _: (SimpleNamespace(
        variant=SimpleNamespace(grid=grid)), None))
    return builder, config, day, paths


def break_field(builder, path):
    from datakit.runtime.daily_integrity import open_netcdf
    with open_netcdf(path) as source:
        data = source.load()
    if builder is tongyun:
        data = data.isel(valid_time=slice(0, 3))
    else:
        data = data.assign_coords(valid_time=[np.datetime64('2026-11-16')])
    if builder is fengshun:
        with ZipFile(path, 'w') as archive:
            archive.writestr('source.nc', data.to_netcdf(engine='scipy'))
    else:
        data.to_netcdf(path, engine='scipy')


def test_source_failure_queues_only_bad_variable_and_dry_run_is_read_only(sources, monkeypatch):
    builder, config, day, paths = sources
    break_field(builder, paths['q', day])
    monkeypatch.setattr(builder, 'main', lambda _: pytest.fail('Must not build incomplete inputs'))
    assert consumer.run(config, builder=builder, write=False) == 0
    assert not config['repair_queue'].exists()
    before = config['download_watermark'].read_bytes()
    assert consumer.run(config, builder=builder, write=True) == 0
    queue = source_repairs.queue_for(config, builder)
    item, = queue.items()
    assert item['variable'] == 'specific_humidity' and item['date'] == str(day)
    assert item['status'] == 'pending'
    assert item['relative_target'].endswith('specific_humidity_20261117' + paths['q', day].suffix)
    assert config['download_watermark'].read_bytes() == before
    assert not config['output_root'].exists()
    # Repeated build polls must not reset cooldown/attempt count.
    claim, = queue.claim()
    queue.finish(claim, success=False, error='temporary network issue')
    assert consumer.run(config, builder=builder, write=True) == 0
    again, = queue.items()
    assert again['attempts'] == 1 and again['next_attempt_at']


def test_missing_temperature_still_discovers_day(sources, monkeypatch):
    builder, config, day, paths = sources
    paths['t', day].unlink()
    monkeypatch.setattr(builder, 'main', lambda _: pytest.fail('Must wait for missing file'))
    consumer.run(config, builder=builder, write=True)
    item, = source_repairs.queue_for(config, builder).items()
    assert item['variable'] == 'temperature'


def test_ready_products_are_skipped_even_if_sources_removed(sources, monkeypatch):
    builder, config, day, paths = sources
    _, output, marker = consumer.product_paths(config, day, builder=builder)
    output.mkdir(parents=True)
    marker.parent.mkdir()
    marker.write_text('{}')
    paths['q', day].unlink()
    monkeypatch.setattr(consumer, 'inspect_day', lambda *a, **k: pytest.fail('Do not recheck READY'))
    assert consumer.run(config, builder=builder, write=True) == 0
    assert not config['repair_queue'].exists()


def test_fixed_sources_can_build_and_publication_holds_source_read_lock(sources, monkeypatch):
    builder, config, day, paths = sources
    queue = source_repairs.queue_for(config, builder)
    queue.enqueue(paths['q', day], 'earlier incomplete file')
    queue.resolve(paths['q', day])
    def fake_build(options):
        # Only an empty stand-in directory, no real Zarr data is written.
        with (queue.records / 'source.lock').open('a') as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Path(options[options.index('--destination') + 1]).mkdir(parents=True)
        return 0
    monkeypatch.setattr(builder, 'main', fake_build)
    assert consumer.run(config, builder=builder, write=True) == 0
    _, output, marker = consumer.product_paths(config, day, builder=builder)
    assert output.is_dir() and marker.exists()
    assert len(list(config['repair_queue'].glob('*.json'))) == 1


def test_tongyun_boundary_failure_repairs_preceding_day(sources, monkeypatch):
    builder, config, day, paths = sources
    if builder is not tongyun:
        pytest.skip('Fengshun has no preceding-day accumulation input')
    paths['tp', day - timedelta(days=1)].unlink()
    consumer.run(config, builder=builder, write=True)
    item, = source_repairs.queue_for(config, builder).items()
    assert item['variable'] == 'total_precipitation'
    assert item['date'] == str(day - timedelta(days=1))
    # Only five boundary files on the preceding date must not trigger a whole
    # nonexistent day of pressure/surface repair requests.
    assert len(list(config['repair_queue'].glob('*.json'))) == 1


def test_build_queue_is_consumed_by_download_deployment(sources, monkeypatch):
    """Offline round trip across the two independently deployed repositories."""
    download = Path(__file__).resolve().parents[3] / 'xmetai-download-tongyun-infer/examples/dataset_downloader/cds_download'
    if not download.is_dir():
        pytest.skip('Cross-deployment test requires the sibling download checkout')
    monkeypatch.syspath_prepend(str(download))
    import daily_integrity as downloader_integrity
    from grouped_download import GroupedVariableRunner
    builder, config, day, paths = sources
    target = paths['q', day]
    break_field(builder, target)
    consumer.run(config, builder=builder, write=True)
    queue = source_repairs.queue_for(config, builder)
    before = config['download_watermark'].read_bytes()
    hours = ['00:00', '06:00', '12:00', '18:00'] if builder is tongyun else ['00:00']
    request = {'variable': ['specific_humidity'], 'year': ['2026'], 'month': ['11'], 'day': ['17'],
               'time': hours, 'pressure_level': [str(v) for v in LEVELS], 'grid': [90., 90.]}
    task = {'dataset': 'reanalysis-era5-pressure-levels', 'request': request, 'target': str(target)}
    calls = []
    class Client:
        def retrieve(self, dataset, request, output):
            calls.append(request)
            coords = {'valid_time': np.asarray([str(day) + 'T' + hour for hour in hours], dtype='datetime64[ns]'),
                      'pressure_level': list(LEVELS), 'latitude': [90., 0., -90.],
                      'longitude': [0., 90., 180., 270.]}
            data = xr.Dataset({'q': (tuple(coords), np.ones(tuple(len(v) for v in coords.values()), dtype='f4'))},
                              coords=coords)
            data.to_netcdf(output, engine='h5netcdf')
    monkeypatch.setitem(sys.modules, 'cdsapi', SimpleNamespace(Client=Client))
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=date.fromisoformat)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    downloader_integrity.add_repair_arguments(parser)
    args = parser.parse_args(['--repair'])
    _, only, failures = downloader_integrity.prepare_repairs(
        args, parser, raw_root=queue.raw_root, records=queue.records, product=queue.product,
        tasks_for_day=lambda _: [task], runner_class=GroupedVariableRunner)
    assert only and failures == 0 and len(calls) == 1
    assert queue.items()[0]['status'] == 'done'
    assert list(consumer.ready_days(config, day, builder=builder)) == [day]
    assert config['download_watermark'].read_bytes() == before
    assert not config['output_root'].exists()  # The test never writes a Zarr.
