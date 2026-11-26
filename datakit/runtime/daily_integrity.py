"""ERA5 daily-file validation and v1 repair-queue protocol.

Bundled in both the download and build deployments. Queue operations use a
short POSIX lock; source readers/publication use records/source.lock. No Zarr
or full field-quality scan is performed here. Valid land-masked SST is allowed.
"""

import hashlib
import json
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile, is_zipfile


VARIABLE_NAMES = {
    'geopotential': ('z',), 'temperature': ('t',),
    'u_component_of_wind': ('u',), 'v_component_of_wind': ('v',),
    'specific_humidity': ('q',), '2m_temperature': ('t2m', '2t'),
    '2m_dewpoint_temperature': ('d2m', '2d'), 'sea_surface_temperature': ('sst',),
    '10m_u_component_of_wind': ('u10', 'u10m', '10u'),
    '10m_v_component_of_wind': ('v10', 'v10m', '10v'),
    '100m_u_component_of_wind': ('u100', 'u100m', '100u'),
    '100m_v_component_of_wind': ('v100', 'v100m', '100v'),
    'mean_sea_level_pressure': ('msl',), 'total_column_water_vapour': ('tcwv',),
    'surface_net_solar_radiation': ('ssr',),
    'surface_solar_radiation_downwards': ('ssrd',),
    'total_sky_direct_solar_radiation_at_surface': ('fdir',),
    'top_net_thermal_radiation': ('ttr',), 'total_precipitation': ('tp',),
}
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
VALIDATOR_VERSION = 1


def signature(path):
    try:
        stat = Path(path).stat()
        return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    except FileNotFoundError:
        return None


@contextmanager
def file_lock(path, *, exclusive=True):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_dataset(dataset, *, variable, times, levels=None, grid=0.25, source_name=None):
    """Check exact coordinates and reject wholly empty time/level slices.

    Values are read in bounded latitude slabs until a finite value is found.
    This catches union-aligned all-NaN frames without rejecting SST's land mask.
    Other isolated missing values remain the build quality contract's concern.
    """
    import numpy as np
    aliases = {variable, *VARIABLE_NAMES.get(variable, ())}
    if source_name:
        aliases.add(source_name)
    matches = [name for name, value in dataset.data_vars.items()
               if name in aliases or value.attrs.get('GRIB_shortName') in aliases]
    if len(matches) != 1:
        raise ValueError(f'{variable}: expected one field, found {matches}')
    field = dataset[matches[0]]
    renames = {old: new for old, new in {'valid_time': 'time', 'pressure_level': 'level',
                'latitude': 'lat', 'longitude': 'lon'}.items() if old in field.dims}
    field = field.rename(renames)
    expected = np.asarray(times, dtype='datetime64[ns]')
    if 'time' not in field.coords or field.time.dims != ('time',):
        raise ValueError(f'{variable}: missing labeled time dimension')
    observed = field.time.values.astype('datetime64[ns]')
    missing = np.setdiff1d(expected, observed)
    if missing.size:
        raise ValueError(f'{variable}: missing {missing.size} timestamp(s), first={missing[0]}')
    if len(observed) != len(expected) or len(np.unique(observed)) != len(observed):
        raise ValueError(f'{variable}: duplicate or unexpected timestamps')
    if levels is not None:
        if ('level' not in field.coords or field.level.dims != ('level',)
                or not np.array_equal(np.sort(field.level.values.astype(float)), np.sort(levels))):
            raise ValueError(f'{variable}: pressure levels do not match the request')
    for name, wanted in [('lat', np.arange(90., -90. - grid / 2, -grid)),
                         ('lon', np.arange(0., 360., grid))]:
        if name not in field.coords or field[name].dims != (name,):
            raise ValueError(f'{variable}: missing {name} coordinate')
        actual = field[name].values.astype(float)
        if name == 'lon':
            actual = actual % 360
        if len(actual) != len(wanted) or not np.allclose(np.sort(actual), np.sort(wanted), atol=1e-5, rtol=0):
            raise ValueError(f'{variable}: incomplete global {grid}-degree {name} grid')
    for t in range(field.sizes['time']):
        for level in range(field.sizes['level'] if 'level' in field.dims else 1):
            selection = {'time': t}
            if 'level' in field.dims:
                selection['level'] = level
            frame = field.isel(selection)
            finite = False
            for start in range(0, frame.sizes['lat'], 64):
                if np.isfinite(frame.isel(lat=slice(start, start + 64)).values).any():
                    finite = True
                    break
            if not finite:
                raise ValueError(f'{variable}: wholly missing time/level slice at {observed[t]}')


@contextmanager
def open_netcdf(path):
    """Read direct NetCDF or one-member ZIP without trusting member paths."""
    import shutil
    import xarray as xr
    path = Path(path)
    with TemporaryDirectory(prefix='era5-check-') as directory:
        payload = path
        if is_zipfile(path):
            with ZipFile(path) as archive:
                members = [m for m in archive.infolist() if not m.is_dir()
                           and m.filename.lower().endswith(('.nc', '.nc4', '.netcdf'))]
                if len(members) != 1:
                    raise ValueError(f'{path}: expected one NetCDF member')
                payload = Path(directory) / 'source.nc'
                with archive.open(members[0]) as incoming, payload.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
        with xr.open_dataset(payload) as dataset:
            yield dataset


def validate_task(task, path):
    request = task['request']
    def single(key):
        value = request[key]
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError('Daily integrity requires one requested date')
            value = value[0]
        return value
    day = date(int(single('year')), int(single('month')), int(single('day')))
    hours = request.get('time', ['00:00'])
    times = [f'{day}T{hour}' for hour in hours]
    daily = 'daily_statistic' in request
    grid = request.get('grid', [1.5, 1.5] if daily else [0.25, 0.25])
    if isinstance(grid, str):
        grid = grid.split('/')
    if float(grid[0]) != float(grid[1]):
        raise ValueError('Daily integrity requires a regular global square grid')
    with open_netcdf(path) as dataset:
        validate_dataset(dataset, variable=request['variable'][0], times=times,
                         levels=[float(v) for v in request['pressure_level']]
                         if 'pressure_level' in request else None, grid=float(grid[0]))


class RepairQueue:
    """v1 queue: one JSON per raw-relative target, shared by both deployments."""

    def __init__(self, records, raw_root, product, *, max_attempts=3, retry_hours=6):
        self.records = Path(records)
        self.raw_root = Path(raw_root).resolve()
        self.product = product
        self.directory = self.records / 'repairs'
        self.max_attempts = max_attempts
        self.retry_hours = retry_hours

    def relative(self, target):
        return str(Path(target).resolve().relative_to(self.raw_root))

    def item_path(self, target):
        key = hashlib.sha256(self.relative(target).encode()).hexdigest()
        return self.directory / f'{key}.json'

    def enqueue(self, target, reason):
        relative = self.relative(target)
        path = self.item_path(target)
        with file_lock(self.records / 'repairs.lock'):
            if path.exists():
                existing = json.loads(path.read_text())
                if existing['status'] != 'done':
                    return existing  # A build poll must not reset retries/cooldown.
            stem = Path(target).stem
            variable, stamp = stem.rsplit('_', 1)
            day = datetime.strptime(stamp, '%Y%m%d').date()
            item = {'version': 1, 'product': self.product, 'relative_target': relative,
                    'date': str(day), 'variable': variable, 'reason': str(reason),
                    'observed_signature': signature(target), 'generation': uuid4().hex,
                    'status': 'pending', 'attempts': 0, 'next_attempt_at': None,
                    'created_at': datetime.now(timezone.utc).isoformat()}
            atomic_json(path, item)
            return item

    def items(self):
        if not self.directory.exists():
            return []
        with file_lock(self.records / 'repairs.lock'):
            return [json.loads(path.read_text()) for path in sorted(self.directory.glob('*.json'))]

    def active(self, target):
        path = self.item_path(target)
        if not path.exists():
            return False
        with file_lock(self.records / 'repairs.lock'):
            return json.loads(path.read_text())['status'] != 'done'

    def resolve(self, target):
        path = self.item_path(target)
        if not path.exists():
            return
        with file_lock(self.records / 'repairs.lock'):
            item = json.loads(path.read_text())
            item.update(status='done', last_error=None,
                        updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(path, item)

    def claim(self, limit=4, *, dry_run=False, day=None, start=None, end=None, retry_blocked=False):
        now = datetime.now(timezone.utc)
        claimed = []
        if not self.directory.exists():
            return claimed
        with file_lock(self.records / 'repairs.lock'):
            entries = [(path, json.loads(path.read_text())) for path in self.directory.glob('*.json')]
            entries.sort(key=lambda entry: (entry[1]['date'], entry[1]['created_at']))
            for path, item in entries:
                if item['product'] != self.product or item['status'] == 'done':
                    continue
                if day is not None and item['date'] != str(day):
                    continue
                if start is not None and item['date'] < str(start):
                    continue
                if end is not None and item['date'] > str(end):
                    continue
                if item['status'] == 'blocked':
                    if not retry_blocked:
                        continue
                    item['attempts'] = 0
                elif item.get('next_attempt_at') and datetime.fromisoformat(item['next_attempt_at']) > now:
                    continue
                if item['attempts'] >= self.max_attempts:
                    if not dry_run:
                        item['status'] = 'blocked'
                        atomic_json(path, item)
                    continue
                if not dry_run:
                    item['attempts'] += 1
                    item['status'] = 'processing'
                    item['next_attempt_at'] = (now + timedelta(hours=self.retry_hours)).isoformat()
                    atomic_json(path, item)
                claimed.append(item)
                if len(claimed) >= limit:
                    break
        return claimed

    def finish(self, item, *, success, error=None):
        target = self.raw_root / item['relative_target']
        path = self.item_path(target)
        with file_lock(self.records / 'repairs.lock'):
            current = json.loads(path.read_text())
            if current['generation'] != item['generation']:
                raise RuntimeError('Repair generation changed during download')
            current['status'] = ('done' if success else
                                 'blocked' if current['attempts'] >= self.max_attempts else 'pending')
            current['last_error'] = str(error) if error else None
            current['updated_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(path, current)


def add_repair_arguments(parser):
    parser.add_argument('--check-start', type=date.fromisoformat, help='Audit historical files from YYYY-MM-DD')
    parser.add_argument('--check-end', type=date.fromisoformat, help='Inclusive audit end (default: check-start)')
    parser.add_argument('--repair', action='store_true', help='Repair audit failures, or only process queued repairs')
    parser.add_argument('--retry-blocked', action='store_true', help='Explicitly reset blocked repairs; requires --repair')
    parser.add_argument('--max-repair-files', type=int, default=4)
    parser.add_argument('--repair-max-attempts', type=int, default=3)
    parser.add_argument('--repair-retry-hours', type=float, default=6)


def prepare_repairs(args, parser, *, raw_root, records, product, tasks_for_day, runner_class):
    """Audit if requested, then service historical repairs before normal catch-up.

    Returns (queue, audit_or_repair_only, failures). No watermark is changed.
    tasks_for_day and runner_class belong to the daily wrapper, not the build.
    """
    if (args.check_end and not args.check_start) or (args.retry_blocked and not args.repair):
        parser.error('--check-end requires --check-start; --retry-blocked requires --repair')
    if args.max_repair_files < 1 or args.repair_max_attempts < 1 or args.repair_retry_hours < 0:
        parser.error('Repair limits must be positive, retry hours non-negative')
    if args.check_start and (args.check_end or args.check_start) < args.check_start:
        parser.error('--check-end must not precede --check-start')
    queue = RepairQueue(records, raw_root, product, max_attempts=args.repair_max_attempts,
                        retry_hours=args.repair_retry_hours)
    preview = args.dry_run or args.plan_only
    issues = 0
    if args.check_start:
        selected = args.check_start
        while selected <= (args.check_end or args.check_start):
            for task in tasks_for_day(selected):
                try:
                    validate_task(task, task['target'])
                    print(f'CHECK OK {task["target"]}', flush=True)
                except Exception as exc:
                    issues += 1
                    print(f'CHECK FAILED {task["target"]}: {exc}', flush=True)
                    if args.repair and not preview:
                        queue.enqueue(task['target'], exc)
            selected += timedelta(days=1)
        if not args.repair:
            return queue, True, issues
    claimed = queue.claim(args.max_repair_files, dry_run=preview,
                          day=args.date if not args.check_start else None,
                          start=args.check_start, end=args.check_end or args.check_start,
                          retry_blocked=args.retry_blocked)
    selected_tasks = []
    failures = 0
    for item in claimed:
        try:
            candidates = tasks_for_day(date.fromisoformat(item['date']))
            task = next(t for t in candidates if queue.relative(t['target']) == item['relative_target']
                        and t['request']['variable'] == [item['variable']])
            selected_tasks.append((item, task))
        except Exception as exc:
            failures += 1
            if not preview:
                queue.finish(item, success=False, error=f'Invalid repair request: {exc}')
    if selected_tasks:
        print(f'REPAIRS: {len(selected_tasks)} variable files selected', flush=True)
        runner = runner_class([t for _, t in selected_tasks], Path(records) / 'status.jsonl',
                              max_workers=1, continue_on_error=True, resume=True,
                              validator=validate_task, repair_queue=queue, repairing=True)
        try:
            runner.run(dry_run=preview)
        except Exception as exc:
            print(f'REPAIR RUN FAILED: {exc}', flush=True)
        if not preview:
            for item, task in selected_tasks:
                try:
                    validate_task(task, task['target'])
                    queue.finish(item, success=True)
                    print(f'REPAIR DONE {task["target"]}', flush=True)
                except Exception as exc:
                    failures += 1
                    queue.finish(item, success=False, error=exc)
                    print(f'REPAIR FAILED {task["target"]}: {exc}', flush=True)
    active = [item for item in queue.items() if item['status'] != 'done']
    if active:
        print(f'REPAIRS WAIT: {len(active)} pending/blocked; inspect {queue.directory}', flush=True)
    return queue, bool(args.check_start or args.repair), failures + len(active) + (issues if preview else 0)
