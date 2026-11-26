"""Preflight daily raw inputs and enqueue precise repairs, never call CDS."""

from datetime import datetime, time, timedelta
from pathlib import Path

from datakit import io
from datakit.runtime.daily_integrity import LEVELS, RepairQueue, open_netcdf, validate_dataset
from datakit.schema import load_mapping, load_schema
from datakit.schema.semantics import TemporalKind


def queue_for(config, builder):
    records = config['download_watermark'].parent
    if Path(config['repair_queue']) != records / 'repairs':
        raise ValueError('repair_queue must be download_watermark.parent/repairs')
    return RepairQueue(records, config['source'].parent, builder.DOWNLOAD_PRODUCT)


def inspect_day(config, day, builder, *, enqueue=False):
    """Check the requested day and Tongyun's preceding accumulation files.

    Existing READY products are excluded by the caller. Only raw-file integrity
    failures are queued; unrelated pipeline/statistics/normalization failures
    are never interpreted as a reason to redownload data.
    """
    description = io.get_description(builder.SOURCE_CONFIG)
    mapping = load_mapping(description)
    spec, _ = load_schema(builder.TARGET_SPEC)
    lat, _ = spec.variant.grid.coordinates()
    grid = float(abs(lat[1] - lat[0]))
    queue = queue_for(config, builder)
    issues = []
    for variable, template in description['layout']['files'].items():
        declaration = mapping.variables[variable]
        accumulated = declaration.temporal_kind == TemporalKind.INTERVAL_ACCUMULATION
        tongyun = builder.DOWNLOAD_PRODUCT == 'tongyun_c78'
        days = [day - timedelta(days=1), day] if tongyun and accumulated else [day]
        for selected in days:
            relative = Path(template.format(day=datetime.combine(selected, time())))
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Source template escapes its root')
            path = config['source'] / relative
            hours = range(24) if tongyun and accumulated else range(0, 24, 6) if tongyun else [0]
            times = [f'{selected}T{hour:02d}:00' for hour in hours]
            try:
                with open_netcdf(path) as dataset:
                    validate_dataset(dataset, variable=path.stem.rsplit('_', 1)[0],
                                     source_name=declaration.source, times=times, grid=grid,
                                     levels=LEVELS if variable in {'z', 't', 'u', 'v', 'q'} else None)
            except Exception as exc:
                issues.append((path, str(exc)))
                print(f'SOURCE INVALID {path}: {exc}', flush=True)
                if enqueue:
                    queue.enqueue(path, exc)
    return issues
