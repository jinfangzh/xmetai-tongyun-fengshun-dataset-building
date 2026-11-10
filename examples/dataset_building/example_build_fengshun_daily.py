"""Build normalized c76 / 1.5 degree daily products from downloaded CDS ZIPs.

Daily means and sums are already computed by CDS. Read exactly one day,
validate its native grid, pack channels, then reuse ERA5 unit transforms with
a 24-hour radiation window. No hourly accumulation or resampling is performed.
"""

from datetime import date, datetime, time
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

from datakit import io
from datakit.schema import load_mapping
from examples.dataset_building import example_build_era5 as era5
from pipelines.dataset import build_zarr
from pipelines.dataset.preparation import pack_source_channels
from pipelines.dataset.reports import BuildReport, print_build_summary, save_build_report


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / "config/data_sources/era5_cds_fengshun_daily_raw.yaml"
TARGET_SPEC = ROOT / "config/dataset_specs/s2s_c76_1p5_h24.yaml"
IDENTITY_TEMPLATE = "s2s.{day:%Y%m%d}.c76.1p5.h24.v1"
CANDIDATE_GLOB = "pl_mean/temperature/*/temperature_*.zip"


def required_files(source, day, description):
    """Require the 16 source ZIPs for this UTC day, excluding .zip.part."""
    source = Path(source).expanduser().resolve()
    required = []
    for template in description["layout"]["files"].values():
        relative = Path(template.format(day=datetime.combine(day, time())))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Source template must stay under its root: {relative}")
        required.append((source / relative, relative))
    missing = [str(path) for path, _ in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Fengshun daily statistics are incomplete. Missing:\n" + "\n".join(missing))
    return required


def read_source(source_path, description, *, spec):
    """Validate each variable before merging; reject wrong times and grids."""
    raw = era5.read_source(source_path, description, spec=spec)
    try:
        mapping = load_mapping(description)
        lat, lon = spec.variant.grid.coordinates()
        fields = []
        for variable, declaration in mapping.variables.items():
            value = raw[declaration.source].rename({f"time_{variable}": "time"})
            # The source is now the service response, rather than a prepared NC.
            # Normalize coordinate representation without interpolating values.
            value = value.assign_coords(lon=value.lon % 360).sortby("lon")
            value = value.sortby("lat", ascending=False)
            if not np.array_equal(value.time.values.astype("datetime64[ns]"), spec.expected_times()):
                raise ValueError(f"{variable}: expected exactly the requested UTC daily timestamp")
            # pack_source_channels can reindex missing coordinates to NaN; here
            # the downloader promises a complete native grid, so reject gaps.
            if (not np.array_equal(value.lat.values, lat)
                    or not np.array_equal(value.lon.values, lon)):
                raise ValueError(f"{variable}: expected the complete native 1.5 degree grid")
            fields.append(value.to_dataset(name=declaration.source))
        physical = xr.merge(fields, join="exact", compat="no_conflicts", combine_attrs="override")
        packed = pack_source_channels(physical, spec, mapping).rename("data").to_dataset()
        packed.attrs["accumulation_window_hours"] = 24
        packed.set_close(raw.close)
        return packed
    except BaseException:
        raw.close()
        raise


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date.fromisoformat, help="UTC YYYY-MM-DD")
    parser.add_argument("--source", required=True, type=Path, help="Downloader raw/cds_daily_1p5 ZIP root")
    parser.add_argument("--destination", required=True, type=Path, help="New daily Zarr path")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", help="Default: s2s.YYYYMMDD.c76.1p5.h24.v1")
    for option in ("mean-nc", "std-nc", "mask-nc", "const-nc", "weight-npy"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="write", action="store_false")
    mode.add_argument("--write", dest="write", action="store_true")
    parser.set_defaults(write=False)
    args = parser.parse_args(argv)
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if (destination == source or destination.is_relative_to(source)
            or source.is_relative_to(destination)):
        parser.error("source and destination paths must not overlap")
    if destination.exists() or destination.is_symlink():
        parser.error(f"destination must not exist: {destination}")
    files = required_files(source, args.date, io.get_description(SOURCE_CONFIG))
    identity = args.dataset_id or IDENTITY_TEMPLATE.format(day=args.date)
    timestamp = f"{args.date.isoformat()}T00:00"
    with TemporaryDirectory(prefix="fengshun-era5-daily-") as directory:
        stage = Path(directory)
        for original, relative in files:
            link = stage / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(original)
        try:
            report = build_zarr(
                source_config=SOURCE_CONFIG, target_spec=TARGET_SPEC,
                source_path=stage, destination=destination, read=read_source,
                process=partial(era5.process, mean_path=args.mean_nc, std_path=args.std_nc,
                                mask_path=args.mask_nc, source_stage="packed", accumulation_hours=24),
                dry_run=not args.write,
                build_overrides={"first_time": timestamp, "last_time": timestamp, "dataset_id": identity},
                auxiliaries={"const.nc": args.const_nc, "weight.npy": args.weight_npy},
            )
        except Exception as exc:
            report = BuildReport(str(destination), dry_run=not args.write)
            report.fail("pipeline", str(exc), exc)
        report_path = save_build_report(report, args.report_dir)
        print_build_summary(report, report_path, processor="fengshun")
        success = (report.coverage["written"] and report.quick_passed if args.write
                   else report.coverage["conformance_passed"])
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
