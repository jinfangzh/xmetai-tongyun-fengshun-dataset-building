"""Build one UTC day from Tongyun's daily CDS download layout.

Reuse example_build_era5's standard normalized c78 processing unchanged.
Only the requested day's files and the previous day's hourly accumulations
are exposed to the build pipeline, using temporary read-only source links.
"""

from datetime import date, datetime, time, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from datakit import io
from datakit.schema import load_mapping
from datakit.schema.semantics import TemporalKind
from examples.dataset_building import example_build_era5


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / "config/data_sources/era5_cds_daily_raw.yaml"
TARGET_SPEC = ROOT / "config/dataset_specs/era5_c78_p25_h6.yaml"


def required_files(source, day, description):
    """Return 18 current-day and five boundary files, rejecting missing inputs."""
    source = Path(source).expanduser().resolve()
    mapping = load_mapping(description)
    required = []
    for variable, template in description["layout"]["files"].items():
        declaration = mapping.variables[variable]
        days = [day]
        if declaration.temporal_kind == TemporalKind.INTERVAL_ACCUMULATION:
            days.insert(0, day - timedelta(days=1))
        for selected_day in days:
            relative = Path(template.format(day=datetime.combine(selected_day, time())))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Source template must stay under its root: {relative}")
            required.append((source / relative, relative))
    missing = [str(path) for path, _ in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Daily ERA5 input is incomplete. 00 UTC requires the previous day's "
            "five hourly accumulation files. Missing:\n" + "\n".join(missing)
        )
    return required


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date.fromisoformat, help="UTC YYYY-MM-DD")
    parser.add_argument("--source", required=True, type=Path, help="Downloader raw/daily root")
    parser.add_argument("--destination", required=True, type=Path, help="New daily Zarr path")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", help="Default: era5.YYYYMMDD.c78.p25.h6.v1")
    # Use the existing c78 product; do not add another product/preprocessing definition.
    for option in ("mean-nc", "std-nc", "mask-nc", "const-nc", "weight-npy"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--content-version")
    parser.add_argument("--data-revision")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="write", action="store_false")
    mode.add_argument("--write", dest="write", action="store_true")
    parser.set_defaults(write=False)
    args = parser.parse_args(argv)

    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    # Check against the real input root before staging hides that relationship.
    if (destination == source or destination.is_relative_to(source)
            or source.is_relative_to(destination)):
        parser.error("source and destination paths must not overlap")
    if destination.exists() or destination.is_symlink():
        parser.error(f"destination must not exist: {destination}")
    files = required_files(source, args.date, io.get_description(SOURCE_CONFIG))
    identity = args.dataset_id or f"era5.{args.date:%Y%m%d}.c78.p25.h6.v1"

    with TemporaryDirectory(prefix="tongyun-era5-daily-") as directory:
        stage = Path(directory)
        for original, relative in files:
            link = stage / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(original)
        options = [
            "--source-config", str(SOURCE_CONFIG), "--target-spec", str(TARGET_SPEC),
            "--source", str(stage), "--destination", str(destination),
            "--start-time", f"{args.date.isoformat()}T00:00",
            "--end-time", f"{args.date.isoformat()}T18:00",
            "--dataset-id", identity, "--report-dir", str(args.report_dir),
            "--source-stage", "raw", "--accumulation-hours", "6",
            "--write" if args.write else "--dry-run",
        ]
        for name in ("mean_nc", "std_nc", "mask_nc", "const_nc", "weight_npy",
                     "content_version", "data_revision"):
            value = getattr(args, name)
            if value is not None:
                options.extend(["--" + name.replace("_", "-"), str(value)])
        return example_build_era5.main(options)


if __name__ == "__main__":
    raise SystemExit(main())
