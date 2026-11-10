"""Consume the CDS download watermark and build each ready day once.

Run periodically beside the existing downloader. Dry-run is the default;
--write creates new independent products through example_build_tongyun_daily.
Fengshun reuses discovery and publication with its own builder module.
"""

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from datakit import io
from examples.dataset_building import example_build_tongyun_daily as daily


def read_watermark(path):
    """A missing watermark means no day has been confirmed complete yet."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("latest_complete_day") if isinstance(payload, dict) else None
        if not isinstance(value, str):
            raise ValueError("latest_complete_day must be a YYYY-MM-DD string")
        result = date.fromisoformat(value)
        if result.isoformat() != value:
            raise ValueError("latest_complete_day must use YYYY-MM-DD")
        return result
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid download watermark {path}: {exc}") from exc


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    names = ("source", "download_watermark", "output_root", "report_dir",
             "mean_nc", "std_nc", "mask_nc", "const_nc", "weight_npy")
    if not isinstance(config, dict) or any(not isinstance(config.get(n), str) for n in names):
        raise ValueError(f"Config requires path strings: {', '.join(names)}")
    for name in names:
        value = Path(config[name]).expanduser()
        config[name] = (value if value.is_absolute() else path.parent / value).resolve()
    source, output = config["source"], config["output_root"]
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("source and output_root must not overlap")
    return config


def product_paths(config, day, *, builder=daily):
    identity = builder.IDENTITY_TEMPLATE.format(day=day)
    return (identity, config["output_root"] / identity,
            config["output_root"] / "_completed" / f"{day:%Y%m%d}.json")


def ready_days(config, latest, *, builder=daily):
    description = io.get_description(builder.SOURCE_CONFIG)
    # Find local candidate dates, but never process beyond the download watermark.
    candidates = set()
    for path in config["source"].glob(builder.CANDIDATE_GLOB):
        try:
            candidate = datetime.strptime(path.stem, "temperature_%Y%m%d").date()
        except ValueError:
            continue
        if candidate <= latest:
            candidates.add(candidate)
    for day in sorted(candidates):
        _, destination, marker = product_paths(config, day, builder=builder)
        if marker.is_file() and destination.is_dir():
            print(f"SKIP {day}: already built", flush=True)
            continue
        try:
            builder.required_files(config["source"], day, description)
        except FileNotFoundError:
            print(f"WAIT {day}: missing required source files", flush=True)
            continue
        yield day


def run(config, *, write=False, max_days=1, builder=daily):
    latest = read_watermark(config["download_watermark"])
    if latest is None:
        print(f"WAIT: no completed download day; watermark missing: {config['download_watermark']}", flush=True)
        return 0
    print(f"Download watermark: {latest}", flush=True)
    failures, attempted = 0, 0
    for day in ready_days(config, latest, builder=builder):
        identity, destination, marker = product_paths(config, day, builder=builder)
        # Recheck at execution time in case a product was published since discovery.
        if marker.is_file() and destination.is_dir():
            print(f"SKIP {day}: already built", flush=True)
            continue
        if destination.exists() or destination.is_symlink():
            print(f"ERROR {day}: output exists without a completion marker: {destination}", flush=True)
            failures += 1
            continue
        if attempted >= max_days:
            break
        # Isolate each attempt. Failed products remain for diagnosis; the next
        # cron invocation uses a new attempt and never overwrites a partial store.
        attempt = config["output_root"] / "_attempts" / uuid4().hex / identity
        options = ["--date", day.isoformat(), "--source", str(config["source"]),
                   "--destination", str(attempt if write else destination),
                   "--dataset-id", identity, "--report-dir", str(config["report_dir"]),
                   "--write" if write else "--dry-run"]
        for name in ("mean_nc", "std_nc", "mask_nc", "const_nc", "weight_npy"):
            options.extend(["--" + name.replace("_", "-"), str(config[name])])
        print(f"BUILD {day}: {'write' if write else 'dry-run'} -> {destination}", flush=True)
        attempted += 1
        try:
            result = builder.main(options)
            if result != 0:
                failures += 1
                print(f"FAILED {day}: inspect reports; retry on next run", flush=True)
                continue
            if write:
                if not attempt.is_dir():
                    raise RuntimeError(f"Successful build did not create {attempt}")
                # The builder returns zero only after write and Quick validation.
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(destination)
                attempt.rename(destination)
                marker.parent.mkdir(parents=True, exist_ok=True)
                payload = {"date": day.isoformat(), "destination": str(destination),
                           "source": str(config["source"]),
                           "completed_at": datetime.now(timezone.utc).isoformat()}
                temporary = marker.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                temporary.replace(marker)
                print(f"READY {day}: {destination}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"FAILED {day}: {exc}; inspect reports and attempt directory", flush=True)
    if not attempted:
        print("No new ready days to build", flush=True)
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-days", type=int, default=1, help="Build at most N dates per run")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--dry-run", dest="write", action="store_false")
    parser.set_defaults(write=False)
    args = parser.parse_args(argv)
    if args.max_days < 1:
        parser.error("--max-days must be positive")
    return run(load_config(args.config), write=args.write, max_days=args.max_days)


if __name__ == "__main__":
    raise SystemExit(main())
