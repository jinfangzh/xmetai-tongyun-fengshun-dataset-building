"""Consume CDS completed-file records and build each ready day once.

Run periodically beside the existing downloader. Dry-run is the default;
--write creates new independent products through example_build_tongyun_daily.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from datakit import io
from examples.dataset_building import example_build_tongyun_daily as daily


def completed_paths(status_path):
    """Use the downloader's completed records; tolerate a concurrently appended tail."""
    if not status_path.exists():
        return set()
    lines = status_path.read_text(encoding="utf-8").splitlines()
    completed = set()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                break
            raise ValueError(f"Invalid download status record: {status_path}:{number}")
        if record.get("status") == "completed":
            completed.add(Path(record["target"]).expanduser().resolve())
    return completed


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    names = ("source", "download_status", "output_root", "report_dir",
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


def ready_days(config, completed):
    description = io.get_description(daily.SOURCE_CONFIG)
    # This variable is present in every complete UTC day; file existence alone
    # is insufficient, so discover candidates from completed status records.
    anchor = config["source"] / "pl_h6" / "temperature"
    candidates = set()
    for path in completed:
        if path.is_relative_to(anchor) and path.name.startswith("temperature_"):
            try:
                candidates.add(datetime.strptime(path.stem, "temperature_%Y%m%d").date())
            except ValueError:
                continue
    for day in sorted(candidates):
        try:
            files = daily.required_files(config["source"], day, description)
        except FileNotFoundError:
            print(f"WAIT {day}: missing daily/boundary files", flush=True)
            continue
        if not all(original.resolve() in completed for original, _ in files):
            print(f"WAIT {day}: waiting for all 23 completed download records", flush=True)
            continue
        yield day


def run(config, *, write=False, max_days=1):
    completed = completed_paths(config["download_status"])
    failures, attempted = 0, 0
    for day in ready_days(config, completed):
        identity = f"era5.tongyun.{day:%Y%m%d}.c78.p25.h6.v1"
        destination = config["output_root"] / identity
        marker = config["output_root"] / "_completed" / f"{day:%Y%m%d}.json"
        if marker.exists() and destination.is_dir():
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
            result = daily.main(options)
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
