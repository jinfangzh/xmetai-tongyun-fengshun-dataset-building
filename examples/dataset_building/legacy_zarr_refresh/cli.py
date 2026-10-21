"""Command-line entry point for the self-contained legacy Zarr refresh example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from legacy_zarr_refresh import (  # noqa: E402
    LegacyZarrSource,
    build_refresh_plan,
    load_profiles,
    migrate_auxiliary_files,
    read_manifest,
    refresh_legacy_zarr,
    select_profiles,
    validate_refreshed_store,
)
from legacy_zarr_refresh.pipeline import (  # noqa: E402
    output_path,
    selection_indices,
)

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Refresh legacy meteorological Zarr stores"
    )
    result.add_argument("--config", type=Path, required=True)
    sub = result.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="read-only source inventory")
    inspect.add_argument("--dataset", action="append", default=[])
    inspect.add_argument("--report", type=Path)

    for name, help_text in (
        ("pilot", "write a small contiguous test slice and validate it"),
        ("convert", "refresh a complete dataset"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--dataset", required=True)
        command.add_argument("--output-root", type=Path, required=True)
        if name == "pilot":
            command.add_argument("--start-index", type=int, default=0)
            command.add_argument("--count", type=int, default=4)
        command.add_argument(
            "--dry-run",
            action="store_true",
            help="show the selected data and transformations without writing",
        )
        command.add_argument("--workers", type=int, default=1)
        command.add_argument("--no-resume", action="store_true")
        command.add_argument("--overwrite", action="store_true")
        command.add_argument("--full-data-check", action="store_true")

    validate = sub.add_parser("validate", help="reopen and compare an existing output")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--zarr", type=Path, required=True)
    validate.add_argument("--full-data-check", action="store_true")

    auxiliary = sub.add_parser(
        "auxiliary", help="migrate const, weight and mask sidecar files"
    )
    auxiliary.add_argument("--dataset", action="append", default=[])
    auxiliary.add_argument("--output-root", type=Path, required=True)
    auxiliary.add_argument(
        "--target-name",
        help="completed output directory name; defaults to the profile output_name",
    )
    auxiliary.add_argument("--dry-run", action="store_true")
    auxiliary.add_argument(
        "--refresh-embedded-mask",
        action="store_true",
        help="replace the Zarr mask from const.nc/lsm after semantic verification",
    )
    auxiliary.add_argument("--overwrite", action="store_true")
    auxiliary.add_argument("--strict", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    profiles = load_profiles(args.config)

    if args.command == "inspect":
        selected = select_profiles(profiles, args.dataset or ["all"])
        report = {
            profile.key: LegacyZarrSource(profile).describe() for profile in selected
        }
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        return

    if args.command == "auxiliary":
        selected = select_profiles(profiles, args.dataset or ["all"])
        reports = [
            migrate_auxiliary_files(
                profile.key,
                {
                    "source": str(profile.input),
                    "outputs": [args.target_name or profile.output_name],
                    "refresh_embedded_mask": args.refresh_embedded_mask,
                },
                args.output_root,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                strict=args.strict,
            )
            for profile in selected
        ]
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    profile = profiles[args.dataset]
    if args.command in {"pilot", "convert"}:
        source = LegacyZarrSource(profile)
        count = args.count if args.command == "pilot" else None
        start = args.start_index if args.command == "pilot" else 0
        indices = selection_indices(len(source.times), start, count)
        destination = output_path(
            profile,
            args.output_root,
            pilot=args.command == "pilot",
            start=start,
            count=len(indices),
        )
        if args.dry_run:
            print(
                json.dumps(
                    build_refresh_plan(source, destination, indices),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        path, manifest = refresh_legacy_zarr(
            profile,
            args.output_root,
            start=start,
            count=count,
            pilot=args.command == "pilot",
            workers=args.workers,
            resume=not args.no_resume,
            overwrite=args.overwrite,
            full_data_check=args.full_data_check,
        )
        print(
            json.dumps(
                {"output": str(path), "validation": manifest["validation"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    manifest = read_manifest(args.zarr)
    source_range = manifest["source_indices"]
    indices = list(range(int(source_range["start"]), int(source_range["end"]) + 1))
    report = validate_refreshed_store(
        args.zarr,
        LegacyZarrSource(profile),
        indices,
        full_data_check=args.full_data_check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
