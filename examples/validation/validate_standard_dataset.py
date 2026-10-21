"""Strict local validation CLI. Run with python -m from the repository root."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import sys
from uuid import uuid4

from pipelines.dataset import validate_standard_dataset
from datakit.validation import to_jsonable
from ._standard_config import load_profile, read_json
from ._standard_reports import report_exit_code, write_reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按独立产品标准严格校验完整数据集；输入只读")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile", type=Path)
    selection.add_argument("--inventory", type=Path)
    parser.add_argument(
        "--data-root", type=Path, help="批量路径根目录，profile directory_name 定位各套数据"
    )
    parser.add_argument("--dataset-path", type=Path, help="单套路径覆盖，仅用于 --profile")
    parser.add_argument("--auxiliary-directory", type=Path)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--sample-blocks", type=int, default=3)
    parser.add_argument("--max-blocks", type=int, help="诊断限量；full 会明确报告未完成")
    parser.add_argument("--memory-budget-mib", type=int, default=2048)
    parser.add_argument(
        "--output", type=Path, required=True, help="报告目录，必须位于所有输入数据目录之外"
    )
    args = parser.parse_args(argv)
    try:
        if args.inventory and (args.dataset_path or args.auxiliary_directory):
            raise ValueError("Single-dataset path overrides cannot be used with inventory")
        profiles = (
            [args.profile]
            if args.profile
            else [args.inventory.parent / entry for entry in read_json(args.inventory)["profiles"]]
        )
        if not profiles:
            raise ValueError("Inventory must contain at least one profile")
        configurations = []
        for profile in profiles:
            profile_digest = hashlib.sha256(profile.read_bytes()).hexdigest()
            spec, doc = load_profile(profile)
            if hashlib.sha256(profile.read_bytes()).hexdigest() != profile_digest:
                raise ValueError("Profile changed while being loaded")
            path = args.dataset_path or (
                args.data_root / spec.directory_name
                if args.data_root and spec.directory_name
                else Path(doc["dataset_path"])
            )
            auxiliary = args.auxiliary_directory or (
                path
                if args.data_root or args.dataset_path
                else Path(doc.get("auxiliary_directory") or path)
            )
            for source in (path, auxiliary):
                if args.output.resolve().is_relative_to(source.resolve()):
                    raise ValueError(
                        "Report output must be outside every input dataset/auxiliary directory"
                    )
            configurations.append((profile, spec, path, auxiliary, doc, profile_digest))
        args.output.mkdir(parents=True, exist_ok=True)
        batch_run_id = uuid4().hex
        interrupted = False

        def interrupt(_signal, _frame):
            nonlocal interrupted
            interrupted = True

        previous = signal.signal(signal.SIGINT, interrupt)
        reports = []
        try:
            for number, (profile, spec, path, auxiliary, document, profile_digest) in enumerate(
                configurations
            ):
                if interrupted:
                    break
                ledger = args.output / f"{number+1:02d}-{batch_run_id}-blocks.jsonl"
                ledger_digest = hashlib.sha256()
                with ledger.open("w", encoding="utf-8", newline="\n") as stream:

                    def record(entry):
                        line = (
                            json.dumps(to_jsonable(entry), ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                        stream.write(line)
                        stream.flush()
                        ledger_digest.update(line.encode("utf-8"))
                        if entry["index"][0] % 100 == 0:
                            print(
                                f"[{spec.dataset_id}] logical block {entry['index'][0]}: {entry['read']}",
                                flush=True,
                            )

                    report = validate_standard_dataset(
                        path,
                        spec,
                        mode=args.mode,
                        auxiliary_directory=auxiliary,
                        sample_blocks=args.sample_blocks,
                        max_blocks=args.max_blocks,
                        memory_budget_mib=args.memory_budget_mib,
                        stop_requested=lambda: interrupted,
                        on_block=record,
                    )
                report.identity.update(
                    profile_path=str(profile),
                    profile_sha256=profile_digest,
                    block_ledger=str(ledger),
                    block_ledger_sha256=ledger_digest.hexdigest(),
                    batch_run_id=batch_run_id,
                )
                report.require(
                    "publication.profile_unchanged",
                    hashlib.sha256(profile.read_bytes()).hexdigest() == profile_digest,
                    "standard_conformance",
                    profile_digest,
                    hashlib.sha256(profile.read_bytes()).hexdigest(),
                )
                report.add(
                    "publication.profile_approval",
                    (
                        "pass"
                        if document.get("profile_status", "approved") == "approved"
                        else "needs-review"
                    ),
                    "standard_conformance",
                    evidence="Profile approval state; templates require independent rule/source confirmation",
                )
                reports.append(report.to_dict())
                write_reports(args.output, reports)
                print(
                    f"[{spec.dataset_id}] scope={spec.validation_scope} quick_passed={report.quick_passed} scan_complete={report.coverage['scan_complete']} release_passed={report.release_passed}",
                    flush=True,
                )
        finally:
            signal.signal(signal.SIGINT, previous)
        (args.output / "batch.json").write_text(
            json.dumps(
                {
                    "requested_profiles": len(configurations),
                    "completed_profiles": len(reports),
                    "interrupted": interrupted,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 2 if interrupted else report_exit_code(reports)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Configuration/output error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
