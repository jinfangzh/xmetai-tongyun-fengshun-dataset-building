"""Report writers used by the runnable examples, outside library validation."""

from __future__ import annotations

import csv
import shutil
import json
from pathlib import Path
from typing import Any


def write_reports(output: Path, reports: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    markdown = [
        "# 标准数据集校验报告",
        "",
        "quick_passed 表示预定抽样及全部适用 quick 条款通过；warning 不阻断。scan_complete/release_passed 在 quick 中始终 false。full 及生产冻结证明单独记录，不能把未执行项描述为失败或已通过。",
        "",
        "| 数据集 | 模式 | 标准 | 完整性 | 质量 | 消费兼容 | 已检/预期块 | quick通过 | 警告数 | 扫描完成 | 发布 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for position, report in enumerate(reports):
        stem = f"{position+1:02d}-{report['run_id']}"
        (output / f"{stem}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        c = report["coverage"]
        row = {
            k: report[k]
            for k in (
                "dataset_id",
                "requested_mode",
                "standard_conformance",
                "data_integrity",
                "data_quality",
                "consumer_compatibility",
                "conversion_fidelity",
                "quick_passed",
                "release_passed",
            )
        }
        row.update(
            validation_scope=report.get("validation_policy", "strict-standard"),
            quick_scan_complete=c.get("quick_scan_complete", False),
            publication_readiness=report.get("publication_readiness", "incomplete"),
            quick_blocking_count=len(report.get("quick_blocking_checks", [])),
            warning_count=len(report.get("warnings", [])),
            scan_complete=c.get("scan_complete", False),
            value_checked=c.get("value_checked", 0),
            expected_blocks=c.get("logical_blocks_expected", 0),
        )
        rows.append(row)
        markdown.append(
            f"| {report['dataset_id']} | {report['requested_mode']} | {report['standard_conformance']} | {report['data_integrity']} | {report['data_quality']} | {report['consumer_compatibility']} | {row['value_checked']}/{row['expected_blocks']} | {report['quick_passed']} | {row['warning_count']} | {row['scan_complete']} | {report['release_status']} |"
        )
        details.extend(
            ["", f"## {report['dataset_id']}", "", "| 检查项 | 状态 | 依据/异常 |", "|---|---|---|"]
        )
        for check in report["checks"]:
            if check["status"] in ("pass", "not-applicable"):
                continue
            detail = (
                str(check.get("evidence") or check.get("observed") or "")
                .replace("|", "\\|")
                .replace("\n", " ")
            )
            details.append(f"| {check['check_id']} | {check['status']} | {detail} |")
        details.append("")
    if rows:
        with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.md").write_text("\n".join(markdown + details), encoding="utf-8")
    batch_run_id = reports[0].get("identity", {}).get("batch_run_id") if reports else None
    if batch_run_id:
        for extension in ("csv", "md"):
            shutil.copyfile(
                output / f"summary.{extension}", output / f"summary-{batch_run_id}.{extension}"
            )


def report_exit_code(reports: list[dict[str, Any]]) -> int:
    if any(any(c["status"] == "fail" for c in r["checks"]) for r in reports):
        return 1
    if any(
        not (r["release_passed"] if r["requested_mode"] == "full" else r["quick_passed"])
        for r in reports
    ):
        return 2
    return 0
