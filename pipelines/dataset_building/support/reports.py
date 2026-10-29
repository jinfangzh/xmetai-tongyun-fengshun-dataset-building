"""Result models, completion summaries, output protection and report persistence."""

from __future__ import annotations

from collections.abc import Sequence
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert report values to JSON-safe objects without emitting nonfinite numbers."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: str
    category: str
    expected: Any = None
    observed: Any = None
    evidence: Any = None

    def __post_init__(self) -> None:
        if self.status not in (
            "pass",
            "fail",
            "needs-review",
            "not-applicable",
            "not-run",
            "blocked",
            "warning",
        ):
            raise ValueError(f"Unknown check status: {self.status}")


@dataclass
class ValidationReport:
    dataset_id: str
    mode: str
    identity: dict[str, Any]
    checks: list[CheckResult] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add(
        self,
        check_id: str,
        status: str,
        category: str,
        expected: Any = None,
        observed: Any = None,
        evidence: Any = None,
    ) -> None:
        self.checks.append(CheckResult(check_id, status, category, expected, observed, evidence))

    def require(
        self,
        check_id: str,
        condition: bool,
        category: str,
        expected: Any = None,
        observed: Any = None,
        evidence: Any = None,
    ) -> bool:
        self.add(check_id, "pass" if condition else "fail", category, expected, observed, evidence)
        return bool(condition)

    def conclusion(self, category: str) -> str:
        statuses = {c.status for c in self.checks if c.category == category}
        if "fail" in statuses:
            return "fail"
        if "needs-review" in statuses:
            return "needs-review"
        if statuses & {"blocked", "not-run"}:
            return "incomplete"
        if "warning" in statuses:
            return "warning"
        if "pass" in statuses:
            return "pass"
        return "not-applicable" if statuses else "incomplete"

    @property
    def release_passed(self) -> bool:
        return (
            self.mode == "full"
            and self.coverage.get("scan_complete", False)
            and all(c.status in ("pass", "not-applicable", "warning") for c in self.checks)
        )

    @property
    def quick_blocking_checks(self) -> list[str]:
        """Failures in the requested quick scope, excluding future unexecuted gates."""
        future_checks = {
            "scan.complete",
            "publication.frozen_snapshot",
            "conversion_fidelity.external_evidence",
        }
        return [
            check.check_id
            for check in self.checks
            if check.status not in ("pass", "not-applicable", "warning")
            and not (check.status == "not-run" and check.check_id in future_checks)
        ]

    @property
    def quick_passed(self) -> bool:
        """All selected samples completed and no applicable quick checks blocked."""
        return (
            self.mode == "quick"
            and self.coverage.get("quick_scan_complete", False)
            and not self.quick_blocking_checks
        )

    def to_dict(self) -> dict[str, Any]:
        categories = (
            "standard_conformance",
            "data_integrity",
            "data_quality",
            "conversion_fidelity",
            "publication_readiness",
            "full_scan",
        )
        conclusions = {name: self.conclusion(name) for name in categories}
        return to_jsonable(
            {
                "dataset_id": self.dataset_id,
                "validation_policy": self.identity.get("validation_scope", "strict-standard"),
                "requested_mode": self.mode,
                "identity": self.identity,
                "run_id": self.run_id,
                "started_at_utc": self.started_at_utc,
                "checks": [
                    {
                        **asdict(c),
                        "dataset_id": self.dataset_id,
                        **self.identity,
                        "run_id": self.run_id,
                        "checked_scope": (
                            self.coverage.get("selection", "metadata/coordinates/auxiliaries")
                            if c.category == "data_quality" and c.check_id.startswith("quality.")
                            else "complete metadata/coordinate/declared auxiliary scope or explicit external receipt"
                        ),
                        "duration": None,
                    }
                    for c in self.checks
                ],
                "coverage": self.coverage,
                "quality": self.quality,
                **conclusions,
                "quick_passed": self.quick_passed,
                "quick_blocking_checks": self.quick_blocking_checks,
                "release_passed": self.release_passed,
                "release_status": "passed" if self.release_passed else "not-approved",
                "blocking_checks": [
                    c.check_id
                    for c in self.checks
                    if c.status not in ("pass", "not-applicable", "warning")
                ],
                "warnings": [asdict(c) for c in self.checks if c.status == "warning"],
                "value_scan_complete": self.coverage.get("value_scan_complete", False),
                "storage_audit_complete": self.coverage.get("storage_audit_complete", False),
                "scan_complete": self.coverage.get("scan_complete", False),
            }
        )


def ensure_report_destination(
    output: Path,
    *,
    directories: Sequence[Path],
    files: Sequence[Path | None] = (),
    directory: bool = False,
) -> None:
    target = output.resolve()
    if any(target.is_relative_to(source.resolve()) for source in directories):
        raise ValueError("Report output must be outside every input dataset/auxiliary directory")
    for source in files:
        if source is not None and (
            target == source.resolve()
            or directory and source.resolve().is_relative_to(target)
        ):
            raise ValueError("Report output must not overwrite input configuration or evidence files")


def write_reports(output: Path, reports: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    markdown = [
        "# 标准数据集校验报告",
        "",
        "quick_passed 表示预定抽样及全部适用 quick 条款通过；warning 不阻断。scan_complete/release_passed 在 quick 中始终 false。full 及生产冻结证明单独记录，不能把未执行项描述为失败或已通过。",
        "",
        "| 数据集 | 模式 | 标准 | 完整性 | 质量 | 已检/预期块 | quick通过 | 警告数 | 扫描完成 | 发布 |",
        "|---|---|---|---|---|---|---|---|---|---|",
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
            f"| {report['dataset_id']} | {report['requested_mode']} | {report['standard_conformance']} | {report['data_integrity']} | {report['data_quality']} | {row['value_checked']}/{row['expected_blocks']} | {report['quick_passed']} | {row['warning_count']} | {row['scan_complete']} | {report['release_status']} |"
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
