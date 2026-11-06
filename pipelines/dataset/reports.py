"""Result models, completion summaries, output protection and report persistence."""

from __future__ import annotations

import math
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from datakit.validation.check_result import CheckResult


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


@dataclass
class ValidationReport:
    dataset_id: str
    mode: str
    identity: dict[str, Any]
    checks: list[CheckResult] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add(
        self,
        check_id: str,
        status: str,
        category: str,
        expected: Any = None,
        observed: Any = None,
        evidence: Any = None,
    ) -> None:
        self.checks.append(
            CheckResult(check_id, status, category, expected, observed, evidence)
        )

    def require(
        self,
        check_id: str,
        condition: bool,
        category: str,
        expected: Any = None,
        observed: Any = None,
        evidence: Any = None,
    ) -> bool:
        self.add(
            check_id,
            "pass" if condition else "fail",
            category,
            expected,
            observed,
            evidence,
        )
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
            and all(
                c.status in ("pass", "not-applicable", "warning") for c in self.checks
            )
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
                "validation_policy": self.identity.get(
                    "validation_scope", "strict-standard"
                ),
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
                            self.identity["checked_scope"]
                            if self.mode == "create"
                            else self.coverage.get(
                                "selection", "metadata/coordinates/auxiliaries"
                            )
                            if c.category == "data_quality"
                            and c.check_id.startswith("quality.")
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
                "storage_audit_complete": self.coverage.get(
                    "storage_audit_complete", False
                ),
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
        raise ValueError(
            "Report output must be outside every input dataset/auxiliary directory"
        )
    for source in files:
        if source is not None and (
            target == source.resolve()
            or directory
            and source.resolve().is_relative_to(target)
        ):
            raise ValueError(
                "Report output must not overwrite input configuration or evidence files"
            )


def save_build_report(report, directory) -> Path:
    """Save the complete build report as a new JSON file without terminal output."""
    directory = Path(directory).expanduser().resolve()
    destination = Path(report.destination).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = directory / f"{destination.name}.{stamp}.{uuid4().hex[:8]}.json"
    ensure_report_destination(output, directories=(destination,))
    directory.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(to_jsonable(report.to_dict()), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def print_build_summary(report, report_path, *, processor):
    """Print task conclusions and actionable failures; retain metrics in JSON."""
    labels = {"pass": "通过", "fail": "失败", "needs-review": "待审核", "warning": "有警告",
              "incomplete": "未完成", "not-applicable": "不适用"}
    validation, creation = report.validation_report, report.creation_report
    result = validation if validation is not None else creation
    success = (report.coverage.get("conformance_passed", False) if report.dry_run
               else report.coverage.get("written", False) and report.quick_passed)
    identity = report.coverage.get("target_contract", {}).get("effective", {})
    dataset_id = identity.get("dataset_id") or (result.dataset_id if result else Path(report.destination).name)
    print(f"{processor.upper()} pipeline：{'完成' if success else '失败'}{'（dry-run）' if report.dry_run else ''}")
    print(f"数据集：{dataset_id}")
    print(f"输出：{report.destination}")
    print("写入：" + ("成功" if report.coverage.get("written")
                     else "未执行（dry-run）" if report.dry_run else "未完成"))
    for category, label in (("standard_conformance", "标准符合性"),
                            ("data_integrity", "数据完整性"), ("data_quality", "数据质量")):
        value = result.conclusion(category) if result else None
        print(f"{label}：{labels.get(value, '未执行')}")
    copied = report.stages.get("auxiliary_copy")
    print("附件复制：" + ("成功" if copied == "passed" else "失败" if copied == "failed"
                         else "未执行（dry-run）" if report.dry_run else "未执行"))
    print("Quick校验：" + ("通过" if report.quick_passed else "未通过" if validation else "未执行"))
    print("完整发布验收：" + ("通过" if validation and validation.release_passed else "未完成"))
    checks = result.checks if result else ()
    failed = [check for check in checks if check.status in {"fail", "needs-review", "blocked"}]
    for failure in report.failures:
        if failed and failure["stage"] in {"validate", "input_check", "write", "create"}:
            continue
        print(f"  失败阶段：{failure['stage']}；原因：{str(failure['message'])[:600]}")
    for check in failed[:10]:
        print(f"  {labels.get(check.status, check.status)}：{check.check_id}")
        evidence = check.evidence
        message = evidence.get("message") if isinstance(evidence, dict) else evidence
        if message:
            print(f"    原因：{str(message)[:600]}")
    if len(failed) > 10:
        print(f"  另有 {len(failed) - 10} 项未通过，详情见完整报告")
    pending = [check.check_id for check in checks if check.status == "not-run"]
    if pending:
        print("待完成项：" + "、".join(pending[:10]) + ("等" if len(pending) > 10 else ""))
    missing = report.coverage.get("missing_auxiliary_inputs", [])
    if missing:
        print("待提供附件：" + "、".join(missing))
    warnings = sum(check.status == "warning" for check in checks)
    if warnings:
        print(f"警告：{warnings}项，详情见完整报告")
    print(f"完整报告：{report_path}")


@dataclass
class BuildReport:
    """Build result; retain both creation and read-back diagnostics."""

    destination: str
    dry_run: bool = False
    stages: dict[str, str] = field(default_factory=lambda: dict.fromkeys(
        ("source_contract", "source_check", "target_contract", "auxiliary_inputs", "read", "process",
         "assemble", "create", "input_check", "write", "auxiliary_copy", "validate"), "not-run"))
    failures: list[dict] = field(default_factory=list)
    creation_report: object = None
    validation_report: object = None
    coverage: dict = field(init=False)

    def __post_init__(self):
        self.coverage = {"destination": self.destination, "dry_run": self.dry_run,
                         "conformance_passed": False, "written": False}

    @property
    def quick_passed(self):
        return (self.stages["validate"] == "passed" and not self.failures
                and self.validation_report.quick_passed)

    def fail(self, stage, message, error=None):
        self.stages[stage] = "failed"
        self.failures.append({"stage": stage, "message": message,
                              "type": type(error).__name__ if error else None})
        # A failed dry-run must never satisfy the unchanged CLI success check.
        if self.dry_run:
            self.coverage["conformance_passed"] = False

    def fail_checks(self, stage, report):
        failures = [check for check in getattr(report, "checks", ())
                    if check.status not in ("pass", "warning", "not-applicable")]
        self.fail(stage, "; ".join(check.check_id for check in failures)
                  or f"{stage} did not pass; see the underlying report")

    def to_dict(self):
        return {"destination": self.destination, "stages": dict(self.stages),
                "failures": list(self.failures), "coverage": dict(self.coverage),
                "quick_passed": self.quick_passed,
                "creation_report": (self.creation_report.to_dict()
                                    if self.creation_report is not None else None),
                "validation_report": (self.validation_report.to_dict()
                                      if self.validation_report is not None else None)}
