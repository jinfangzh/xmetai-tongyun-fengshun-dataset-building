"""Individual validation outcomes, independent of task/report lifecycle."""

from dataclasses import dataclass
from typing import Any


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


def add_check(
    checks, check_id, status, category, expected=None, observed=None, evidence=None
):
    """Append one outcome without interpreting publication or scan state."""
    checks.append(CheckResult(check_id, status, category, expected, observed, evidence))


def require_check(
    checks, check_id, condition, category, expected=None, observed=None, evidence=None
):
    """Record a boolean requirement and return its truth value for prerequisites."""
    add_check(
        checks,
        check_id,
        "pass" if condition else "fail",
        category,
        expected,
        observed,
        evidence,
    )
    return bool(condition)
