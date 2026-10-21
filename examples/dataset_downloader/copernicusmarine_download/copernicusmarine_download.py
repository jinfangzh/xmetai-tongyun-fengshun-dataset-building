"""Build and run resumable Copernicus Marine subset download batches."""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import yaml


class CopernicusMarineTaskBuilder:
    """Expand Profile and Batch YAML into one Copernicus Marine task per day."""

    def __init__(
        self,
        profile_paths: list[Path],
        output_roots: list[Path],
        start_date: str,
        end_date: str,
    ) -> None:
        self.start_date = self._parse_date(start_date)
        self.end_date = self._parse_date(end_date)
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be later than end_date")
        if not profile_paths or len(profile_paths) != len(output_roots):
            raise ValueError("profiles and output roots must be non-empty and aligned")

        self.max_workers = 1
        self.max_retries = 0
        self.retry_delay_seconds = 0.0
        self.continue_on_error = False
        self.plan_path: Path | None = None
        self.status_log_path: Path | None = None
        self.profiles: list[tuple[Path, dict[str, Any], Path]] = []
        for profile_path, output_root in zip(profile_paths, output_roots, strict=True):
            resolved = profile_path.expanduser().resolve()
            with resolved.open(encoding="utf-8") as stream:
                profile = yaml.safe_load(stream)
            if not isinstance(profile, dict):
                raise ValueError(f"Profile must be a mapping: {resolved}")
            self._validate_profile(profile, resolved)
            self.profiles.append((resolved, profile, output_root.expanduser().resolve()))

    @classmethod
    def from_batch_task(cls, batch_task: str | Path) -> "CopernicusMarineTaskBuilder":
        batch_path = Path(batch_task).expanduser().resolve()
        with batch_path.open(encoding="utf-8") as stream:
            batch = yaml.safe_load(stream)
        if not isinstance(batch, dict):
            raise ValueError(f"Batch task must be a mapping: {batch_path}")

        profiles = batch.get("profiles")
        storage = batch.get("storage")
        period = batch.get("period")
        if not isinstance(profiles, list) or not profiles:
            raise ValueError("batch field 'profiles' must be a non-empty list")
        if not isinstance(storage, dict) or "root_path" not in storage:
            raise ValueError("batch field 'storage.root_path' is required")
        if not isinstance(period, dict):
            raise ValueError("batch field 'period' must be a mapping")
        missing = {"start_date", "end_date"}.difference(period)
        if missing:
            raise ValueError("Missing required period fields: " + ", ".join(sorted(missing)))

        storage_root = cls._resolve_path(storage["root_path"], batch_path, "storage.root_path")
        profile_paths: list[Path] = []
        output_roots: list[Path] = []
        for entry in profiles:
            if not isinstance(entry, dict) or set(entry) != {"path", "output_subdir"}:
                raise ValueError("profile entries must contain exactly path and output_subdir")
            profile_paths.append(cls._resolve_path(entry["path"], batch_path, "profiles.path"))
            subdir = entry["output_subdir"]
            if not isinstance(subdir, str) or Path(subdir).is_absolute():
                raise ValueError("profile output_subdir must be a relative string")
            output_roots.append(storage_root / subdir)

        builder = cls(
            profile_paths,
            output_roots,
            str(period["start_date"]),
            str(period["end_date"]),
        )
        execution = batch.get("execution", {})
        if not isinstance(execution, dict):
            raise ValueError("batch field 'execution' must be a mapping")
        builder.max_workers = execution.get("max_workers", 1)
        builder.max_retries = execution.get("max_retries", 0)
        builder.retry_delay_seconds = execution.get("retry_delay_seconds", 0.0)
        builder.continue_on_error = execution.get("continue_on_error", False)
        if not isinstance(builder.max_workers, int) or builder.max_workers < 1:
            raise ValueError("execution.max_workers must be an integer of at least 1")
        if not isinstance(builder.max_retries, int) or builder.max_retries < 0:
            raise ValueError("execution.max_retries must be a non-negative integer")
        if not isinstance(builder.retry_delay_seconds, (int, float)) or builder.retry_delay_seconds < 0:
            raise ValueError("execution.retry_delay_seconds must be non-negative")
        if not isinstance(builder.continue_on_error, bool):
            raise ValueError("execution.continue_on_error must be a boolean")

        records = batch.get("records")
        if records is None:
            records_root = storage_root / "_records" / batch_path.stem
            builder.plan_path = records_root / "plan.txt"
            builder.status_log_path = records_root / "status.jsonl"
        else:
            if not isinstance(records, dict):
                raise ValueError("batch field 'records' must be a mapping")
            builder.plan_path = cls._resolve_path(records.get("plan_path"), batch_path, "records.plan_path")
            builder.status_log_path = cls._resolve_path(
                records.get("status_log_path"), batch_path, "records.status_log_path"
            )
        return builder

    @staticmethod
    def _resolve_path(value: Any, batch_path: Path, field: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"batch field '{field}' must be a non-empty string")
        path = Path(value).expanduser()
        return (path if path.is_absolute() else batch_path.parent / path).resolve()

    @staticmethod
    def _parse_date(value: str) -> date:
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD") from error

    @staticmethod
    def _validate_profile(profile: dict[str, Any], path: Path) -> None:
        required = {"dataset_id", "variables", "longitude", "latitude", "depth", "file_tag"}
        missing = required.difference(profile)
        if missing:
            raise ValueError(f"Missing required fields in {path}: {', '.join(sorted(missing))}")
        if not isinstance(profile["dataset_id"], str) or not profile["dataset_id"]:
            raise ValueError("dataset_id must be a non-empty string")
        variables = profile["variables"]
        if (
            not isinstance(variables, list)
            or not variables
            or not all(isinstance(item, str) and item for item in variables)
            or len(set(variables)) != len(variables)
        ):
            raise ValueError("variables must contain unique non-empty strings")
        for field in ("longitude", "latitude", "depth"):
            bounds = profile[field]
            if not isinstance(bounds, dict) or set(bounds) != {"minimum", "maximum"}:
                raise ValueError(f"{field} must contain exactly minimum and maximum")
            if not all(isinstance(bounds[key], (int, float)) for key in bounds):
                raise ValueError(f"{field} bounds must be numeric")
            if bounds["minimum"] > bounds["maximum"]:
                raise ValueError(f"{field}.minimum must not exceed maximum")
        if not isinstance(profile["file_tag"], str) or not profile["file_tag"]:
            raise ValueError("file_tag must be a non-empty string")
        allowed = required | {"dataset_version", "dataset_part", "service"}
        unknown = set(profile).difference(allowed)
        if unknown:
            raise ValueError("Unsupported profile fields: " + ", ".join(sorted(unknown)))

    def _iter_dates(self):
        current = self.start_date
        while current <= self.end_date:
            yield current
            current += timedelta(days=1)

    def build(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for _, profile, output_root in self.profiles:
            for day in self._iter_dates():
                request = {
                    "dataset_id": profile["dataset_id"],
                    "variables": profile["variables"],
                    "minimum_longitude": profile["longitude"]["minimum"],
                    "maximum_longitude": profile["longitude"]["maximum"],
                    "minimum_latitude": profile["latitude"]["minimum"],
                    "maximum_latitude": profile["latitude"]["maximum"],
                    "minimum_depth": profile["depth"]["minimum"],
                    "maximum_depth": profile["depth"]["maximum"],
                    "start_datetime": day.isoformat(),
                    "end_datetime": day.isoformat(),
                }
                for field in ("dataset_version", "dataset_part", "service"):
                    if profile.get(field) is not None:
                        request[field] = profile[field]
                target = output_root / str(day.year) / f"{day:%Y%m%d}_{profile['file_tag']}.nc"
                tasks.append({"request": request, "target": str(target)})
        targets = [task["target"] for task in tasks]
        if len(targets) != len(set(targets)):
            raise ValueError("Duplicate output target; use distinct profile output_subdir values")
        return tasks


class CopernicusMarineTaskRunner:
    def __init__(
        self,
        tasks: list[dict[str, Any]],
        status_log_path: str | Path,
        max_workers: int = 1,
        max_retries: int = 0,
        retry_delay_seconds: float = 0.0,
        continue_on_error: bool = False,
        resume: bool = True,
    ) -> None:
        self.tasks = tasks
        self.status_log_path = Path(status_log_path).expanduser().resolve()
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.continue_on_error = continue_on_error
        self.resume = resume
        self.plan_id = hashlib.sha256(self._canonical(tasks).encode()).hexdigest()
        self._log_lock = Lock()

    @staticmethod
    def _canonical(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _task_id(cls, task: dict[str, Any]) -> str:
        return hashlib.sha256(cls._canonical(task).encode()).hexdigest()

    def _completed_ids(self) -> set[str]:
        if not self.status_log_path.exists():
            return set()
        completed = set()
        lines = self.status_log_path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if number == len(lines):
                    break
                raise ValueError(f"Invalid JSONL record at {self.status_log_path}:{number}") from error
            if record.get("status") == "completed":
                completed.add(record["task_id"])
        return completed

    def export_plan(self, path: str | Path, batch_task: str | Path) -> None:
        plan_path = Path(path).expanduser().resolve()
        lines = [
            f"# generated_at: {datetime.now(timezone.utc).isoformat()}",
            f"# plan_id: {self.plan_id}",
            f"# batch_task: {Path(batch_task).expanduser().resolve()}",
            f"# task_count: {len(self.tasks)}",
            *(task["target"] for task in self.tasks),
        ]
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = plan_path.with_name(plan_path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(plan_path)

    def _append_log(self, task: dict[str, Any], status: str, attempt: int, error: str | None = None) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "plan_id": self.plan_id,
            "task_id": self._task_id(task),
            "status": status,
            "dataset_id": task["request"]["dataset_id"],
            "date": task["request"]["start_datetime"],
            "target": task["target"],
            "attempt": attempt,
        }
        if error is not None:
            record["error"] = error
        with self._log_lock:
            self.status_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.status_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _download(self, task: dict[str, Any], subset: Callable[..., Any]) -> None:
        target = Path(task["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.stem + ".part.nc")
        for attempt in range(1, self.max_retries + 2):
            partial.unlink(missing_ok=True)
            if attempt == 1:
                self._append_log(task, "started", attempt)
            try:
                subset(
                    **task["request"],
                    output_directory=str(partial.parent),
                    output_filename=partial.name,
                    overwrite=True,
                )
                if not partial.is_file():
                    raise RuntimeError(f"Copernicus Marine did not create target file: {partial}")
                partial.replace(target)
            except Exception as error:
                if attempt <= self.max_retries:
                    self._append_log(task, "retrying", attempt, repr(error))
                    time.sleep(self.retry_delay_seconds)
                    continue
                self._append_log(task, "failed", attempt, repr(error))
                raise
            else:
                self._append_log(task, "completed", attempt)
                return

    def run(self, subset: Callable[..., Any] | None = None, dry_run: bool = False) -> dict[str, int]:
        completed_ids = self._completed_ids() if self.resume else set()
        summary = {"total": len(self.tasks), "completed": 0, "skipped": 0, "failed": 0}
        pending = []
        for task in self.tasks:
            if self._task_id(task) in completed_ids and Path(task["target"]).is_file():
                summary["skipped"] += 1
            else:
                pending.append(task)
        if dry_run:
            for task in pending:
                print(json.dumps(task, ensure_ascii=False))
            return summary
        if subset is None:
            import copernicusmarine

            # Equivalent to running `copernicusmarine login`: the first run asks
            # for credentials and stores them; later runs reuse the saved login.
            login_succeeded = copernicusmarine.login()
            if login_succeeded is False:
                raise RuntimeError("Copernicus Marine login failed")
            subset = copernicusmarine.subset

        if self.max_workers == 1:
            for task in pending:
                try:
                    self._download(task, subset)
                except Exception:
                    summary["failed"] += 1
                    if not self.continue_on_error:
                        raise
                else:
                    summary["completed"] += 1
            return summary
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._download, task, subset) for task in pending]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    summary["failed"] += 1
                    if not self.continue_on_error:
                        for item in futures:
                            item.cancel()
                        raise
                else:
                    summary["completed"] += 1
        return summary


def run_batch(
    batch_task: str | Path,
    *,
    subset: Callable[..., Any] | None = None,
    dry_run: bool = False,
    plan_only: bool = False,
    resume: bool = True,
) -> dict[str, int]:
    builder = CopernicusMarineTaskBuilder.from_batch_task(batch_task)
    tasks = builder.build()
    if builder.plan_path is None or builder.status_log_path is None:
        raise ValueError("plan and status paths are required")
    runner = CopernicusMarineTaskRunner(
        tasks,
        builder.status_log_path,
        builder.max_workers,
        builder.max_retries,
        builder.retry_delay_seconds,
        builder.continue_on_error,
        resume,
    )
    runner.export_plan(builder.plan_path, batch_task)
    if plan_only:
        return {"total": len(tasks), "completed": 0, "skipped": 0, "failed": 0}
    return runner.run(subset, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable Copernicus Marine subset batches.")
    parser.add_argument("batch_task", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_batch(
        args.batch_task,
        dry_run=args.dry_run,
        plan_only=args.plan_only,
        resume=not args.force_redownload,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
