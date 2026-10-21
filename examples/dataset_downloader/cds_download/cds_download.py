"""Build and run resumable CDS download batches."""

import argparse
import calendar
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, local
from typing import Any, Iterator, Literal, NamedTuple

import yaml


class DatasetSpec(NamedTuple):
    request_kind: Literal["hourly", "daily"]
    level_kind: Literal["single", "pressure", "land"]
    requires_product_type: bool
    fixed_suffix: str | None = None


# Dataset identity and request rules live here; downstream code uses categories.
# Multiple physical profiles (sfc/wave/static or sfc/t2m) may share a dataset.
DATASET_SPECS = {
    "reanalysis-era5-single-levels": DatasetSpec("hourly", "single", True),
    "reanalysis-era5-pressure-levels": DatasetSpec("hourly", "pressure", True),
    "reanalysis-era5-land": DatasetSpec("hourly", "land", False),
    "derived-era5-single-levels-daily-statistics": DatasetSpec("daily", "single", True, "zip"),
    "derived-era5-pressure-levels-daily-statistics": DatasetSpec("daily", "pressure", True, "zip"),
    # Retain support for this daily endpoint even without a bundled profile.
    "derived-era5-land-daily-statistics": DatasetSpec("daily", "land", False, "zip"),
}


def dataset_spec(profile: dict[str, Any]) -> DatasetSpec:
    dataset = profile.get("dataset")
    if not isinstance(dataset, str) or dataset not in DATASET_SPECS:
        raise ValueError(
            f"Unclassified CDS dataset: {dataset!r}; add its request rules to DATASET_SPECS"
        )
    return DATASET_SPECS[dataset]


class CDSTaskBuilder:
    def __init__(
        self,
        profile_paths: str | Path | Sequence[str | Path],
        start_month: str,
        end_month: str,
        output_roots: Sequence[str | Path | None] | None = None,
    ) -> None:
        self.start_month = self._parse_month(start_month)
        self.end_month = self._parse_month(end_month)
        if self.start_month > self.end_month:
            raise ValueError("start_month must not be later than end_month")

        if isinstance(profile_paths, (str, Path)):
            profile_paths = [profile_paths]
        if not profile_paths:
            raise ValueError("at least one profile path is required")
        if output_roots is None:
            output_roots = [None] * len(profile_paths)
        if len(output_roots) != len(profile_paths):
            raise ValueError("output_roots must match the number of profiles")

        self.max_workers = 1
        self.max_retries = 0
        self.retry_delay_seconds = 0.0
        self.continue_on_error = False
        self.log_path: Path | None = None
        self.status_log_path: Path | None = None
        self.plan_path: Path | None = None
        self.profiles: list[tuple[Path, dict[str, Any], Path]] = []
        for profile_path_value, output_root_value in zip(
            profile_paths, output_roots, strict=True
        ):
            profile_path = Path(profile_path_value).expanduser().resolve()
            with profile_path.open(encoding="utf-8") as profile_file:
                profile = yaml.safe_load(profile_file)
            if not isinstance(profile, dict):
                raise ValueError(f"Profile must be a mapping: {profile_path}")
            expanded_profiles = self._expand_profile(profile, profile_path)
            if output_root_value is None:
                legacy_output_root = profile.get("download_store_path")
                if not isinstance(legacy_output_root, str):
                    raise ValueError(
                        f"No output path provided for profile: {profile_path}"
                    )
                output_root = profile_path.parent / legacy_output_root
            else:
                output_root = Path(output_root_value).expanduser()
            for expanded, statistic in expanded_profiles:
                target_root = output_root / statistic if statistic else output_root
                self.profiles.append((profile_path, expanded, target_root.resolve()))

    @classmethod
    def from_batch_task(cls, batch_task: str | Path) -> "CDSTaskBuilder":
        batch_path = Path(batch_task).expanduser().resolve()
        with batch_path.open(encoding="utf-8") as batch_file:
            batch = yaml.safe_load(batch_file)
        if not isinstance(batch, dict):
            raise ValueError(f"Batch task must be a mapping: {batch_path}")

        required_fields = {"profiles"}
        missing_fields = required_fields.difference(batch)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Missing required fields in {batch_path}: {missing}")

        period = batch.get("period")
        legacy_period_fields = {"start_month", "end_month"}.intersection(batch)
        if period is not None and legacy_period_fields:
            raise ValueError("use either 'period' or top-level month fields, not both")
        if period is None:
            period = batch
        if not isinstance(period, dict):
            raise ValueError("batch field 'period' must be a mapping")
        missing_period_fields = {"start_month", "end_month"}.difference(period)
        if missing_period_fields:
            missing = ", ".join(sorted(missing_period_fields))
            raise ValueError(f"Missing required period fields: {missing}")

        profile_values = batch["profiles"]
        if not isinstance(profile_values, list) or not profile_values:
            raise ValueError("batch field 'profiles' must be a non-empty list")

        storage = batch.get("storage")
        if storage is not None and not isinstance(storage, dict):
            raise ValueError("batch field 'storage' must be a mapping")
        storage_root: Path | None = None
        if isinstance(storage, dict):
            if "root_path" not in storage:
                raise ValueError("Missing required storage field: root_path")
            storage_root = cls._resolve_batch_path(
                storage["root_path"], batch_path, "storage.root_path"
            )

        profile_paths = []
        output_roots: list[Path | None] = []
        for profile_value in profile_values:
            if isinstance(profile_value, str):
                profile_path_value = profile_value
                output_root = None
            elif isinstance(profile_value, dict):
                if storage_root is None:
                    raise ValueError(
                        "batch field 'storage.root_path' is required for profile mappings"
                    )
                if set(profile_value) != {"path", "output_subdir"}:
                    raise ValueError(
                        "profile mappings must contain exactly path and output_subdir"
                    )
                profile_path_value = profile_value["path"]
                output_subdir = profile_value["output_subdir"]
                if not isinstance(output_subdir, str):
                    raise ValueError("profile output_subdir must be a string")
                output_subdir_path = Path(output_subdir)
                if output_subdir_path.is_absolute():
                    raise ValueError("profile output_subdir must be relative")
                output_root = storage_root / output_subdir_path
            else:
                raise ValueError("batch profiles must be strings or mappings")
            if not isinstance(profile_path_value, str):
                raise ValueError("all batch profile paths must be strings")
            profile_path = Path(profile_path_value).expanduser()
            if not profile_path.is_absolute():
                profile_path = batch_path.parent / profile_path
            profile_paths.append(profile_path)
            output_roots.append(output_root)

        builder = cls(
            profile_paths,
            str(period["start_month"]),
            str(period["end_month"]),
            output_roots,
        )
        if "execution" in batch and "runner" in batch:
            raise ValueError("use either 'execution' or 'runner', not both")
        execution = batch.get("execution", batch.get("runner", {}))
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
        if (
            not isinstance(builder.retry_delay_seconds, (int, float))
            or builder.retry_delay_seconds < 0
        ):
            raise ValueError("execution.retry_delay_seconds must be non-negative")
        if not isinstance(builder.continue_on_error, bool):
            raise ValueError("execution.continue_on_error must be a boolean")

        if "records" in batch and "log_path" in batch:
            raise ValueError("use either 'records' or 'log_path', not both")
        records = batch.get("records")
        if records is None:
            if "log_path" in batch:
                records = {"status_log_path": batch["log_path"]}
            elif storage_root is not None:
                records_root = storage_root / "_records" / batch_path.stem
                builder.plan_path = records_root / "plan.txt"
                builder.status_log_path = records_root / "status.jsonl"
                builder.log_path = builder.status_log_path
                return builder
            else:
                raise ValueError(
                    "storage.root_path is required when records/log_path is omitted"
                )
        if not isinstance(records, dict):
            raise ValueError("batch field 'records' must be a mapping")
        status_fields = [
            field
            for field in ("status_log_path", "event_log_path")
            if field in records
        ]
        if len(status_fields) != 1:
            raise ValueError(
                "records must contain exactly one of status_log_path or event_log_path"
            )
        status_field = status_fields[0]
        status_log_path = cls._resolve_batch_path(
            records[status_field], batch_path, f"records.{status_field}"
        )
        if "target_list_path" in records and "plan_path" in records:
            raise ValueError(
                "use either 'records.target_list_path' or 'records.plan_path', not both"
            )
        plan_value = records.get("plan_path", records.get("target_list_path"))
        if plan_value is None:
            builder.plan_path = status_log_path.with_suffix(".plan.txt")
        else:
            builder.plan_path = cls._resolve_batch_path(
                plan_value, batch_path, "records.plan_path"
            )
        builder.status_log_path = status_log_path
        builder.log_path = status_log_path
        return builder

    @staticmethod
    def _resolve_batch_path(value: Any, batch_path: Path, field: str) -> Path:
        if not isinstance(value, str):
            raise ValueError(f"batch field '{field}' must be a string")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = batch_path.parent / path
        return path.resolve()

    @staticmethod
    def _parse_month(value: str) -> datetime:
        try:
            return datetime.strptime(value, "%Y-%m")
        except ValueError as error:
            raise ValueError(f"Invalid month {value!r}; expected YYYY-MM") from error

    @classmethod
    def _expand_profile(
        cls, profile: dict[str, Any], profile_path: Path
    ) -> list[tuple[dict[str, Any], str | None]]:
        """Expand local statistic groups into ordinary CDS request profiles."""
        if "statistics" not in profile:
            cls._validate_profile(profile, profile_path)
            return [(profile, None)]
        if dataset_spec(profile).request_kind != "daily":
            raise ValueError("statistics requires a daily statistics dataset")
        if {"variable", "daily_statistic"}.intersection(profile):
            raise ValueError("grouped profiles must put variables inside statistics")
        groups = profile["statistics"]
        if not isinstance(groups, dict) or not groups:
            raise ValueError("statistics must be a non-empty mapping")
        common = {key: value for key, value in profile.items() if key != "statistics"}
        expanded = []
        for statistic, selection in groups.items():
            if statistic not in {"daily_mean", "daily_sum", "daily_max", "daily_min"}:
                raise ValueError(f"unsupported daily_statistic: {statistic}")
            if not isinstance(selection, dict) or set(selection) != {"variable"}:
                raise ValueError("each statistics entry must contain exactly variable")
            item = dict(common, daily_statistic=statistic, variable=selection["variable"])
            cls._validate_profile(item, profile_path)
            expanded.append((item, statistic))
        return expanded

    @staticmethod
    def _validate_profile(profile: dict[str, Any], profile_path: Path) -> None:
        spec = dataset_spec(profile)
        daily = spec.request_kind == "daily"
        required_fields = {"dataset", "variable"}
        if daily:
            required_fields.update({"daily_statistic", "time_zone", "frequency"})
        else:
            required_fields.update({"time", "data_format", "download_format"})
        if spec.requires_product_type:
            required_fields.add("product_type")
        if spec.level_kind == "pressure":
            required_fields.add("pressure_level")
        missing_fields = required_fields.difference(profile)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Missing required fields in {profile_path}: {missing}")

        if spec.level_kind != "pressure" and "pressure_level" in profile:
            raise ValueError("pressure_level is only valid for pressure datasets")
        if spec.level_kind == "pressure":
            levels = profile["pressure_level"]
            if not isinstance(levels, list) or not levels:
                raise ValueError("pressure_level must be a non-empty list")

        variables = profile["variable"]
        if not isinstance(variables, list) or not variables:
            raise ValueError("config field 'variable' must be a non-empty list")
        if not all(isinstance(variable, str) and variable for variable in variables):
            raise ValueError("all variable values must be non-empty strings")
        if len(set(variables)) != len(variables):
            raise ValueError("duplicate variables in profile")
        if {"year", "month", "day"}.intersection(profile):
            raise ValueError("dates belong in the batch period, not the profile")
        if daily:
            forbidden = {"time", "data_format", "download_format"}.intersection(profile)
            if forbidden:
                raise ValueError("daily profiles must omit: " + ", ".join(sorted(forbidden)))
            if profile["daily_statistic"] not in {
                "daily_mean", "daily_sum", "daily_max", "daily_min"
            }:
                raise ValueError("unsupported daily_statistic")
            if profile["frequency"] not in {"1_hourly", "3_hourly", "6_hourly"}:
                raise ValueError("frequency must be 1_hourly, 3_hourly or 6_hourly")
            zone = profile["time_zone"]
            if not isinstance(zone, str) or not re.fullmatch(r"utc[+-](?:0[0-9]|1[0-4]):00", zone):
                raise ValueError("time_zone must be an hourly UTC offset, e.g. utc+00:00")
        else:
            if {"daily_statistic", "frequency", "time_zone"}.intersection(profile):
                raise ValueError("daily statistics fields require a daily dataset")
            times = profile["time"]
            if (not isinstance(times, list) or not times
                or not all(isinstance(t, str) and re.fullmatch(r"(?:[01][0-9]|2[0-3]):00", t) for t in times)
                or len(set(times)) != len(times)):
                raise ValueError("time must contain unique hourly HH:00 strings")

    def _iter_months(self) -> Iterator[tuple[int, int]]:
        current = self.start_month
        while current <= self.end_month:
            yield current.year, current.month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

    @staticmethod
    def _target_suffix(profile: dict[str, Any]) -> str:
        spec = dataset_spec(profile)
        if spec.fixed_suffix is not None:
            return spec.fixed_suffix
        download_format = profile["download_format"]
        if download_format != "unarchived":
            return str(download_format)

        data_format_suffixes = {
            "netcdf": "nc",
            "grib": "grib",
        }
        data_format = profile["data_format"]
        try:
            return data_format_suffixes[data_format]
        except KeyError as error:
            raise ValueError(
                "Cannot determine target suffix for "
                f"data_format={data_format!r}, download_format='unarchived'"
            ) from error

    def build(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for profile_path, profile, output_root in self.profiles:
            for variable in profile["variable"]:
                for year, month in self._iter_months():
                    days = [
                        f"{day:02d}"
                        for day in range(
                            1, calendar.monthrange(year, month)[1] + 1
                        )
                    ]
                    request: dict[str, Any] = {}
                    for key, value in profile.items():
                        if (
                            key in {"dataset", "download_store_path"}
                            or value is None
                            or (key == "area" and value == [])
                        ):
                            continue
                        request[key] = [variable] if key == "variable" else value
                        if key == "variable":
                            request["year"] = [str(year)]
                            request["month"] = [f"{month:02d}"]
                            request["day"] = days

                    suffix = self._target_suffix(profile)
                    target = output_root / variable / str(year) / (
                        f"{variable}_{year}{month:02d}.{suffix}"
                    )
                    tasks.append(
                        {
                            "dataset": profile["dataset"],
                            "request": request,
                            "target": str(target),
                        }
                    )

        targets: set[str] = set()
        for task in tasks:
            if task["target"] in targets:
                raise ValueError(
                    f"Duplicate output target: {task['target']}; "
                    "use separate output_subdir values for different profiles/statistics"
                )
            targets.add(task["target"])
        return tasks


CDSTask = CDSTaskBuilder


class CDSTaskRunner:
    def __init__(
        self,
        tasks: list[dict[str, Any]],
        log_path: str | Path,
        max_workers: int = 1,
        max_retries: int = 0,
        retry_delay_seconds: float = 0.0,
        continue_on_error: bool = False,
        resume: bool = True,
    ) -> None:
        self.tasks = tasks
        self.log_path = Path(log_path).expanduser().resolve()
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.continue_on_error = continue_on_error
        self.resume = resume
        self.plan_id = hashlib.sha256(
            json.dumps(tasks, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._log_lock = Lock()

    @staticmethod
    def _task_id(task: dict[str, Any]) -> str:
        payload = json.dumps(task, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _completed_task_ids(self) -> set[str]:
        if not self.log_path.exists():
            return set()

        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        completed: set[str] = set()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if line_number == len(lines):
                    break
                raise ValueError(
                    f"Invalid JSONL record at {self.log_path}:{line_number}"
                ) from error
            if record.get("status") == "completed":
                completed.add(record["task_id"])
        return completed

    def export_target_list(
        self, target_list_path: str | Path, batch_task: str | Path
    ) -> None:
        path = Path(target_list_path).expanduser().resolve()
        lines = [
            f"# generated_at: {datetime.now(timezone.utc).isoformat()}",
            f"# plan_id: {self.plan_id}",
            f"# batch_task: {Path(batch_task).expanduser().resolve()}",
            f"# task_count: {len(self.tasks)}",
            *(task["target"] for task in self.tasks),
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as target_list_file:
            target_list_file.write("\n".join(lines) + "\n")
            target_list_file.flush()
            os.fsync(target_list_file.fileno())
        temporary_path.replace(path)

    def export_plan(self, plan_path: str | Path, batch_task: str | Path) -> None:
        self.export_target_list(plan_path, batch_task)

    def _append_log(
        self,
        task: dict[str, Any],
        task_id: str,
        status: str,
        error: str | None = None,
        attempt: int | None = None,
    ) -> None:
        request = task["request"]
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "plan_id": self.plan_id,
            "task_id": task_id,
            "status": status,
            "dataset": task["dataset"],
            "variable": request.get("variable"),
            "year": request.get("year"),
            "month": request.get("month"),
            "target": task["target"],
        }
        if error is not None:
            record["error"] = error
        if attempt is not None:
            record["attempt"] = attempt

        with self._log_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                os.fsync(log.fileno())

    def _download_task(self, task: dict[str, Any], get_client: Any) -> None:
        task_id = self._task_id(task)
        target = Path(task["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        partial_target = target.with_name(f"{target.name}.part")

        for attempt in range(1, self.max_retries + 2):
            partial_target.unlink(missing_ok=True)
            if attempt == 1:
                self._append_log(task, task_id, "started", attempt=attempt)

            try:
                client = get_client(reconnect=attempt > 1)
                client.retrieve(
                    task["dataset"],
                    task["request"],
                    str(partial_target),
                )
                if not partial_target.is_file():
                    raise RuntimeError(
                        f"CDS client did not create target file: {partial_target}"
                    )
                partial_target.replace(target)
            except Exception as error:
                if attempt <= self.max_retries:
                    self._append_log(
                        task, task_id, "retrying", repr(error), attempt
                    )
                    time.sleep(self.retry_delay_seconds)
                    continue
                self._append_log(task, task_id, "failed", repr(error), attempt)
                raise
            else:
                self._append_log(task, task_id, "completed", attempt=attempt)
                return

    def run(
        self,
        client: Any | None = None,
        dry_run: bool = False,
    ) -> dict[str, int]:
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if client is not None and self.max_workers > 1:
            raise ValueError("a shared client can only be used with max_workers=1")

        completed_task_ids = self._completed_task_ids() if self.resume else set()
        summary = {
            "total": len(self.tasks),
            "completed": 0,
            "skipped": 0,
            "failed": 0,
        }

        pending_tasks = []
        for task in self.tasks:
            task_id = self._task_id(task)
            target = Path(task["target"])
            if task_id in completed_task_ids and target.is_file():
                summary["skipped"] += 1
                continue
            pending_tasks.append(task)

        if dry_run:
            for task in pending_tasks:
                print(json.dumps(task, ensure_ascii=False))
            return summary

        if client is None:
            import cdsapi

            thread_state = local()

            def get_client(reconnect: bool = False) -> Any:
                if reconnect or not hasattr(thread_state, "client"):
                    thread_state.client = cdsapi.Client()
                return thread_state.client

        else:

            def get_client(reconnect: bool = False) -> Any:
                return client

        def download_with_thread_client(task: dict[str, Any]) -> None:
            self._download_task(task, get_client)

        if self.max_workers == 1:
            for task in pending_tasks:
                try:
                    self._download_task(task, get_client)
                except Exception:
                    summary["failed"] += 1
                    if not self.continue_on_error:
                        raise
                else:
                    summary["completed"] += 1
            return summary

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(download_with_thread_client, task): task
                for task in pending_tasks
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    summary["failed"] += 1
                    if not self.continue_on_error:
                        for pending in futures:
                            pending.cancel()
                        raise
                else:
                    summary["completed"] += 1

        return summary


def run_batch(
    batch_task: str | Path,
    log_path: str | Path | None = None,
    client: Any | None = None,
    dry_run: bool = False,
    plan_only: bool = False,
    plan_path: str | Path | None = None,
    target_list_path: str | Path | None = None,
    status_log_path: str | Path | None = None,
    resume: bool = True,
) -> dict[str, int]:
    builder = CDSTaskBuilder.from_batch_task(batch_task)
    if status_log_path is not None and log_path is not None:
        raise ValueError("use either status_log_path or log_path, not both")
    resolved_log_path = status_log_path or log_path or builder.status_log_path
    if target_list_path is not None and plan_path is not None:
        raise ValueError("use either target_list_path or plan_path, not both")
    resolved_plan_path = (
        plan_path or target_list_path or builder.plan_path
    )
    if resolved_log_path is None:
        raise ValueError("status_log_path is required")
    if resolved_plan_path is None:
        raise ValueError("plan_path is required")
    runner = CDSTaskRunner(
        builder.build(),
        resolved_log_path,
        builder.max_workers,
        builder.max_retries,
        builder.retry_delay_seconds,
        builder.continue_on_error,
        resume,
    )
    runner.export_plan(resolved_plan_path, batch_task)
    if plan_only:
        return {"total": len(runner.tasks), "completed": 0, "skipped": 0, "failed": 0}
    return runner.run(client, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run CDS download tasks with status-based resume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  Resume the batch (default):
        python examples/dataset_downloader/cds_download/cds_download.py examples/dataset_downloader/cds_download/batch_task.yaml

  Generate the download plan without downloading:
        python examples/dataset_downloader/cds_download/cds_download.py examples/dataset_downloader/cds_download/batch_task.yaml --plan-only

  Ignore completed status and download every task again:
        python examples/dataset_downloader/cds_download/cds_download.py examples/dataset_downloader/cds_download/batch_task.yaml --force-redownload
""",
    )
    parser.add_argument("batch_task", type=Path, help="Batch Task YAML path")
    parser.add_argument(
        "--status-log-path",
        "--event-log-path",
        "--log-path",
        dest="status_log_path",
        type=Path,
        help="override the status JSONL path",
    )
    parser.add_argument(
        "--plan-path",
        "--target-list-path",
        dest="plan_path",
        type=Path,
        help="override the download plan path",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only",
        "--target-list-only",
        dest="plan_only",
        action="store_true",
        help="generate the plan and exit without downloading",
    )
    mode.add_argument(
        "--force-redownload",
        action="store_true",
        help="ignore completed status and download every task again",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print pending task requests without downloading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_batch(
        args.batch_task,
        status_log_path=args.status_log_path,
        dry_run=args.dry_run,
        plan_only=args.plan_only,
        plan_path=args.plan_path,
        resume=not args.force_redownload,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
