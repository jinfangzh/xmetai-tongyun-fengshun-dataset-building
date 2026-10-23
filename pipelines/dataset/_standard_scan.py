"""Main block scanning, physical coverage and bounded progress bookkeeping.

The caller owns opened arrays and the quality accumulator. This module never
loads a whole dataset, writes reports, or constructs product expectations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from datakit.schema.standard_dataset import DatasetValidationSpec
from datakit.validation._standard_quality import StandardQualityAccumulator
from datakit.validation.reports import ValidationReport
from datakit.validation.zarr_inspection import expected_zarr_blocks, zarr_chunk_path

from ._validation_runtime import _fingerprint, _resident_memory_bytes

INTEGRITY = "data_integrity"


def scan_standard_blocks(
    array: Any,
    metadata: dict[str, Any],
    spec: DatasetValidationSpec,
    report: ValidationReport,
    quality: StandardQualityAccumulator,
    *,
    stable: bool,
    sample_blocks: int,
    max_blocks: int | None,
    memory_budget_mib: int,
    stop_requested: Callable[[], bool] | None,
    on_block: Callable[[dict[str, Any]], None] | None,
) -> bool:
    """Scan selected raw frames and return whether producer stability survived.

    Flush the last completed range even if a callback raises or interrupts.
    Overall conclusions and metadata stability remain the caller's concern.
    """
    root = Path(metadata["path"])
    mode, coverage = report.mode, report.coverage
    total = spec.expected_shape[0]
    completed_start = completed_end = None
    try:
        selected = (
            set(np.linspace(0, total - 1, min(total, sample_blocks), dtype=int).tolist())
            if mode == "quick"
            else None
        )
        coverage["selection"] = (
            sorted(selected) if selected is not None else "all expected blocks in index order"
        )
        approved = set(spec.allowed_omitted)
        scanned = 0
        for index, selection in expected_zarr_blocks(spec.expected_shape, spec.expected_chunks):
            if selected is not None and index[0] not in selected:
                continue
            if (max_blocks is not None and scanned >= max_blocks) or (
                stop_requested is not None and stop_requested()
            ):
                coverage["interrupted"] = True
                break
            key = zarr_chunk_path(metadata, "data", index)
            fingerprint = _fingerprint(key)
            if fingerprint is not None:
                storage_status = "physically_present"
            elif index in approved and stable:
                storage_status = "allowed_omitted"
            else:
                storage_status = "unapproved_missing"
            coverage[storage_status] += 1
            coverage["storage_unknown"] -= 1
            entry = {
                "index": index,
                "selection": [[s.start, s.stop] for s in selection],
                "key": str(key.relative_to(root)),
                "storage": storage_status,
            }
            try:
                block = np.asarray(array[selection])
                if block.shape != tuple(
                    s.stop - s.start for s in selection
                ) or block.dtype != np.dtype("f2"):
                    raise ValueError(f"Raw block shape/dtype differs: {block.shape}, {block.dtype}")
                entry["anomalies"] = quality.update(block)
                del block
                coverage["value_checked"] += 1
                entry["read"] = "value_checked"
                if completed_end is not None and index[0] == completed_end + 1:
                    completed_end = index[0]
                else:
                    if completed_start is not None and len(coverage["completed_ranges"]) < 128:
                        coverage["completed_ranges"].append([completed_start, completed_end])
                    elif completed_start is not None:
                        coverage["range_detail_truncated"] = True
                    completed_start = completed_end = index[0]
            except Exception as exc:
                coverage["read_failed"] += 1
                entry.update(
                    read="read_failed", error={"type": type(exc).__name__, "message": str(exc)}
                )
                coverage["error_count"] += 1
                if len(coverage["errors"]) < 20:
                    coverage["errors"].append(entry)
            coverage["not_scanned"] -= 1
            scanned += 1
            if _fingerprint(key) != fingerprint:
                stable = False
                report.add("publication.block_changed", "fail", INTEGRITY, observed=index)
            if on_block is not None:
                on_block(entry)
            resident = _resident_memory_bytes()
            if resident is not None:
                previous = coverage.get("sampled_peak_resident_bytes")
                coverage["sampled_peak_resident_bytes"] = max(resident, previous or 0)
            if resident is not None and resident > memory_budget_mib * 1024**2:
                report.add("scan.peak_budget", "fail", INTEGRITY, memory_budget_mib * 1024**2, resident)
                coverage["interrupted"] = True
                break
        if mode == "full" and coverage["storage_unknown"] == 0:
            extra_keys = []
            extra_count = 0
            for file in (root / "data").rglob("*"):
                if not file.is_file() or file.name in ("zarr.json", ".zarray", ".zattrs"):
                    continue
                relative = file.relative_to(root / "data").as_posix()
                raw = metadata["arrays"]["data"]["raw"]
                encoding = raw.get(
                    "chunk_key_encoding",
                    {
                        "name": "v2",
                        "configuration": {"separator": raw.get("dimension_separator", ".")},
                    },
                )
                separator = encoding.get("configuration", {}).get(
                    "separator", "/" if encoding["name"] == "default" else "."
                )
                components = relative.split(separator)
                if encoding["name"] == "default" and components and components[0] == "c":
                    components = components[1:]
                good = len(components) == 4 and all(c.isdigit() for c in components)
                if good:
                    parsed = tuple(int(c) for c in components)
                    good = (
                        parsed[1:] == (0, 0, 0)
                        and 0 <= parsed[0] < total
                        and zarr_chunk_path(metadata, "data", parsed) == file
                    )
                if not good:
                    extra_count += 1
                    if len(extra_keys) < 20:
                        extra_keys.append(relative)
            report.require(
                "storage.no_unexpected_keys",
                extra_count == 0,
                INTEGRITY,
                0,
                {"count": extra_count, "examples": extra_keys},
            )
    finally:
        if completed_start is not None and len(coverage["completed_ranges"]) < 128:
            coverage["completed_ranges"].append([completed_start, completed_end])
        elif completed_start is not None:
            coverage["range_detail_truncated"] = True
    return stable
