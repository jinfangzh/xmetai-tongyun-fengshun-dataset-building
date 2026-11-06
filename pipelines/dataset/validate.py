"""Read-only target-contract validation; source/target comparisons are separate."""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from numbers import Integral, Real
from typing import Any, Callable

import numpy as np
import xarray as xr
import yaml

from datakit import io as dataset_io
from datakit.runtime import memory
from datakit.schema import verify_schema_unchanged
from datakit.schema.specifications import AuxiliaryDefinition, DatasetSchema
from datakit.validation.storage import (
    INTEGRITY,
    block_storage_status,
    check_array_encoding,
    check_raw_block,
    check_root_metadata,
    valid_chunk_key,
)
from datakit.validation.structure import (
    STANDARD,
    check_auxiliary_format,
    check_channel_semantics,
    check_dataset_format,
    check_declared_auxiliaries,
    check_declared_root,
    check_mask_labels,
    check_region_layout,
)
from datakit.validation.values import (
    StandardQualityAccumulator,
    check_auxiliary_finite,
    check_auxiliary_values,
    check_mask_values,
    check_statistics,
)
from datakit.validation.zarr_inspection import (
    expected_zarr_blocks,
    file_sha256,
    metadata_digest,
    open_zarr_array_readonly,
    physical_layout_supported,
    read_zarr_block,
    read_zarr_metadata,
    zarr_chunk_path,
)

from .configuration import (
    DatasetValidationSpec,
    verify_execution_unchanged,
)
from .evidence import (
    attach_conversion_evidence,
    check_auxiliary_provenance,
    check_auxiliary_role,
    check_dataset_identity,
    check_mask_provenance,
    check_product_confirmation,
    check_production_receipt,
)
from .reports import (
    ValidationReport,
)


def _check_auxiliary(
    obj: np.ndarray | xr.Dataset,
    definition: AuxiliaryDefinition,
    spec: DatasetSchema,
    mask: np.ndarray | None,
    report: ValidationReport,
) -> None:
    check_auxiliary_provenance(obj, definition, spec, report)
    values, checks = check_auxiliary_format(obj, definition, spec)
    report.checks.extend(checks)
    if values is not None:
        report.checks.extend(check_auxiliary_finite(values, definition))
        check_auxiliary_role(definition, spec, report)
        report.checks.extend(check_auxiliary_values(values, definition, spec, mask))


def _fingerprint(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino
    except FileNotFoundError:
        return None


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
            set(
                np.linspace(0, total - 1, min(total, sample_blocks), dtype=int).tolist()
            )
            if mode == "quick"
            else None
        )
        coverage["selection"] = (
            sorted(selected)
            if selected is not None
            else "all expected blocks in index order"
        )
        approved = set(spec.allowed_omitted)
        scanned = 0
        for index, selection in expected_zarr_blocks(
            spec.expected_shape, spec.expected_chunks
        ):
            if selected is not None and index[0] not in selected:
                continue
            if (max_blocks is not None and scanned >= max_blocks) or (
                stop_requested is not None and stop_requested()
            ):
                coverage["interrupted"] = True
                break
            key = zarr_chunk_path(metadata, "data", index)
            fingerprint = _fingerprint(key)
            storage_status = block_storage_status(
                present=fingerprint is not None,
                approved=index in approved,
                stable=stable,
            )
            coverage[storage_status] += 1
            coverage["storage_unknown"] -= 1
            entry = {
                "index": index,
                "selection": [[s.start, s.stop] for s in selection],
                "key": str(key.relative_to(root)),
                "storage": storage_status,
            }
            try:
                block = read_zarr_block(array, selection)
                check_raw_block(
                    block,
                    tuple(s.stop - s.start for s in selection),
                    spec.arrays["data"].dtype,
                )
                entry["anomalies"] = quality.update(block)
                del block
                coverage["value_checked"] += 1
                entry["read"] = "value_checked"
                if completed_end is not None and index[0] == completed_end + 1:
                    completed_end = index[0]
                else:
                    if (
                        completed_start is not None
                        and len(coverage["completed_ranges"]) < 128
                    ):
                        coverage["completed_ranges"].append(
                            [completed_start, completed_end]
                        )
                    elif completed_start is not None:
                        coverage["range_detail_truncated"] = True
                    completed_start = completed_end = index[0]
            except Exception as exc:
                coverage["read_failed"] += 1
                entry.update(
                    read="read_failed",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
                coverage["error_count"] += 1
                if len(coverage["errors"]) < 20:
                    coverage["errors"].append(entry)
            coverage["not_scanned"] -= 1
            scanned += 1
            if _fingerprint(key) != fingerprint:
                stable = False
                report.add(
                    "publication.block_changed", "fail", INTEGRITY, observed=index
                )
            if on_block is not None:
                on_block(entry)
            resident = memory.resident_memory_bytes()
            if resident is not None:
                previous = coverage.get("sampled_peak_resident_bytes")
                coverage["sampled_peak_resident_bytes"] = max(resident, previous or 0)
            if resident is not None and resident > memory_budget_mib * 1024**2:
                report.add(
                    "scan.peak_budget",
                    "fail",
                    INTEGRITY,
                    memory_budget_mib * 1024**2,
                    resident,
                )
                coverage["interrupted"] = True
                break
        if mode == "full" and coverage["storage_unknown"] == 0:
            extra_keys = []
            extra_count = 0
            for file in (root / "data").rglob("*"):
                if not file.is_file() or file.name in (
                    "zarr.json",
                    ".zarray",
                    ".zattrs",
                ):
                    continue
                relative = file.relative_to(root / "data").as_posix()
                if not valid_chunk_key(file, metadata, total):
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


def _check_scan_options(mode, sample_blocks, max_blocks, memory_budget_mib,
                        stop_requested, on_block):
    if not isinstance(mode, str) or mode not in ("quick", "full"):
        raise ValueError("mode must be 'quick' or 'full'")
    for name, value in (("sample_blocks", sample_blocks), ("max_blocks", max_blocks)):
        if name == "max_blocks" and value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (isinstance(memory_budget_mib, (bool, np.bool_))
            or not isinstance(memory_budget_mib, Real)
            or not math.isfinite(memory_budget_mib) or memory_budget_mib < 1):
        raise ValueError("memory_budget_mib must be finite and at least 1")
    for name, callback in (("stop_requested", stop_requested), ("on_block", on_block)):
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable or None")


def _read_bound_receipt(reference, label):
    # Hash exactly the bytes that are parsed, rather than reopening after hashing.
    raw = reference.path.read_bytes()
    observed_hash = hashlib.sha256(raw).hexdigest()
    if observed_hash.lower() != reference.sha256.lower():
        raise ValueError(f"{label} SHA256 differs from frozen reference")
    receipt = json.loads(raw)
    if not isinstance(receipt, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return receipt, observed_hash


def _verify_production_evidence(
    spec: DatasetValidationSpec, report: ValidationReport
) -> bool:
    reference = spec.production_evidence
    if reference is None or not spec.snapshot_id:
        return check_production_receipt(spec, report, None)
    try:
        receipt, observed_hash = _read_bound_receipt(reference, "Production receipt")
        return check_production_receipt(spec, report, receipt, observed_hash)
    except (OSError, ValueError, TypeError) as exc:
        report.add(
            "publication.frozen_snapshot",
            "fail",
            "publication_readiness",
            evidence=f"{type(exc).__name__}: {exc}",
        )
        return False


def _check_declared_conversion_evidence(spec, report):
    """Verify supplied evidence and declared requirements, never run a conversion."""
    category = "conversion_fidelity"
    reference = spec.conversion_evidence
    if reference is not None:
        try:
            receipt, _ = _read_bound_receipt(reference, "Conversion receipt")
            attach_conversion_evidence(report, receipt)
        except (OSError, ValueError, TypeError) as exc:
            report.add(
                f"{category}.external_evidence", "fail", category, evidence=str(exc)
            )
    else:
        report.add(
            f"{category}.external_evidence",
            "not-run" if spec.conversion_required else "not-applicable",
            category,
            evidence=(
                "Contract requires an independently generated conversion receipt; no source/target comparison is executed here"
                if spec.conversion_required
                else "Not selected by this instance profile"
            ),
        )


def _validate_dataset(
    path: str | Path,
    spec: DatasetSchema,
    *,
    mode: str = "quick",
    auxiliary_directory: str | Path | None = None,
    sample_blocks: int = 3,
    max_blocks: int | None = None,
    memory_budget_mib: int = 8192,
    stop_requested: Callable[[], bool] | None = None,
    on_block: Callable[[dict[str, Any]], None] | None = None,
) -> ValidationReport:
    """Validate a local published Zarr without repairing or mutating its contents.

    Full reads every independently expected frame and audits its physical key.
    Quick selects deterministic first/interior/last frames; it never authorizes
    release. ``max_blocks`` bounds a diagnostic run and makes full incomplete.
    The caller can stream block evidence and request a cooperative interruption.
    An interrupted report preserves mutually exclusive counters, not a pass.
    """
    if not isinstance(spec, DatasetValidationSpec):
        spec = DatasetValidationSpec.from_schema(spec)
    spec.validate_configuration()
    root = Path(path).expanduser().resolve()
    auxiliary = (
        Path(auxiliary_directory).expanduser().resolve()
        if auxiliary_directory
        else root
    )
    identity = {
        "product_id": spec.product.product_id,
        "product_version": spec.product.version,
        "variant_id": spec.variant_name,
        "contract_id": "xmetai.normalized-grid",
        "contract_version": spec.contract_version,
        "config_id": spec.config_id,
        "template_version": spec.template_version,
        "schema_path": str(spec.schema_path) if spec.schema_path is not None else None,
        "schema_sha256": spec.schema_sha256,
        "execution_config_path": spec.execution_config_path,
        "execution_config_sha256": spec.execution_config_sha256,
        "data_revision": spec.data_revision,
        "snapshot_id": spec.snapshot_id,
        "path": str(root),
        "tool_version": "datakit-standard-validation/1.8",
        "validation_scope": spec.validation_scope,
        "validation_basis": "target_contract_and_declared_rules",
        "source_target_comparison": False,
        "mask_schema_id": spec.product.mask.schema_id,
        "mask_schema_version": spec.product.mask.version,
    }
    report = ValidationReport(spec.dataset_id, mode, identity)
    check_dataset_identity(root.name, spec, report)
    total = spec.expected_shape[0]
    coverage = report.coverage
    coverage.update(
        expected_shape=spec.expected_shape,
        logical_block_shape=spec.expected_chunks,
        logical_blocks_expected=total,
        expected_index_range=[0, total - 1],
        expected_grid_sha256=hashlib.sha256(
            json.dumps([spec.expected_shape, spec.expected_chunks]).encode()
        ).hexdigest(),
        approved_omitted_indices=spec.allowed_omitted,
        value_checked=0,
        read_failed=0,
        not_scanned=total,
        physically_present=0,
        allowed_omitted=0,
        unapproved_missing=0,
        storage_unknown=total,
        value_scan_complete=False,
        storage_audit_complete=False,
        scan_complete=False,
        quick_scan_complete=False,
        interrupted=False,
        completed_ranges=[],
        range_detail_truncated=False,
        errors=[],
        error_count=0,
    )
    started = time.monotonic()
    stable = _verify_production_evidence(spec, report)
    _check_declared_conversion_evidence(spec, report)
    metadata_before = None
    quality = None
    loaded_regions = {}
    try:
        metadata = read_zarr_metadata(root)
        metadata_before = metadata_digest(metadata)
        coverage["metadata_sha256_before"] = metadata_before
        report.checks.extend(check_root_metadata(metadata, spec))
        coverage["raw_encoding"] = metadata["arrays"].get("data", {}).get("raw")
        raw_bytes = (
            math.prod(spec.expected_chunks)
            * np.dtype(spec.arrays["data"].dtype).itemsize
        )
        estimate = (
            3 * raw_bytes
            + math.prod(spec.expected_shape[2:]) * (26 + len(spec.valid_regions))
            + 256 * 1024**2
        )
        coverage.update(
            memory_budget_bytes=memory_budget_mib * 1024**2,
            estimated_peak_bytes=estimate,
            raw_block_bytes=raw_bytes,
        )
        resident_before = memory.resident_memory_bytes()
        coverage["process_resident_before_bytes"] = resident_before
        coverage["sampled_peak_resident_bytes"] = resident_before
        if not report.require(
            "scan.memory_budget",
            estimate <= memory_budget_mib * 1024**2
            and (
                resident_before is None
                or resident_before <= memory_budget_mib * 1024**2
            ),
            INTEGRITY,
            {
                "estimated_peak_bytes": estimate,
                "process_resident_before": resident_before,
            },
            memory_budget_mib * 1024**2,
        ):
            return report
        with ExitStack() as stack:
            dataset = stack.enter_context(
                xr.open_zarr(
                    str(root),
                    consolidated=False,
                    chunks=None,
                    mask_and_scale=False,
                    decode_times=True,
                )
            )
            channel_ok, checks = check_dataset_format(dataset, spec)
            report.checks.extend(checks)
            layout_ok = False
            if channel_ok is not None:
                stats_ok, checks = check_statistics(
                    dataset["mean"].values, dataset["std"].values
                )
                report.checks.extend(checks)
                report.checks.extend(check_mask_labels(dataset, spec))
                mask_ok, checks = check_mask_values(
                    dataset[spec.product.mask.array_name].values, spec
                )
                report.checks.extend(checks)
                check_mask_provenance(spec, report)
                report.checks.extend(
                    check_channel_semantics(
                        dataset, spec, channel_ok=channel_ok, stats_ok=stats_ok
                    )
                )
                check_product_confirmation(spec, report)
                layout_ok = channel_ok and stats_ok and mask_ok
            if "data" not in metadata["arrays"]:
                raise ValueError("Required /data array missing")
            (chunks_ok, supported), checks = check_array_encoding(
                metadata, dataset, spec
            )
            report.checks.extend(checks)
            # Small arrays are also audited: missing physical coordinates/statistics
            # must not silently pass via their fill values.
            for name, raw in metadata["arrays"].items():
                if name == "data":
                    continue
                small_supported, reason = physical_layout_supported(metadata, name)
                if not small_supported:
                    report.add(
                        f"storage.required_array.{name}",
                        "blocked",
                        INTEGRITY,
                        evidence=reason,
                    )
                    continue
                present = True
                for index, _ in expected_zarr_blocks(raw["shape"], raw["chunks"]):
                    present = (
                        zarr_chunk_path(metadata, name, index).is_file() and present
                    )
                report.require(
                    f"storage.required_array.{name}",
                    present,
                    INTEGRITY,
                    "all physical keys present",
                )
            mask_name = spec.product.mask.array_name
            mask = (
                dataset[mask_name].values
                if mask_name in dataset
                and dataset[mask_name].shape
                == spec.product.mask.shape(spec.variant.grid)
                else None
            )
            declared_files = {
                definition.filename for definition in spec.variant.auxiliaries
            }
            root_metadata_files = (
                {"zarr.json"}
                if metadata["zarr_format"] == 3
                else {".zgroup", ".zattrs", ".zmetadata"}
            )
            root_metadata_files.update(spec.additional_metadata_files)
            if auxiliary == root:
                root_metadata_files.update(declared_files)
            undeclared_root = [
                entry.name
                for entry in root.iterdir()
                if (entry.is_file() and entry.name not in root_metadata_files)
                or (entry.is_dir() and entry.name not in metadata["arrays"])
            ]
            undeclared_auxiliary = [
                name
                for name in undeclared_root
                if (root / name).is_file() and (root / name).suffix in (".nc", ".npy")
            ]
            report.checks.extend(
                check_declared_root(
                    spec,
                    expected=root_metadata_files | set(metadata["arrays"]),
                    undeclared=undeclared_root,
                    undeclared_auxiliary=undeclared_auxiliary,
                )
            )
            unexpected = [
                file.name
                for file in auxiliary.iterdir()
                if file.is_file()
                and file.suffix in (".nc", ".npy")
                and file.name not in declared_files
            ]
            report.checks.extend(check_declared_auxiliaries(spec, unexpected))
            for definition in spec.variant.auxiliaries:
                file = auxiliary / definition.filename
                if not spec.product.mask.check_external_masks and (
                    definition.role
                    in ("external_mask", "land_copy", "sea_copy", "sst_mask")
                    or definition.filename
                    in ("mask.npy", "mask.nc", "land_mask.nc", "sst_mask.nc")
                ):
                    report.add(
                        f"auxiliary.{definition.filename}.scope",
                        "not-applicable",
                        STANDARD,
                        evidence="Embedded Zarr mask/mask_channel are authoritative; external mask file was not read or used for acceptance",
                    )
                    continue
                if not file.exists():
                    report.add(
                        f"auxiliary.{definition.filename}.exists",
                        "fail" if definition.required else "not-applicable",
                        STANDARD,
                    )
                    continue
                try:
                    if not file.resolve().is_relative_to(auxiliary):
                        raise ValueError("Auxiliary path escapes release directory")
                    if definition.format == "npy":
                        values = np.load(file, mmap_mode="r", allow_pickle=False)
                        _check_auxiliary(values, definition, spec, mask, report)
                        del values
                    else:
                        with dataset_io.open(file).read(mask_and_scale=False) as obj:
                            _check_auxiliary(obj, definition, spec, mask, report)
                except (OSError, ValueError, KeyError) as exc:
                    report.add(
                        f"auxiliary.{definition.filename}.read",
                        "fail",
                        INTEGRITY,
                        evidence=f"{type(exc).__name__}: {exc}",
                    )
            if not layout_ok or not chunks_ok or not supported:
                report.add(
                    "scan.prerequisites",
                    "blocked",
                    INTEGRITY,
                    evidence="Invalid structure or unaudited storage mapping prevents strict scan; independent auxiliary checks were retained",
                )
                return report
            for name, reference in spec.valid_regions.items():
                if file_sha256(reference.path) != reference.sha256:
                    report.add(
                        f"quality.region.{name}.source",
                        "fail",
                        "data_quality",
                        evidence="Frozen region mask SHA256 mismatch",
                    )
                    return report
                region = np.load(reference.path, mmap_mode="r", allow_pickle=False)
                region_ok, checks = check_region_layout(region, name, spec)
                report.checks.extend(checks)
                if not region_ok:
                    return report
                if getattr(region, "_mmap", None) is not None:
                    stack.callback(region._mmap.close)
                loaded_regions[name] = region
            quality = StandardQualityAccumulator(
                spec,
                dataset["mean"].values,
                dataset["std"].values,
                mask,
                loaded_regions,
            )
            array = stack.enter_context(open_zarr_array_readonly(root, "data"))
            stable = scan_standard_blocks(
                array,
                metadata,
                spec,
                report,
                quality,
                stable=stable,
                sample_blocks=sample_blocks,
                max_blocks=max_blocks,
                memory_budget_mib=memory_budget_mib,
                stop_requested=stop_requested,
                on_block=on_block,
            )
    except KeyboardInterrupt:
        coverage["interrupted"] = True
        report.add(
            "scan.interrupted",
            "not-run",
            INTEGRITY,
            evidence="KeyboardInterrupt; partial report retained",
        )
    except Exception as exc:
        report.add(
            "scan.exception",
            "fail",
            INTEGRITY,
            evidence={"type": type(exc).__name__, "message": str(exc)},
        )
    finally:
        if spec.execution_config_path is not None:
            try:
                verify_execution_unchanged(spec)
                report.add("publication.execution_unchanged", "pass", STANDARD)
            except (OSError, ValueError) as exc:
                report.add(
                    "publication.execution_unchanged",
                    "fail",
                    STANDARD,
                    evidence=str(exc),
                )
        if spec.schema_path is not None:
            try:
                try:
                    verify_schema_unchanged(spec)
                except ValueError:
                    observed_schema = file_sha256(spec.schema_path)
                else:
                    observed_schema = spec.schema_sha256
                report.require(
                    "publication.schema_unchanged",
                    observed_schema == spec.schema_sha256,
                    STANDARD,
                    spec.schema_sha256,
                    observed_schema,
                )
            except OSError as exc:
                report.add(
                    "publication.schema_unchanged", "fail", STANDARD, evidence=str(exc)
                )
        if stop_requested is not None and stop_requested():
            coverage["interrupted"] = True
        coverage["range_detail_limit"] = 128
        if mode == "full":
            count = coverage["value_checked"] + coverage["read_failed"]
            coverage["unscanned_ranges"] = [[count, total - 1]] if count < total else []
        else:
            # Quick selection is recorded explicitly; the ledger distinguishes
            # unread selections if the diagnostic scan was interrupted.
            coverage["unscanned_ranges"] = "Complement of read indices in block ledger"
        if quality is not None:
            checks, metrics = quality.finish()
            report.checks.extend(checks)
            report.quality.update(metrics)
        if coverage["value_checked"] == 0:
            report.add(
                "quality.main_data_scan",
                "not-run",
                "data_quality",
                evidence="No main data blocks checked; auxiliary checks do not certify main data quality",
            )
        try:
            after = metadata_digest(read_zarr_metadata(root))
            coverage["metadata_sha256_after"] = after
            metadata_same = metadata_before is not None and after == metadata_before
            report.require(
                "publication.metadata_unchanged",
                metadata_same,
                INTEGRITY,
                metadata_before,
                after,
            )
        except (OSError, ValueError, KeyError) as exc:
            metadata_same = False
            report.add(
                "publication.metadata_unchanged", "fail", INTEGRITY, evidence=str(exc)
            )
        coverage["value_scan_complete"] = (
            mode == "full" and coverage["value_checked"] == total
        )
        coverage["quality_checks_complete"] = (
            coverage["value_scan_complete"]
            and quality is not None
            and (
                spec.validation_scope == "confirmed-format"
                or len(spec.quality) == len(spec.product.channels)
            )
            and all(state.count > 0 for state in quality.states.values())
        )
        for name in loaded_regions:
            reference = spec.valid_regions[name]
            try:
                same = file_sha256(reference.path) == reference.sha256
            except OSError:
                same = False
            report.require(
                f"metadata.valid_region.{name}.unchanged",
                same,
                INTEGRITY,
                reference.sha256,
            )
        coverage["storage_audit_complete"] = (
            mode == "full"
            and coverage["storage_unknown"] == 0
            and coverage["unapproved_missing"] == 0
        )
        metadata_consistent = not any(
            c.status in ("fail", "blocked")
            and c.check_id.startswith(
                ("metadata.", "storage.required_array.", "storage.no_unexpected_keys")
            )
            for c in report.checks
        )
        coverage["scan_complete"] = (
            coverage["value_scan_complete"]
            and coverage["quality_checks_complete"]
            and coverage["storage_audit_complete"]
            and stable
            and metadata_same
            and metadata_consistent
            and not coverage["interrupted"]
        )
        coverage["quick_scan_complete"] = (
            mode == "quick"
            and isinstance(coverage.get("selection"), list)
            and coverage["value_checked"] == len(coverage["selection"])
            and coverage["read_failed"] == 0
            and quality is not None
            and metadata_same
            and metadata_consistent
            and not coverage["interrupted"]
        )
        coverage["peak_process_memory_bytes"] = memory.peak_memory_bytes()
        coverage["duration_seconds"] = time.monotonic() - started
        coverage["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report.require(
            "storage.no_unapproved_missing",
            coverage["unapproved_missing"] == 0,
            INTEGRITY,
            0,
            coverage["unapproved_missing"],
        )
        report.require(
            "scan.no_read_failures",
            coverage["read_failed"] == 0,
            INTEGRITY,
            0,
            coverage["read_failed"],
        )
        report.add(
            "scan.complete",
            "pass" if coverage["scan_complete"] else "not-run",
            "full_scan",
            True,
            coverage["scan_complete"],
            "Quick or interrupted/unsupported/evidence-incomplete runs cannot authorize release",
        )
    return report


def validate_zarr(
    target, spec, output=None, *, mode="full", auxiliary_directory=None,
    sample_blocks=3, max_blocks=None, memory_budget_mib=8192,
    stop_requested=None, on_block=None,
):
    """Check one target against its contract and declared numerical rules.

    Verify structure, storage, coordinates, statistics, masks and declared values,
    auxiliaries and publication evidence. Optional conversion receipts are only
    verified as evidence; this API never reads source data or executes transforms.
    Full checks all expected frames; quick/max_blocks/interruption cannot certify
    full release. Execution controls must be valid before any input is read.
    output=None writes nothing; .yaml reports cannot overwrite inputs or reports.
    """
    from datakit.schema import load_schema

    from .configuration import DatasetValidationSpec, validation_input_files
    from .reports import ensure_report_destination

    if output is not None and Path(output).suffix != ".yaml":
        raise ValueError("Validation report output must use the .yaml suffix")
    _check_scan_options(mode, sample_blocks, max_blocks, memory_budget_mib,
                        stop_requested, on_block)
    target = Path(target).expanduser().resolve()
    if isinstance(spec, (str, Path)):
        spec, _ = load_schema(spec)
    if not isinstance(spec, DatasetSchema):
        raise TypeError("spec must be a DatasetSchema or a schema YAML path")
    if not isinstance(spec, DatasetValidationSpec):
        spec = DatasetValidationSpec.from_schema(spec)
    destination = None
    if output is not None:
        destination = Path(output).expanduser().absolute()
        auxiliary = (
            Path(auxiliary_directory or target).expanduser().resolve()
        )
        ensure_report_destination(
            destination,
            directories=(target, auxiliary),
            files=validation_input_files(spec),
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite an existing report: {destination}"
            )
    report = _validate_dataset(
        target, spec, mode=mode, auxiliary_directory=auxiliary_directory,
        sample_blocks=sample_blocks, max_blocks=max_blocks,
        memory_budget_mib=memory_budget_mib, stop_requested=stop_requested,
        on_block=on_block,
    )
    if destination is not None:
        report.identity["report_path"] = str(destination)
        content = yaml.safe_dump(
            report.to_dict(), allow_unicode=True, sort_keys=False
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(content)
    return report
