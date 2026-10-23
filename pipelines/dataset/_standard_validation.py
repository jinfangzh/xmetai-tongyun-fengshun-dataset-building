"""Read-only path orchestration for independently specified dataset validation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import xarray as xr

from datakit import io as dataset_io
from datakit.validation.zarr_inspection import (
    consolidated_mismatches,
    expected_zarr_blocks,
    file_sha256,
    metadata_digest,
    open_zarr_array_readonly,
    physical_layout_supported,
    read_zarr_metadata,
    zarr_chunk_path,
)
from datakit.schema.standard_dataset import DatasetValidationSpec
from datakit.validation._standard_auxiliary import check_standard_auxiliary
from datakit.validation._standard_quality import StandardQualityAccumulator
from datakit.validation.evidence import attach_standard_evidence
from datakit.validation.reports import ValidationReport
from datakit.validation.standard_dataset import STANDARD, check_standard_dataset

from ._standard_scan import scan_standard_blocks
from ._validation_runtime import _peak_memory_bytes, _resident_memory_bytes

INTEGRITY = "data_integrity"


def _verify_production_evidence(spec: DatasetValidationSpec, report: ValidationReport) -> bool:
    reference = spec.production_evidence
    if reference is None or not spec.snapshot_id:
        report.add(
            "publication.frozen_snapshot",
            "not-run" if report.mode == "quick" and not spec.allowed_omitted else "needs-review",
            "publication_readiness",
            evidence="Require independently frozen production receipt and stopped writes/immutable snapshot",
        )
        return False
    try:
        observed_hash = file_sha256(reference.path)
        if observed_hash.lower() != reference.sha256.lower():
            raise ValueError("Production receipt SHA256 differs from independently frozen value")
        with reference.path.open(encoding="utf-8") as stream:
            receipt = json.load(stream)
        expected = {
            "dataset_id": spec.dataset_id,
            "data_revision": spec.data_revision,
            "snapshot_id": spec.snapshot_id,
            "shape": list(spec.expected_shape),
            "logical_blocks_expected": spec.expected_shape[0],
            "complete": True,
            "allowed_omitted": [list(index) for index in spec.allowed_omitted],
        }
        good = all(receipt.get(key) == value for key, value in expected.items()) and receipt.get(
            "stability"
        ) in ("immutable-snapshot", "writes-stopped")
        return report.require(
            "publication.frozen_snapshot",
            good,
            "publication_readiness",
            expected,
            receipt,
            {
                "path": str(reference.path),
                "sha256": observed_hash,
                "limitation": "Producer attests stability; metadata digest alone cannot prove absence of concurrent value writes",
            },
        )
    except (OSError, ValueError, TypeError) as exc:
        report.add(
            "publication.frozen_snapshot",
            "fail",
            "publication_readiness",
            evidence=f"{type(exc).__name__}: {exc}",
        )
        return False


def validate_standard_dataset(
    path: str | Path,
    spec: DatasetValidationSpec,
    *,
    mode: str = "quick",
    auxiliary_directory: str | Path | None = None,
    sample_blocks: int = 3,
    max_blocks: int | None = None,
    memory_budget_mib: int = 2048,
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
    spec.validate_configuration()
    if (
        mode not in ("quick", "full")
        or sample_blocks < 1
        or memory_budget_mib < 1
        or (max_blocks is not None and max_blocks < 1)
    ):
        raise ValueError("Invalid mode, sample count, maximum blocks or memory budget")
    root = Path(path).expanduser().resolve()
    auxiliary = Path(auxiliary_directory).expanduser().resolve() if auxiliary_directory else root
    identity = {
        "product_id": spec.product.product_id,
        "product_version": spec.product.version,
        "variant_id": spec.variant_name,
        "contract_id": "xmetai.normalized-grid",
        "contract_version": spec.contract_version,
        "profile_version": spec.profile_version,
        "data_revision": spec.data_revision,
        "snapshot_id": spec.snapshot_id,
        "path": str(root),
        "tool_version": "datakit-standard-validation/1.4",
        "validation_scope": spec.validation_scope,
        "mask_schema_id": spec.product.mask.schema_id,
        "mask_schema_version": spec.product.mask.version,
    }
    report = ValidationReport(spec.dataset_id, mode, identity)
    if spec.directory_name:
        report.require(
            "publication.directory_name",
            root.name == spec.directory_name,
            STANDARD,
            spec.directory_name,
            root.name,
        )
        report.add(
            "publication.identity_mapping",
            (
                "pass"
                if spec.directory_name == spec.dataset_id or spec.identity_mapping_evidence
                else "needs-review"
            ),
            STANDARD,
            spec.dataset_id,
            spec.directory_name,
            spec.identity_mapping_evidence,
        )
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
    for category, required, reference in (
        ("consumer_compatibility", spec.consumer_required, spec.consumer_evidence),
        ("conversion_fidelity", spec.conversion_required, spec.conversion_evidence),
    ):
        if reference is not None:
            try:
                if file_sha256(reference.path).lower() != reference.sha256.lower():
                    raise ValueError("External receipt SHA256 differs from frozen reference")
                with reference.path.open(encoding="utf-8") as stream:
                    attach_standard_evidence(report, json.load(stream), category=category)
            except (OSError, ValueError, TypeError) as exc:
                report.add(f"{category}.external_evidence", "fail", category, evidence=str(exc))
        else:
            report.add(
                f"{category}.external_evidence",
                "not-run" if required else "not-applicable",
                category,
                evidence=(
                    "Required consumer/source checks must be run independently and attached; Zarr readability does not certify a model"
                    if required
                    else "Not selected by this instance profile"
                ),
            )
    report.add(
        "combination.scope",
        "not-applicable",
        "cross_dataset_compatibility",
        evidence="Single-dataset validation; use explicit combination checks before joint consumption",
    )
    metadata_before = None
    quality = None
    loaded_regions = {}
    try:
        metadata = read_zarr_metadata(root)
        metadata_before = metadata_digest(metadata)
        coverage["metadata_sha256_before"] = metadata_before
        report.require(
            "storage.zarr_format",
            metadata["zarr_format"] == spec.zarr_format,
            STANDARD,
            spec.zarr_format,
            metadata["zarr_format"],
        )
        report.require(
            "layout.no_nested_groups",
            not metadata["nested_groups"],
            STANDARD,
            [],
            metadata["nested_groups"],
        )
        mismatches = consolidated_mismatches(metadata)
        report.add(
            "metadata.consolidated_matches_raw",
            "not-applicable" if mismatches is None else "fail" if mismatches else "pass",
            INTEGRITY,
            [],
            mismatches,
            "Independent raw children compared against inline/.zmetadata",
        )
        coverage["raw_encoding"] = metadata["arrays"].get("data", {}).get("raw")
        raw_bytes = math.prod(spec.expected_chunks) * np.dtype("f2").itemsize
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
        resident_before = _resident_memory_bytes()
        coverage["process_resident_before_bytes"] = resident_before
        coverage["sampled_peak_resident_bytes"] = resident_before
        if not report.require(
            "scan.memory_budget",
            estimate <= memory_budget_mib * 1024**2
            and (resident_before is None or resident_before <= memory_budget_mib * 1024**2),
            INTEGRITY,
            {"estimated_peak_bytes": estimate, "process_resident_before": resident_before},
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
            layout_ok = check_standard_dataset(dataset, spec, report)
            if "data" not in metadata["arrays"]:
                raise ValueError("Required /data array missing")
            item = metadata["arrays"]["data"]
            chunks_ok = report.require(
                "storage.data_chunks",
                item["chunks"] == spec.expected_chunks,
                STANDARD,
                spec.expected_chunks,
                item["chunks"],
            )
            for name, raw in metadata["arrays"].items():
                if name in dataset:
                    report.require(
                        f"storage.dimensions.{name}",
                        raw["dimensions"] == tuple(dataset[name].dims),
                        STANDARD,
                        tuple(dataset[name].dims),
                        raw["dimensions"],
                    )
                    legacy_dimensions = raw["attributes"].get("_ARRAY_DIMENSIONS")
                    if metadata["zarr_format"] == 3 and legacy_dimensions is not None:
                        report.require(
                            f"metadata.legacy_dimensions.{name}",
                            tuple(legacy_dimensions) == raw["dimensions"],
                            STANDARD,
                            raw["dimensions"],
                            legacy_dimensions,
                        )
            supported, support_reason = physical_layout_supported(metadata)
            report.add(
                "storage.physical_support",
                "pass" if supported else "blocked",
                INTEGRITY,
                "audited nonsharded directory",
                support_reason,
            )
            # Small arrays are also audited: missing physical coordinates/statistics
            # must not silently pass via their fill values.
            for name, raw in metadata["arrays"].items():
                if name == "data":
                    continue
                small_supported, reason = physical_layout_supported(metadata, name)
                if not small_supported:
                    report.add(
                        f"storage.required_array.{name}", "blocked", INTEGRITY, evidence=reason
                    )
                    continue
                present = True
                for index, _ in expected_zarr_blocks(raw["shape"], raw["chunks"]):
                    present = zarr_chunk_path(metadata, name, index).is_file() and present
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
                and dataset[mask_name].shape == spec.product.mask.shape(spec.variant.grid)
                else None
            )
            declared_files = {definition.filename for definition in spec.variant.auxiliaries}
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
            report.add(
                "layout.declared_root_objects",
                (
                    "fail"
                    if set(undeclared_root) - set(undeclared_auxiliary)
                    else "warning" if undeclared_auxiliary else "pass"
                ),
                STANDARD,
                sorted(root_metadata_files | set(metadata["arrays"])),
                undeclared_root,
                "Unregistered NC/NPY auxiliaries are potential issues, not acceptance blockers; undeclared main arrays/other objects remain failures",
            )
            unexpected = [
                file.name
                for file in auxiliary.iterdir()
                if file.is_file()
                and file.suffix in (".nc", ".npy")
                and file.name not in declared_files
            ]
            report.add(
                "auxiliary.declared_files",
                "warning" if unexpected else "pass",
                STANDARD,
                sorted(declared_files),
                unexpected,
                "Unregistered auxiliaries were not used as approved roles and need separate review",
            )
            for definition in spec.variant.auxiliaries:
                file = auxiliary / definition.filename
                if not spec.product.mask.check_external_masks and (
                    definition.role in ("external_mask", "land_copy", "sea_copy", "sst_mask")
                    or definition.filename in ("mask.npy", "mask.nc", "land_mask.nc", "sst_mask.nc")
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
                        check_standard_auxiliary(values, definition, spec, mask, report)
                        del values
                    else:
                        with dataset_io.open(file).read(mask_and_scale=False) as obj:
                            check_standard_auxiliary(obj, definition, spec, mask, report)
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
                valid = (
                    isinstance(region, np.ndarray)
                    and region.shape == spec.variant.grid.shape
                    and region.dtype == np.dtype("bool")
                )
                if not report.require(
                    f"quality.region.{name}.layout",
                    valid,
                    "data_quality",
                    (spec.variant.grid.shape, "bool"),
                    (getattr(region, "shape", None), str(getattr(region, "dtype", None))),
                    spec.valid_region_provenance[name],
                ):
                    return report
                if getattr(region, "_mmap", None) is not None:
                    stack.callback(region._mmap.close)
                loaded_regions[name] = region
            quality = StandardQualityAccumulator(
                spec, dataset["mean"].values, dataset["std"].values, mask, loaded_regions
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
            quality.finish(report)
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
                "publication.metadata_unchanged", metadata_same, INTEGRITY, metadata_before, after
            )
        except (OSError, ValueError, KeyError) as exc:
            metadata_same = False
            report.add("publication.metadata_unchanged", "fail", INTEGRITY, evidence=str(exc))
        coverage["value_scan_complete"] = mode == "full" and coverage["value_checked"] == total
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
                f"metadata.valid_region.{name}.unchanged", same, INTEGRITY, reference.sha256
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
        coverage["peak_process_memory_bytes"] = _peak_memory_bytes()
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
