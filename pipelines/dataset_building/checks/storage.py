"""Rules for raw encoding, physical coverage and frozen production evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from datakit.validation.zarr_inspection import (
    consolidated_mismatches,
    physical_layout_supported,
    zarr_chunk_path,
)
from datakit.schema.datasets.specifications import DatasetSchema

from ..support.reports import ValidationReport
from ..support.specifications import DatasetValidationSpec

STANDARD = "standard_conformance"
INTEGRITY = "data_integrity"


def check_root_metadata(
    metadata: dict[str, Any], spec: DatasetSchema, report: ValidationReport,
) -> None:
    report.require(
        "storage.zarr_format", metadata["zarr_format"] == spec.zarr_format,
        STANDARD, spec.zarr_format, metadata["zarr_format"],
    )
    report.require(
        "layout.no_nested_groups", not metadata["nested_groups"],
        STANDARD, [], metadata["nested_groups"],
    )
    mismatches = consolidated_mismatches(metadata)
    report.add(
        "metadata.consolidated_matches_raw",
        "not-applicable" if mismatches is None else "fail" if mismatches else "pass",
        INTEGRITY, [], mismatches,
        "Independent raw children compared against inline/.zmetadata",
    )


def check_array_encoding(
    metadata: dict[str, Any],
    dataset: xr.Dataset,
    spec: DatasetSchema,
    report: ValidationReport,
) -> tuple[bool, bool]:
    item = metadata["arrays"]["data"]
    chunks_ok = report.require(
        "storage.data_chunks", item["chunks"] == spec.expected_chunks,
        STANDARD, spec.expected_chunks, item["chunks"],
    )
    for name, definition in spec.arrays.items():
        if name != "data" and definition.chunks is not None and name in metadata["arrays"]:
            expected_shape = spec.array_layouts()[name][1]
            chunks = tuple(
                size if value == "full" else value
                for size, value in zip(expected_shape, definition.chunks, strict=True)
            )
            report.require(
                f"storage.{name}_chunks", metadata["arrays"][name]["chunks"] == chunks,
                STANDARD, chunks, metadata["arrays"][name]["chunks"],
            )
    for name, raw in metadata["arrays"].items():
        if name in dataset:
            report.require(
                f"storage.dimensions.{name}", raw["dimensions"] == tuple(dataset[name].dims),
                STANDARD, tuple(dataset[name].dims), raw["dimensions"],
            )
            legacy_dimensions = raw["attributes"].get("_ARRAY_DIMENSIONS")
            if metadata["zarr_format"] == 3 and legacy_dimensions is not None:
                report.require(
                    f"metadata.legacy_dimensions.{name}",
                    tuple(legacy_dimensions) == raw["dimensions"],
                    STANDARD, raw["dimensions"], legacy_dimensions,
                )
    supported, support_reason = physical_layout_supported(metadata)
    report.add(
        "storage.physical_support", "pass" if supported else "blocked", INTEGRITY,
        "audited nonsharded directory", support_reason,
    )
    return chunks_ok, supported


def block_storage_status(
    *, present: bool, approved: bool, stable: bool,
) -> str:
    if present:
        return "physically_present"
    if approved and stable:
        return "allowed_omitted"
    return "unapproved_missing"


def check_raw_block(block: np.ndarray, shape: tuple[int, ...], dtype: str) -> None:
    if block.shape != shape or block.dtype != np.dtype(dtype):
        raise ValueError(f"Raw block shape/dtype differs: {block.shape}, {block.dtype}")


def valid_chunk_key(
    file: Path, metadata: dict[str, Any], total: int,
) -> bool:
    root = Path(metadata["path"])
    relative = file.relative_to(root / "data").as_posix()
    raw = metadata["arrays"]["data"]["raw"]
    encoding = raw.get(
        "chunk_key_encoding",
        {"name": "v2", "configuration": {"separator": raw.get("dimension_separator", ".")}},
    )
    separator = encoding.get("configuration", {}).get(
        "separator", "/" if encoding["name"] == "default" else ".",
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
    return good


def check_production_receipt(
    spec: DatasetValidationSpec,
    report: ValidationReport,
    receipt: dict[str, Any] | None,
    observed_hash: str | None = None,
) -> bool:
    reference = spec.production_evidence
    if reference is None or not spec.snapshot_id:
        report.add(
            "publication.frozen_snapshot",
            "not-run" if report.mode == "quick" and not spec.allowed_omitted else "needs-review",
            "publication_readiness",
            evidence="Require independently frozen production receipt and stopped writes/immutable snapshot",
        )
        return False
    if receipt is None:
        raise ValueError("Production receipt contents are required")
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
        "publication.frozen_snapshot", good, "publication_readiness", expected, receipt,
        {
            "path": str(reference.path), "sha256": observed_hash,
            "limitation": "Producer attests stability; metadata digest alone cannot prove absence of concurrent value writes",
        },
    )
