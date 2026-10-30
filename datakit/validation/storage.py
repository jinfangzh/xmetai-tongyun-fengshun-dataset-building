"""Zarr encoding and physical block integrity checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from datakit.schema.specifications import (
    DatasetSchema,
)

from .check_result import CheckResult, add_check, require_check

STANDARD = "standard_conformance"

from pathlib import Path

from .zarr_inspection import (
    consolidated_mismatches,
    physical_layout_supported,
    zarr_chunk_path,
)

INTEGRITY = "data_integrity"


def check_root_metadata(
    metadata: dict[str, Any], spec: DatasetSchema
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    require_check(
        checks,
        "storage.zarr_format",
        metadata["zarr_format"] == spec.zarr_format,
        STANDARD,
        spec.zarr_format,
        metadata["zarr_format"],
    )
    require_check(
        checks,
        "layout.no_nested_groups",
        not metadata["nested_groups"],
        STANDARD,
        [],
        metadata["nested_groups"],
    )
    mismatches = consolidated_mismatches(metadata)
    add_check(
        checks,
        "metadata.consolidated_matches_raw",
        "not-applicable" if mismatches is None else "fail" if mismatches else "pass",
        INTEGRITY,
        [],
        mismatches,
        "Independent raw children compared against inline/.zmetadata",
    )
    return checks


def check_array_encoding(
    metadata: dict[str, Any], dataset: xr.Dataset, spec: DatasetSchema
) -> tuple[tuple[bool, bool], list[CheckResult]]:
    checks: list[CheckResult] = []
    item = metadata["arrays"]["data"]
    chunks_ok = require_check(
        checks,
        "storage.data_chunks",
        item["chunks"] == spec.expected_chunks,
        STANDARD,
        spec.expected_chunks,
        item["chunks"],
    )
    for name, definition in spec.arrays.items():
        if (
            name != "data"
            and definition.chunks is not None
            and (name in metadata["arrays"])
        ):
            expected_shape = spec.array_layouts()[name][1]
            chunks = tuple(
                (
                    size if value == "full" else value
                    for size, value in zip(
                        expected_shape, definition.chunks, strict=True
                    )
                )
            )
            require_check(
                checks,
                f"storage.{name}_chunks",
                metadata["arrays"][name]["chunks"] == chunks,
                STANDARD,
                chunks,
                metadata["arrays"][name]["chunks"],
            )
    for name, raw in metadata["arrays"].items():
        if name in dataset:
            require_check(
                checks,
                f"storage.dimensions.{name}",
                raw["dimensions"] == tuple(dataset[name].dims),
                STANDARD,
                tuple(dataset[name].dims),
                raw["dimensions"],
            )
            legacy_dimensions = raw["attributes"].get("_ARRAY_DIMENSIONS")
            if metadata["zarr_format"] == 3 and legacy_dimensions is not None:
                require_check(
                    checks,
                    f"metadata.legacy_dimensions.{name}",
                    tuple(legacy_dimensions) == raw["dimensions"],
                    STANDARD,
                    raw["dimensions"],
                    legacy_dimensions,
                )
    supported, support_reason = physical_layout_supported(metadata)
    add_check(
        checks,
        "storage.physical_support",
        "pass" if supported else "blocked",
        INTEGRITY,
        "audited nonsharded directory",
        support_reason,
    )
    return ((chunks_ok, supported), checks)


def block_storage_status(
    *,
    present: bool,
    approved: bool,
    stable: bool,
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
    file: Path,
    metadata: dict[str, Any],
    total: int,
) -> bool:
    root = Path(metadata["path"])
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
        "separator",
        "/" if encoding["name"] == "default" else ".",
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
