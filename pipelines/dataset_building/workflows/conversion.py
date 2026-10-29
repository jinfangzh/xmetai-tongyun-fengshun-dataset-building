"""Streamed source-to-target fidelity checks of explicit local Zarr stores."""

from __future__ import annotations

import json
import math
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import xarray as xr

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
from datakit.schema.datasets.inputs import verify_schema_unchanged
from datakit.schema.datasets.specifications import DatasetSchema

from ..checks.conversion import (
    check_channels,
    check_conversion_configuration,
    check_coordinates,
    check_layout,
    check_statistics,
    compare_channel_values,
    comparison_complete,
)
from ..support.inputs import (
    load_conversion_run,
    validation_input_files,
    verify_execution_unchanged,
)
from ..support.reports import ensure_report_destination, to_jsonable
from ..support.specifications import (
    ConversionValidationSpec,
    DatasetValidationSpec,
)


def validate_standard_conversion(
    source: str | Path,
    target: str | Path,
    target_spec: DatasetSchema,
    conversion: ConversionValidationSpec,
    *,
    mode: str = "full",
    sample_blocks: int = 3,
    max_blocks: int | None = None,
    memory_budget_mib: int = 2048,
    on_block: Callable[[dict[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Compare every mapped channel, statistics and specified frame.

    Supports nonsharded local Zarr with explicit four-axis source layout,
    reordering, latitude reversal and approved linear changes. Nonlinear source
    processing requires an independently prepared reference. Snapshots must be
    frozen before this call; it never manufactures production evidence.
    """
    if not isinstance(target_spec, DatasetValidationSpec):
        target_spec = DatasetValidationSpec.from_schema(target_spec)
    target_spec.validate_configuration()
    verify_execution_unchanged(target_spec)
    if target_spec.schema_path is not None or target_spec.schema_sha256 is not None:
        verify_schema_unchanged(target_spec)
    if (
        mode not in ("quick", "full")
        or sample_blocks < 1
        or memory_budget_mib < 1
        or (max_blocks is not None and max_blocks < 1)
    ):
        raise ValueError("Invalid conversion scan options")
    rules = conversion.rules
    check_conversion_configuration(target_spec, conversion)
    indices = [conversion.source_channels.index(r.source_channel) for r in rules]
    receipt = {
        "dataset_id": target_spec.dataset_id,
        "data_revision": target_spec.data_revision,
        "snapshot_id": target_spec.snapshot_id,
        "product_id": target_spec.product.product_id,
        "product_version": target_spec.product.version,
        "config_id": target_spec.config_id,
        "template_version": target_spec.template_version,
        "schema_sha256": target_spec.schema_sha256,
        "source_identity": {
            "id": conversion.source_identity,
            "snapshot_id": conversion.source_snapshot_id,
            "path": str(source),
        },
        "rules_version": conversion.rules_version,
        "mapping": asdict(conversion),
        "comparison_complete": False,
        "checks": {},
    }
    checks = receipt["checks"]

    def check(name, passed, observed=None):
        checks[name] = {
            "status": "pass" if passed else "fail",
            "observed": observed,
            "evidence": "Explicit frozen source/target mapping and tolerances",
        }

    before_source, before_target = read_zarr_metadata(source), read_zarr_metadata(target)
    source_digest, target_digest = metadata_digest(before_source), metadata_digest(before_target)
    with ExitStack() as stack:
        src = stack.enter_context(
            xr.open_zarr(str(source), consolidated=False, chunks=None, mask_and_scale=False)
        )
        dst = stack.enter_context(
            xr.open_zarr(str(target), consolidated=False, chunks=None, mask_and_scale=False)
        )
        lat, lon = target_spec.variant.grid.coordinates()
        source_times, target_times = conversion.expected_times(), target_spec.expected_times()
        coords_ok = check_coordinates(src, dst, target_spec, conversion)
        check("coordinates", coords_ok)
        channels_ok = check_channels(src, dst, target_spec, conversion)
        check("channels", channels_ok)
        layout_ok = check_layout(src, dst, target_spec, conversion)
        source_supported, reason = physical_layout_supported(
            before_source, conversion.source_variable
        )
        target_supported, target_reason = physical_layout_supported(before_target)
        source_chunks = before_source["arrays"][conversion.source_variable]["chunks"]
        estimate = (
            math.prod(source_chunks) * src[conversion.source_variable].dtype.itemsize * 3
            + math.prod(target_spec.expected_chunks) * np.dtype(target_spec.arrays["data"].dtype).itemsize * 3
            + len(lat) * len(lon) * 32
            + 256 * 1024**2
        )
        mapping_ok = (
            layout_ok
            and source_supported
            and target_supported
            and estimate <= memory_budget_mib * 1024**2
        )
        check(
            "mapping",
            mapping_ok,
            {
                "layout": layout_ok,
                "source_support": reason,
                "target_support": target_reason,
                "estimated_peak_bytes": estimate,
            },
        )
        statistics = None
        if conversion.source_statistics is not None:
            reference = conversion.source_statistics
            if file_sha256(reference.path) != reference.sha256:
                raise ValueError("Frozen source statistics receipt SHA256 mismatch")
            with reference.path.open(encoding="utf-8") as stream:
                statistics = json.load(stream)
        statistics_ok = check_statistics(src, dst, conversion, indices, statistics)
        check("statistics", statistics_ok)
        total = len(target_times)
        checked = failed = 0
        values_ok = storage_ok = True
        if coords_ok and channels_ok and mapping_ok:
            source_array = stack.enter_context(
                open_zarr_array_readonly(source, conversion.source_variable)
            )
            target_array = stack.enter_context(open_zarr_array_readonly(target, "data"))
            selected = (
                range(total)
                if mode == "full"
                else np.linspace(0, total - 1, min(sample_blocks, total), dtype=int)
            )
            positions = np.searchsorted(source_times, target_times)
            for ti in selected:
                if stop_requested is not None and stop_requested():
                    break
                if max_blocks is not None and checked + failed >= max_blocks:
                    break
                si = int(positions[ti])
                entry = {"target_index": int(ti), "source_index": si, "channels": {}}
                try:
                    for index, _ in expected_zarr_blocks(
                        (1, len(conversion.source_channels), len(lat), len(lon)),
                        (1, *source_chunks[1:]),
                    ):
                        key = (si // source_chunks[0], *index[1:])
                        storage_ok = (
                            zarr_chunk_path(
                                before_source, conversion.source_variable, key
                            ).is_file()
                            and storage_ok
                        )
                    storage_ok = (
                        zarr_chunk_path(before_target, "data", (int(ti), 0, 0, 0)).is_file()
                        and storage_ok
                    )
                    sb = read_zarr_block(
                        source_array, (slice(si, si + 1), slice(None), slice(None), slice(None))
                    )
                    tb = read_zarr_block(
                        target_array, (slice(ti, ti + 1), slice(None), slice(None), slice(None))
                    )
                    if conversion.latitude_flip:
                        sb = sb[:, :, ::-1, :]
                    for ci, rule in enumerate(rules):
                        result = compare_channel_values(
                            tb[:, ci], sb[:, indices[ci]], rule
                        )
                        entry["channels"][rule.target_channel] = result
                        values_ok = result["passed"] and values_ok
                    del sb, tb
                    checked += 1
                except Exception as exc:
                    failed += 1
                    values_ok = False
                    entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
                if on_block:
                    on_block(entry)
        unchanged = source_digest == metadata_digest(
            read_zarr_metadata(source)
        ) and target_digest == metadata_digest(read_zarr_metadata(target))
        complete = comparison_complete(
            mode=mode, checked=checked, total=total, failed=failed,
            storage_ok=storage_ok, unchanged=unchanged,
            interrupted=False,
        )
        if complete and stop_requested is not None:
            complete = not stop_requested()
        receipt["comparison_complete"] = complete
        check(
            "values",
            values_ok and checked > 0,
            {"checked_blocks": checked, "failed_blocks": failed},
        )
        check(
            "coverage",
            complete,
            {
                "expected": total,
                "checked": checked,
                "failed": failed,
                "not_compared": total - checked - failed,
                "storage_present": storage_ok,
                "metadata_unchanged": unchanged,
            },
        )
    if target_spec.schema_path is not None or target_spec.schema_sha256 is not None:
        verify_schema_unchanged(target_spec)
    verify_execution_unchanged(target_spec)
    return to_jsonable(receipt)


def run_conversion_validation(
    config_path: Path,
    output: Path,
    *,
    mode: str = "full",
    max_blocks: int | None = None,
    memory_budget_mib: int = 2048,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    spec, source, target, conversion = load_conversion_run(config_path)
    for destination in (output, output.with_suffix(".blocks.jsonl")):
        ensure_report_destination(
            destination, directories=(source, target),
            files=(
                *validation_input_files(spec),
                conversion.source_statistics.path if conversion.source_statistics else None,
            ),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".blocks.jsonl").open("w", encoding="utf-8") as stream:
        def record(entry: dict[str, Any]) -> None:
            stream.write(json.dumps(to_jsonable(entry), ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        receipt = validate_standard_conversion(
            source, target, spec, conversion, mode=mode, max_blocks=max_blocks,
            memory_budget_mib=memory_budget_mib, stop_requested=stop_requested, on_block=record,
        )
    verify_schema_unchanged(spec)
    verify_execution_unchanged(spec)
    receipt["config_sha256"] = spec.execution_config_sha256
    output.write_text(
        json.dumps(to_jsonable(receipt), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return receipt
