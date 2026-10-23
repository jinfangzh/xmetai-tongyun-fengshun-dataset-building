"""Streamed conversion and joint-use checks of explicit local Zarr stores."""

from __future__ import annotations

import json
import math
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import xarray as xr

from datakit.schema.standard_dataset import (
    ConversionValidationSpec,
    DatasetValidationSpec,
)
from datakit.validation import to_jsonable
from datakit.validation.standard_comparison import compare_standard_channel_values
from datakit.validation.standard_dataset import check_standard_time_alignment
from datakit.validation.zarr_inspection import (
    expected_zarr_blocks,
    file_sha256,
    metadata_digest,
    open_zarr_array_readonly,
    physical_layout_supported,
    read_zarr_metadata,
    zarr_chunk_path,
)


def validate_standard_conversion(
    source: str | Path,
    target: str | Path,
    target_spec: DatasetValidationSpec,
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
    target_spec.validate_configuration()
    if (
        mode not in ("quick", "full")
        or sample_blocks < 1
        or memory_budget_mib < 1
        or (max_blocks is not None and max_blocks < 1)
    ):
        raise ValueError("Invalid conversion scan options")
    rules = conversion.rules
    if tuple(r.target_channel for r in rules) != target_spec.product.channel_names or len(
        set(conversion.source_channels)
    ) != len(conversion.source_channels):
        raise ValueError("Rules must cover every ordered target channel exactly once")
    if (
        not conversion.source_identity
        or not conversion.source_snapshot_id
        or not conversion.rules_version
    ):
        raise ValueError("Source identity, snapshot and rules version are required")
    indices = [conversion.source_channels.index(r.source_channel) for r in rules]
    receipt = {
        "dataset_id": target_spec.dataset_id,
        "data_revision": target_spec.data_revision,
        "snapshot_id": target_spec.snapshot_id,
        "product_id": target_spec.product.product_id,
        "product_version": target_spec.product.version,
        "profile_version": target_spec.profile_version,
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
        coords_ok = (
            np.array_equal(src.time.values, source_times)
            and np.array_equal(dst.time.values, target_times)
            and np.array_equal(src.lat.values, lat[::-1] if conversion.latitude_flip else lat)
            and np.array_equal(src.lon.values, lon)
            and np.array_equal(dst.lat.values, lat)
            and np.array_equal(dst.lon.values, lon)
            and np.isin(target_times, source_times).all()
        )
        check("coordinates", coords_ok)
        channels_ok = (
            tuple(str(v) for v in src[conversion.source_channel_coordinate].values)
            == conversion.source_channels
            and tuple(str(v) for v in dst.channel.values) == target_spec.product.channel_names
        )
        check("channels", channels_ok)
        layout_ok = (
            tuple(src[conversion.source_variable].dims) == conversion.source_dimensions
            and src[conversion.source_variable].shape
            == (len(source_times), len(conversion.source_channels), len(lat), len(lon))
            and tuple(dst.data.dims) == ("time", "channel", "lat", "lon")
            and dst.data.shape == target_spec.expected_shape
        )
        source_supported, reason = physical_layout_supported(
            before_source, conversion.source_variable
        )
        target_supported, target_reason = physical_layout_supported(before_target)
        source_chunks = before_source["arrays"][conversion.source_variable]["chunks"]
        estimate = (
            math.prod(source_chunks) * src[conversion.source_variable].dtype.itemsize * 3
            + math.prod(target_spec.expected_chunks) * 2 * 3
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
        if conversion.source_statistics is not None:
            reference = conversion.source_statistics
            if file_sha256(reference.path) != reference.sha256:
                raise ValueError("Frozen source statistics receipt SHA256 mismatch")
            with reference.path.open(encoding="utf-8") as stream:
                statistics = json.load(stream)
            if (
                statistics.get("source_identity") != conversion.source_identity
                or statistics.get("source_snapshot_id") != conversion.source_snapshot_id
                or tuple(statistics.get("channels", ())) != conversion.source_channels
                or not statistics.get("provenance")
            ):
                raise ValueError("Source statistics receipt identity/channels/provenance mismatch")
            sm, ss = np.asarray(statistics["mean"]), np.asarray(statistics["std"])
            statistics_ok = (
                sm.shape == ss.shape == (len(conversion.source_channels),)
                and np.isfinite(sm).all()
                and np.isfinite(ss).all()
                and (ss > 0).all()
            )
        else:
            statistics_ok = all(
                key in src
                and src[key].dims == (conversion.source_channel_coordinate,)
                and src[key].shape == (len(conversion.source_channels),)
                for key in (conversion.source_mean_variable, conversion.source_std_variable)
            )
            if statistics_ok:
                sm, ss = (
                    src[conversion.source_mean_variable].values,
                    src[conversion.source_std_variable].values,
                )
        if statistics_ok:
            dm, ds = dst["mean"].values, dst["std"].values
            for index, rule in enumerate(rules):
                mean = (
                    rule.expected_mean
                    if rule.expected_mean is not None
                    else float(sm[indices[index]]) * rule.mean_factor + rule.mean_offset
                )
                std = (
                    rule.expected_std
                    if rule.expected_std is not None
                    else float(ss[indices[index]]) * rule.std_factor
                )
                statistics_ok = (
                    bool(
                        np.isfinite(mean)
                        and np.isfinite(std)
                        and std > 0
                        and np.isclose(dm[index], mean, atol=rule.atol, rtol=rule.rtol)
                        and np.isclose(ds[index], std, atol=rule.atol, rtol=rule.rtol)
                    )
                    and statistics_ok
                )
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
                    sb = np.asarray(source_array[(slice(si, si + 1), slice(None), slice(None), slice(None))])
                    tb = np.asarray(target_array[(slice(ti, ti + 1), slice(None), slice(None), slice(None))])
                    if conversion.latitude_flip:
                        sb = sb[:, :, ::-1, :]
                    for ci, rule in enumerate(rules):
                        result = compare_standard_channel_values(
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
        complete = (
            mode == "full"
            and checked == total
            and not failed
            and storage_ok
            and unchanged
            and not (stop_requested and stop_requested())
        )
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
    return to_jsonable(receipt)


def validate_standard_combination(
    paths: list[str | Path],
    specs: list[DatasetValidationSpec],
    *,
    policy: str = "exact",
    allow_intersection_loss: bool = False,
    shared_arrays: tuple[str, ...] = (),
    individual_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check original grids/time axes and explicitly shared auxiliary roles."""
    if (
        len(paths) != len(specs)
        or len(paths) < 2
        or set(shared_arrays) - {"mask", "mean", "std", "mask_channel"}
    ):
        raise ValueError("At least two matching paths/specs and known shared arrays required")
    with ExitStack() as stack:
        datasets = [
            stack.enter_context(
                xr.open_zarr(str(p), consolidated=False, chunks=None, mask_and_scale=False)
            )
            for p in paths
        ]
        alignment = check_standard_time_alignment([d.time.values for d in datasets], policy=policy)
        grid = all(
            np.array_equal(datasets[0].lat.values, d.lat.values)
            and np.array_equal(datasets[0].lon.values, d.lon.values)
            for d in datasets[1:]
        )
        channels = [tuple(str(v) for v in d.channel.values) for d in datasets]
        channels_ok = all(
            names == spec.product.channel_names for names, spec in zip(channels, specs)
        )
        shared = {
            name: all(
                (
                    np.array_equal(datasets[0][name].values, d[name].values, equal_nan=True)
                    if datasets[0][name].dtype.kind == "f"
                    else np.array_equal(datasets[0][name].values, d[name].values)
                )
                for d in datasets[1:]
            )
            for name in shared_arrays
        }
        reports = individual_reports or []
        prerequisites = len(reports) == len(specs) and all(
            r.get("release_passed")
            and r.get("dataset_id") == s.dataset_id
            and r.get("identity", {}).get("path") == str(Path(p).expanduser().resolve())
            and r.get("identity", {}).get("snapshot_id") == s.snapshot_id
            and r.get("identity", {}).get("data_revision") == s.data_revision
            and r.get("coverage", {}).get("metadata_sha256_after")
            == metadata_digest(read_zarr_metadata(p))
            for r, s, p in zip(reports, specs, paths)
        )
        loss_ok = allow_intersection_loss or not any(alignment["lost_counts"])
        return {
            "passed": bool(
                prerequisites
                and grid
                and channels_ok
                and alignment["passed"]
                and loss_ok
                and all(shared.values())
            ),
            "individual_release_passed": prerequisites,
            "grid_equal": grid,
            "channels_exact": channels_ok,
            "concatenated_channels": sum((list(c) for c in channels), []),
            "time_alignment": alignment,
            "intersection_loss_approved": allow_intersection_loss,
            "shared_arrays": shared,
            "independent_auxiliary_roles": [s.auxiliary_provenance for s in specs],
        }
