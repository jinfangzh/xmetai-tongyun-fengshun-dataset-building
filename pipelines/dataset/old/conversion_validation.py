"""Streamed source-to-target fidelity checks of explicit local Zarr stores."""

from __future__ import annotations

import json
import math
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import xarray as xr

from datakit.schema.parser import resolve_config_path, verify_schema_unchanged
from datakit.schema.specifications import DatasetSchema, EvidenceReference
from datakit.validation.comparison import compare_channel_values
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

from ..configuration import (
    DatasetValidationSpec,
    load_schema_reference,
    read_frozen_json,
    validation_input_files,
    verify_execution_unchanged,
)
from ..reports import ensure_report_destination, to_jsonable


@dataclass(frozen=True)
class ConversionChannelRule:
    """Approved source-to-target linear mapping, applied to values once."""

    target_channel: str
    source_channel: str
    atol: float
    rtol: float
    value_factor: float = 1.0
    value_offset: float = 0.0
    mean_factor: float = 1.0
    mean_offset: float = 0.0
    std_factor: float = 1.0
    expected_mean: float | None = None
    expected_std: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.atol,
            self.rtol,
            self.value_factor,
            self.value_offset,
            self.mean_factor,
            self.mean_offset,
            self.std_factor,
        )
        if (
            not np.isfinite(values).all()
            or self.atol < 0
            or self.rtol < 0
            or self.std_factor <= 0
        ):
            raise ValueError("Conversion factors/tolerances must be finite and valid")


@dataclass(frozen=True)
class ConversionValidationSpec:
    source_identity: str
    source_snapshot_id: str
    rules_version: str
    source_first_time: str
    source_last_time: str
    source_cadence_hours: int
    source_channels: tuple[str, ...]
    rules: tuple[ConversionChannelRule, ...]
    source_variable: str = "data"
    source_channel_coordinate: str = "channel"
    source_dimensions: tuple[str, ...] = ("time", "channel", "lat", "lon")
    source_mean_variable: str = "mean"
    source_std_variable: str = "std"
    latitude_flip: bool = False
    source_statistics: EvidenceReference | None = None

    def expected_times(self) -> np.ndarray:
        start, end = (
            np.datetime64(self.source_first_time, "ns"),
            np.datetime64(self.source_last_time, "ns"),
        )
        if self.source_cadence_hours <= 0:
            raise ValueError("Invalid source cadence")
        step = np.timedelta64(self.source_cadence_hours, "h").astype("timedelta64[ns]")
        if np.isnat(start) or np.isnat(end) or end < start or (end - start) % step:
            raise ValueError("Invalid independently frozen source coverage")
        return np.arange(start, end + step, step)


def load_conversion_run(
    config_path: Path,
) -> tuple[DatasetValidationSpec, Path, Path, ConversionValidationSpec]:
    config, config_sha256 = read_frozen_json(config_path)
    spec, locations = load_schema_reference(
        config_path,
        config,
        schema_key="target_schema",
        config_sha256=config_sha256,
    )
    source = resolve_config_path(config_path, config["source_path"])
    target = (
        resolve_config_path(config_path, config["target_path"])
        if config.get("target_path")
        else Path(locations["dataset_path"])
    )
    values = dict(config["conversion"])
    values["rules"] = tuple(ConversionChannelRule(**rule) for rule in values["rules"])
    values["source_channels"] = tuple(values["source_channels"])
    if values.get("source_statistics"):
        reference = values["source_statistics"]
        values["source_statistics"] = EvidenceReference(
            resolve_config_path(config_path, reference["path"]),
            reference["sha256"],
        )
    if "source_dimensions" in values:
        values["source_dimensions"] = tuple(values["source_dimensions"])
    return spec, source, target, ConversionValidationSpec(**values)


def check_conversion_configuration(
    spec: DatasetSchema,
    conversion: ConversionValidationSpec,
) -> None:
    if tuple(
        r.target_channel for r in conversion.rules
    ) != spec.product.channel_names or len(set(conversion.source_channels)) != len(
        conversion.source_channels
    ):
        raise ValueError("Rules must cover every ordered target channel exactly once")
    if (
        not conversion.source_identity
        or not conversion.source_snapshot_id
        or not conversion.rules_version
    ):
        raise ValueError("Source identity, snapshot and rules version are required")


def check_coordinates(
    src: xr.Dataset,
    dst: xr.Dataset,
    spec: DatasetSchema,
    conversion: ConversionValidationSpec,
) -> bool:
    lat, lon = spec.variant.grid.coordinates()
    source_times, target_times = conversion.expected_times(), spec.expected_times()
    return bool(
        np.array_equal(src.time.values, source_times)
        and np.array_equal(dst.time.values, target_times)
        and np.array_equal(
            src.lat.values, lat[::-1] if conversion.latitude_flip else lat
        )
        and np.array_equal(src.lon.values, lon)
        and np.array_equal(dst.lat.values, lat)
        and np.array_equal(dst.lon.values, lon)
        and np.isin(target_times, source_times).all()
    )


def check_channels(
    src: xr.Dataset,
    dst: xr.Dataset,
    spec: DatasetSchema,
    conversion: ConversionValidationSpec,
) -> bool:
    return (
        tuple(str(v) for v in src[conversion.source_channel_coordinate].values)
        == conversion.source_channels
        and tuple(str(v) for v in dst.channel.values) == spec.product.channel_names
    )


def check_layout(
    src: xr.Dataset,
    dst: xr.Dataset,
    spec: DatasetSchema,
    conversion: ConversionValidationSpec,
) -> bool:
    lat, lon = spec.variant.grid.coordinates()
    return (
        tuple(src[conversion.source_variable].dims) == conversion.source_dimensions
        and src[conversion.source_variable].shape
        == (
            len(conversion.expected_times()),
            len(conversion.source_channels),
            len(lat),
            len(lon),
        )
        and tuple(dst.data.dims) == ("time", "channel", "lat", "lon")
        and dst.data.shape == spec.expected_shape
    )


def check_statistics(
    src: xr.Dataset,
    dst: xr.Dataset,
    conversion: ConversionValidationSpec,
    indices: list[int],
    statistics: dict[str, Any] | None,
) -> bool:
    if conversion.source_statistics is not None:
        if statistics is None:
            raise ValueError("Frozen source statistics receipt contents are required")
        if (
            statistics.get("source_identity") != conversion.source_identity
            or statistics.get("source_snapshot_id") != conversion.source_snapshot_id
            or tuple(statistics.get("channels", ())) != conversion.source_channels
            or not statistics.get("provenance")
        ):
            raise ValueError(
                "Source statistics receipt identity/channels/provenance mismatch"
            )
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
        for index, rule in enumerate(conversion.rules):
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
    return bool(statistics_ok)


def comparison_complete(
    *,
    mode: str,
    checked: int,
    total: int,
    failed: int,
    storage_ok: bool,
    unchanged: bool,
    interrupted: bool,
) -> bool:
    return (
        mode == "full"
        and checked == total
        and not failed
        and storage_ok
        and unchanged
        and not interrupted
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

    before_source, before_target = (
        read_zarr_metadata(source),
        read_zarr_metadata(target),
    )
    source_digest, target_digest = (
        metadata_digest(before_source),
        metadata_digest(before_target),
    )
    with ExitStack() as stack:
        src = stack.enter_context(
            xr.open_zarr(
                str(source), consolidated=False, chunks=None, mask_and_scale=False
            )
        )
        dst = stack.enter_context(
            xr.open_zarr(
                str(target), consolidated=False, chunks=None, mask_and_scale=False
            )
        )
        lat, lon = target_spec.variant.grid.coordinates()
        source_times, target_times = (
            conversion.expected_times(),
            target_spec.expected_times(),
        )
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
            math.prod(source_chunks)
            * src[conversion.source_variable].dtype.itemsize
            * 3
            + math.prod(target_spec.expected_chunks)
            * np.dtype(target_spec.arrays["data"].dtype).itemsize
            * 3
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
                        zarr_chunk_path(
                            before_target, "data", (int(ti), 0, 0, 0)
                        ).is_file()
                        and storage_ok
                    )
                    sb = read_zarr_block(
                        source_array,
                        (slice(si, si + 1), slice(None), slice(None), slice(None)),
                    )
                    tb = read_zarr_block(
                        target_array,
                        (slice(ti, ti + 1), slice(None), slice(None), slice(None)),
                    )
                    if conversion.latitude_flip:
                        sb = sb[:, :, ::-1, :]
                    for ci, rule in enumerate(rules):
                        result = compare_channel_values(
                            tb[:, ci],
                            sb[:, indices[ci]],
                            value_factor=rule.value_factor,
                            value_offset=rule.value_offset,
                            atol=rule.atol,
                            rtol=rule.rtol,
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
            mode=mode,
            checked=checked,
            total=total,
            failed=failed,
            storage_ok=storage_ok,
            unchanged=unchanged,
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
            destination,
            directories=(source, target),
            files=(
                *validation_input_files(spec),
                conversion.source_statistics.path
                if conversion.source_statistics
                else None,
            ),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".blocks.jsonl").open("w", encoding="utf-8") as stream:

        def record(entry: dict[str, Any]) -> None:
            stream.write(
                json.dumps(to_jsonable(entry), ensure_ascii=False, allow_nan=False)
                + "\n"
            )
            stream.flush()

        receipt = validate_standard_conversion(
            source,
            target,
            spec,
            conversion,
            mode=mode,
            max_blocks=max_blocks,
            memory_budget_mib=memory_budget_mib,
            stop_requested=stop_requested,
            on_block=record,
        )
    verify_schema_unchanged(spec)
    verify_execution_unchanged(spec)
    receipt["config_sha256"] = spec.execution_config_sha256
    output.write_text(
        json.dumps(to_jsonable(receipt), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return receipt
