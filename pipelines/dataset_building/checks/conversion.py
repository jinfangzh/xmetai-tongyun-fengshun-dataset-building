"""Source-to-target correspondence, value fidelity and conversion receipt rules."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from datakit.schema.datasets.specifications import DatasetSchema

from ..support.reports import ValidationReport
from ..support.specifications import ConversionChannelRule, ConversionValidationSpec

CONVERSION_CHECKS = frozenset(
    {"mapping", "coordinates", "channels", "statistics", "values", "coverage"}
)


def compare_channel_values(
    actual: np.ndarray, source: np.ndarray, rule: ConversionChannelRule,
) -> dict:
    """Compare one channel without clipping, filling or broadcasting."""
    if actual.shape != source.shape:
        return {
            "passed": False,
            "reason": "different raw shapes",
            "actual_shape": actual.shape,
            "source_shape": source.shape,
        }
    expected = source.astype("f4")
    expected *= rule.value_factor
    expected += rule.value_offset
    same_missing = (
        np.array_equal(np.isnan(actual), np.isnan(expected))
        and np.array_equal(np.isposinf(actual), np.isposinf(expected))
        and np.array_equal(np.isneginf(actual), np.isneginf(expected))
    )
    finite = np.isfinite(actual) & np.isfinite(expected)
    difference = np.abs(actual[finite].astype("f4") - expected[finite])
    outside = int((difference > rule.atol + rule.rtol * np.abs(expected[finite])).sum())
    return {
        "passed": same_missing and not outside,
        "missing_pattern_equal": same_missing,
        "outside_tolerance": outside,
        "max_absolute_error": float(difference.max()) if difference.size else None,
        "finite_compared": int(finite.sum()),
        "atol": rule.atol,
        "rtol": rule.rtol,
    }


def check_conversion_configuration(
    spec: DatasetSchema, conversion: ConversionValidationSpec,
) -> None:
    if tuple(r.target_channel for r in conversion.rules) != spec.product.channel_names or len(
        set(conversion.source_channels)
    ) != len(conversion.source_channels):
        raise ValueError("Rules must cover every ordered target channel exactly once")
    if (
        not conversion.source_identity
        or not conversion.source_snapshot_id
        or not conversion.rules_version
    ):
        raise ValueError("Source identity, snapshot and rules version are required")


def check_coordinates(
    src: xr.Dataset, dst: xr.Dataset, spec: DatasetSchema, conversion: ConversionValidationSpec,
) -> bool:
    lat, lon = spec.variant.grid.coordinates()
    source_times, target_times = conversion.expected_times(), spec.expected_times()
    return bool(
        np.array_equal(src.time.values, source_times)
        and np.array_equal(dst.time.values, target_times)
        and np.array_equal(src.lat.values, lat[::-1] if conversion.latitude_flip else lat)
        and np.array_equal(src.lon.values, lon)
        and np.array_equal(dst.lat.values, lat)
        and np.array_equal(dst.lon.values, lon)
        and np.isin(target_times, source_times).all()
    )


def check_channels(
    src: xr.Dataset, dst: xr.Dataset, spec: DatasetSchema, conversion: ConversionValidationSpec,
) -> bool:
    return (
        tuple(str(v) for v in src[conversion.source_channel_coordinate].values)
        == conversion.source_channels
        and tuple(str(v) for v in dst.channel.values) == spec.product.channel_names
    )


def check_layout(
    src: xr.Dataset, dst: xr.Dataset, spec: DatasetSchema, conversion: ConversionValidationSpec,
) -> bool:
    lat, lon = spec.variant.grid.coordinates()
    return (
        tuple(src[conversion.source_variable].dims) == conversion.source_dimensions
        and src[conversion.source_variable].shape
        == (len(conversion.expected_times()), len(conversion.source_channels), len(lat), len(lon))
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
        for index, rule in enumerate(conversion.rules):
            mean = (
                rule.expected_mean if rule.expected_mean is not None
                else float(sm[indices[index]]) * rule.mean_factor + rule.mean_offset
            )
            std = (
                rule.expected_std if rule.expected_std is not None
                else float(ss[indices[index]]) * rule.std_factor
            )
            statistics_ok = (
                bool(
                    np.isfinite(mean) and np.isfinite(std) and std > 0
                    and np.isclose(dm[index], mean, atol=rule.atol, rtol=rule.rtol)
                    and np.isclose(ds[index], std, atol=rule.atol, rtol=rule.rtol)
                )
                and statistics_ok
            )
    return bool(statistics_ok)


def comparison_complete(
    *, mode: str, checked: int, total: int, failed: int,
    storage_ok: bool, unchanged: bool, interrupted: bool,
) -> bool:
    return (
        mode == "full" and checked == total and not failed
        and storage_ok and unchanged and not interrupted
    )


def attach_conversion_evidence(
    report: ValidationReport, evidence: dict[str, Any],
) -> None:
    """Check a previously read receipt, not the underlying source/target data."""
    category = "conversion_fidelity"
    required = CONVERSION_CHECKS
    expected = {
        "dataset_id": report.dataset_id,
        "data_revision": report.identity["data_revision"],
        "snapshot_id": report.identity["snapshot_id"],
        "product_id": report.identity["product_id"],
        "product_version": report.identity["product_version"],
        "config_id": report.identity["config_id"],
        "template_version": report.identity["template_version"],
        "schema_sha256": report.identity["schema_sha256"],
    }
    checks = evidence.get("checks", {})
    bound = bool(expected["snapshot_id"]) and all(evidence.get(k) == v for k, v in expected.items())
    report.require(
        f"{category}.identity", bound, category, expected, {k: evidence.get(k) for k in expected}
    )
    if not bound:
        return
    report.require(
        f"{category}.check_coverage",
        isinstance(checks, dict) and set(checks) == required,
        category,
        sorted(required),
        sorted(checks) if isinstance(checks, dict) else checks,
    )
    if not isinstance(checks, dict):
        return
    metadata_fields = ("source_identity", "rules_version", "mapping")
    report.require(
        f"{category}.provenance",
        all(evidence.get(k) for k in metadata_fields),
        category,
        metadata_fields,
        {k: evidence.get(k) for k in metadata_fields},
    )
    report.require(
        "conversion_fidelity.full_comparison",
        evidence.get("comparison_complete") is True,
        category,
        True,
        evidence.get("comparison_complete"),
    )
    for name in sorted(required):
        item = checks.get(name)
        if not isinstance(item, dict):
            report.add(f"{category}.{name}", "not-run", category)
            continue
        status = item.get("status")
        if status not in ("pass", "fail", "needs-review", "blocked", "not-run"):
            status = "needs-review"
        if status == "pass" and not item.get("evidence"):
            status = "needs-review"
        report.add(
            f"{category}.{name}", status, category,
            item.get("expected"), item.get("observed"), item.get("evidence"),
        )
