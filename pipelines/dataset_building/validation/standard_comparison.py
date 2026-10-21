"""Strict numeric comparisons with explicit mappings and frozen tolerances."""

from __future__ import annotations

import numpy as np

from ..definitions.standard_dataset import ConversionChannelRule


def compare_standard_channel_values(
    actual: np.ndarray, source: np.ndarray, rule: ConversionChannelRule
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
