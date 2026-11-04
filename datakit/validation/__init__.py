"""Reusable in-memory data validation and diagnostics."""

from .comparison import (
    build_statistics_diff,
    collect_structure,
    compare_channels,
    compare_coordinates,
    compare_datasets,
    compute_bias_mean,
    compute_bias_mean_by_channel,
    normalize_channel_dim,
    normalize_core_dims,
    select_data_array,
    summarize_statistics,
)
from .integrity import (
    RangeReport,
    analyze_range,
    check_complete_year,
    count_invalid,
    describe_data,
)
from .check_result import CheckResult
from .structure import (
    check_channels,
    check_coordinates,
    check_dimensions,
    check_same_layout,
)

__all__ = [
    "check_dimensions",
    "check_coordinates",
    "check_channels",
    "check_same_layout",
    "CheckResult",
    "compare_datasets",
    "RangeReport",
    "analyze_range",
    "build_statistics_diff",
    "check_complete_year",
    "collect_structure",
    "compare_channels",
    "compare_coordinates",
    "compute_bias_mean",
    "compute_bias_mean_by_channel",
    "count_invalid",
    "describe_data",
    "normalize_channel_dim",
    "normalize_core_dims",
    "select_data_array",
    "summarize_statistics",
]
