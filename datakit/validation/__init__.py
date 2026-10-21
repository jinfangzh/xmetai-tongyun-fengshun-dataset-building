"""Reusable in-memory data validation and diagnostics."""

from .comparison import (
    build_statistics_diff,
    compare_channels,
    compare_coordinates,
    compute_bias_mean,
    compute_bias_mean_by_channel,
    denormalize_zarr_array,
    collect_structure,
    load_channel_statistic_from_zarr,
    normalize_channel_dim,
    normalize_core_dims,
    save_report_json,
    select_data_array,
    summarize_statistics,
    validate_channel_values,
    validate_coordinate_values,
    validate_same_layout,
    validate_structure,
)
from .era5 import (
    validate_era5_layout,
    validate_era5_dataset_schema,
    expected_six_hourly_times,
    validate_era5_c116_layout,
    validate_six_hourly_day,
)
from .integrity import (
    RangeReport,
    analyze_range,
    check_complete_year,
    count_invalid,
    describe_data,
)
from .schema import validate_weather_array
from .reports import CheckResult, ValidationReport, to_jsonable
from .standard_dataset import check_standard_dataset, check_standard_time_alignment
from .standard_comparison import compare_standard_channel_values
from .evidence import attach_standard_evidence

__all__ = [
    "RangeReport",
    "CheckResult",
    "ValidationReport",
    "to_jsonable",
    "check_standard_dataset",
    "check_standard_time_alignment",
    "compare_standard_channel_values",
    "attach_standard_evidence",
    "analyze_range",
    "build_statistics_diff",
    "check_complete_year",
    "collect_structure",
    "compare_channels",
    "compare_coordinates",
    "compute_bias_mean",
    "compute_bias_mean_by_channel",
    "count_invalid",
    "denormalize_zarr_array",
    "describe_data",
    "expected_six_hourly_times",
    "load_channel_statistic_from_zarr",
    "normalize_channel_dim",
    "normalize_core_dims",
    "save_report_json",
    "select_data_array",
    "summarize_statistics",
    "validate_channel_values",
    "validate_era5_layout",
    "validate_era5_dataset_schema",
    "validate_era5_c116_layout",
    "validate_coordinate_values",
    "validate_six_hourly_day",
    "validate_same_layout",
    "validate_structure",
    "validate_weather_array",
]
