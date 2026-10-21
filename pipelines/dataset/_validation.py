"""Private validation workflow pipeline implementation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import xarray as xr

from datakit.io import read_netcdf, read_zarr
from datakit.validation import (
    build_statistics_diff,
    collect_structure,
    compare_channels,
    compare_coordinates,
    compute_bias_mean,
    compute_bias_mean_by_channel,
    denormalize_zarr_array,
    normalize_core_dims,
    save_report_json,
    select_data_array,
    summarize_statistics,
    validate_channel_values,
    validate_coordinate_values,
    validate_same_layout,
    validate_structure,
)


def validate_zarr_netcdf_consistency(
    zarr_path: str | Path,
    nc_path: str | Path,
    *,
    variable: str = "data",
    coord_atol: float = 0.0,
    report_json: str | Path | None = None,
    log_channels: Sequence[str] = (),
) -> dict[str, object]:
    """Validate one denormalized Zarr source against one NetCDF file.

    The function performs structure, coordinate, channel, elementwise-bias and
    summary-statistics checks, and returns a structured report dictionary.
    """

    report: dict[str, object] = {
        "input": {
            "zarr_path": str(zarr_path),
            "nc_path": str(nc_path),
            "variable": str(variable),
            "coord_atol": float(coord_atol),
            "log_channels": [str(name) for name in log_channels],
        },
        "status": "running",
    }

    zarr_opened = read_zarr(zarr_path)
    nc_opened = read_netcdf(nc_path)

    try:
        zarr_data = normalize_core_dims(select_data_array(zarr_opened, variable, source_name="zarr"))
        nc_data = normalize_core_dims(select_data_array(nc_opened, variable, source_name="netcdf"))

        zarr_data = denormalize_zarr_array(zarr_data, zarr_path, log_channels=log_channels)

        validate_structure(zarr_data)
        validate_structure(nc_data)
        validate_coordinate_values(zarr_data)
        validate_coordinate_values(nc_data)
        validate_channel_values(zarr_data)
        validate_channel_values(nc_data)

        zarr_for_compare = zarr_data.transpose("time", "channel", "lat", "lon")
        nc_for_compare = nc_data.transpose("time", "channel", "lat", "lon")

        report["structure"] = {
            "zarr_denorm": collect_structure(zarr_for_compare),
            "netcdf": collect_structure(nc_for_compare),
        }

        coord_result = compare_coordinates(zarr_for_compare, nc_for_compare, atol=coord_atol)
        report["coordinates"] = coord_result

        channel_result = compare_channels(zarr_for_compare, nc_for_compare)
        report["channels"] = channel_result

        if not all(coord_result.values()):
            raise ValueError("Coordinate mismatch found between Zarr and NetCDF")
        if not bool(channel_result["count_equal"]) or not bool(channel_result["order_equal"]):
            raise ValueError("Channel mismatch found between Zarr and NetCDF")

        validate_same_layout(nc_for_compare, zarr_for_compare)
        bias = compute_bias_mean(nc_for_compare, zarr_for_compare)
        channel_bias = compute_bias_mean_by_channel(nc_for_compare, zarr_for_compare)
        report["elementwise"] = {
            "bias_mean": float(bias),
            "bias_mean_by_channel": channel_bias,
        }

        zarr_stats = summarize_statistics(zarr_for_compare)
        nc_stats = summarize_statistics(nc_for_compare)
        report["statistics"] = {
            "zarr_denorm": zarr_stats,
            "netcdf": nc_stats,
            "diff_netcdf_minus_zarr_denorm": build_statistics_diff(nc_stats, zarr_stats),
        }

        report["status"] = "passed"
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        if report_json is not None:
            save_report_json(report_json, report)
        if hasattr(zarr_opened, "close"):
            zarr_opened.close()
        if hasattr(nc_opened, "close"):
            nc_opened.close()
