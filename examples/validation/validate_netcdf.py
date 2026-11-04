"""Compare converted NetCDF against reference NetCDF and report numeric differences.

Example:
    python examples/validation/validate_netcdf.py \
        --converted /path/to/converted.nc \
        --reference /path/to/reference.nc \
        --variable data \
        --channel-report-csv /tmp/channel_diff.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from datakit.validation import (
    analyze_range,
    count_invalid,
)
from datakit.validation.structure import (
    check_channels,
    check_coordinates,
    check_dimensions,
)


def _decode_channel(values: list[Any]) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in values
    ]


def _normalize_dims(array: xr.DataArray) -> xr.DataArray:
    if "level" in array.dims and "channel" not in array.dims:
        array = array.rename({"level": "channel"})
    return array


def _load_legacy_real_netcdf(path: Path) -> xr.DataArray:
    """Load legacy NetCDF weather data stored as raw HDF5 arrays."""

    with h5py.File(path, "r", locking=False) as handle:
        data = np.array(handle["data"][...], copy=True)
        channels = _decode_channel(handle["channel"][...].tolist())

        lat = np.array(handle["lat"][...], copy=True)
        lon = np.array(handle["lon"][...], copy=True)

        raw_time = np.array(handle["time"][...], copy=True).reshape(-1)
        time = pd.to_datetime(
            raw_time.astype(np.int64),
            unit="D",
            origin=pd.Timestamp("1950-01-01"),
        )

    if data.ndim == 3:
        array = xr.DataArray(
            data,
            dims=("channel", "lat", "lon"),
            coords={
                "channel": channels,
                "lat": lat,
                "lon": lon,
            },
            name="data",
        )
        return array.expand_dims(time=[pd.Timestamp(time[0])])

    if data.ndim == 4:
        return xr.DataArray(
            data,
            dims=("time", "channel", "lat", "lon"),
            coords={
                "time": pd.DatetimeIndex(time),
                "channel": channels,
                "lat": lat,
                "lon": lon,
            },
            name="data",
        )

    raise ValueError(f"Unsupported legacy data rank: {data.ndim}")


def _load_netcdf_array(path: Path, variable: str) -> xr.DataArray:
    """Load one data variable from NetCDF, with legacy fallback."""

    try:
        opened = xr.open_dataset(path)
        try:
            if variable in opened.data_vars:
                array = opened[variable]
            elif len(opened.data_vars) == 1:
                array = opened[next(iter(opened.data_vars))]
            else:
                raise KeyError(
                    f"Variable {variable!r} not found in {path}; "
                    f"available: {list(opened.data_vars)}"
                )

            array = _normalize_dims(array).load()
            if "time" not in array.dims and "time" in array.coords:
                array = array.expand_dims(time=[pd.Timestamp(array.time.values)])
            return array
        finally:
            opened.close()
    except Exception:
        return _normalize_dims(_load_legacy_real_netcdf(path)).load()


def _build_common_order(left: xr.DataArray, right: xr.DataArray) -> tuple[str, ...]:
    preferred = ("time", "channel", "lat", "lon")
    shared = [name for name in preferred if name in left.dims and name in right.dims]
    shared.extend(
        name for name in left.dims if name in right.dims and name not in shared
    )
    return tuple(shared)


def _align_for_compare(
    converted: xr.DataArray,
    reference: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align two arrays using coordinate intersection for comparable dimensions."""

    common_order = _build_common_order(converted, reference)
    if not common_order:
        raise ValueError(
            "No shared dimensions found between converted and reference arrays"
        )

    for name in common_order:
        if name not in converted.coords:
            converted = converted.assign_coords(
                {name: np.arange(converted.sizes[name])}
            )
        if name not in reference.coords:
            reference = reference.assign_coords(
                {name: np.arange(reference.sizes[name])}
            )

    converted_aligned, reference_aligned = xr.align(
        converted.transpose(*common_order),
        reference.transpose(*common_order),
        join="inner",
    )
    for dim in common_order:
        if converted_aligned.sizes.get(dim, 0) == 0:
            raise ValueError(f"No overlap on dimension {dim!r} after alignment")
    return converted_aligned, reference_aligned


def _compute_diff_metrics(
    converted: xr.DataArray,
    reference: xr.DataArray,
) -> dict[str, float]:
    converted_values = np.asarray(converted.values)
    reference_values = np.asarray(reference.values)

    finite_mask = np.isfinite(converted_values) & np.isfinite(reference_values)
    valid_count = int(finite_mask.sum())
    if valid_count == 0:
        raise ValueError("No finite overlapping values available for comparison")

    delta = converted_values[finite_mask] - reference_values[finite_mask]
    abs_delta = np.abs(delta)
    squared = delta * delta

    metrics: dict[str, float] = {
        "bias_mean": float(np.mean(delta)),
        "mae": float(np.mean(abs_delta)),
        "rmse": float(np.sqrt(np.mean(squared))),
    }
    return metrics


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for key in ("bias_mean", "mae", "rmse"):
        value = metrics[key]
        if np.isnan(value):
            print(f"{key:>16}: NaN")
        else:
            print(f"{key:>16}: {value:.8e}")


def _build_channel_report(
    converted: xr.DataArray,
    reference: xr.DataArray,
) -> pd.DataFrame:
    if "channel" not in converted.dims:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for channel in converted.channel.values.tolist():
        lhs = converted.sel(channel=channel)
        rhs = reference.sel(channel=channel)
        metrics = _compute_diff_metrics(lhs, rhs)
        rows.append({"channel": str(channel), **metrics})
    return pd.DataFrame(rows)


def _print_basic_quality(label: str, array: xr.DataArray) -> None:
    print("\n" + "=" * 60)
    print(f"{label} dataset info")
    print("=" * 60)
    print(array)

    print("\nValidation")
    checks = check_dimensions(array, required_dims=("time", "lat", "lon"))
    checks += check_coordinates(array)
    channel = "channel" if "channel" in array.dims else "level"
    checks += check_channels(array, coordinate=channel)
    failures = [check for check in checks if check.status == "fail"]
    if failures:
        print("Validation failed:")
        for check in failures:
            print(f"[{check.status}] {check.check_id}: {check.evidence}")
        return
    print("Validation passed")

    nan_count, inf_count = count_invalid(array)
    print("\nInvalid value statistics")
    print(f"NaN count : {nan_count}")
    print(f"Inf count : {inf_count}")

    print("\nRange analysis")
    print(analyze_range(array, axis=channel))


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Compare converted NetCDF with reference NetCDF"
    )

    parser.add_argument(
        "--converted",
        type=Path,
        required=True,
        help="Path to converted NetCDF file",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to reference (ground truth) NetCDF file",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="data",
        help="Variable name to compare",
    )
    parser.add_argument(
        "--channel-report-csv",
        type=Path,
        default=None,
        help="Optional path to save per-channel difference report as CSV",
    )

    args = parser.parse_args()

    converted = _load_netcdf_array(args.converted, args.variable)
    reference = _load_netcdf_array(args.reference, args.variable)

    _print_basic_quality("Converted", converted)
    _print_basic_quality("Reference", reference)

    converted_aligned, reference_aligned = _align_for_compare(converted, reference)
    summary = _compute_diff_metrics(
        converted_aligned,
        reference_aligned,
    )
    _print_metrics("Numerical difference summary", summary)

    channel_report = _build_channel_report(
        converted_aligned,
        reference_aligned,
    )
    if not channel_report.empty:
        print("\n" + "=" * 60)
        print("Per-channel numerical difference")
        print("=" * 60)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(channel_report)

        if args.channel_report_csv is not None:
            args.channel_report_csv.parent.mkdir(parents=True, exist_ok=True)
            channel_report.to_csv(args.channel_report_csv, index=False)
            print(f"Saved channel report: {args.channel_report_csv}")


if __name__ == "__main__":
    main()
