"""Private validation workflow pipeline implementation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from datakit import io as dataset_io
from datakit.transforms import denormalize_xarray
from datakit.validation import (
    build_statistics_diff,
    collect_structure,
    compare_channels,
    compare_coordinates,
    compute_bias_mean,
    compute_bias_mean_by_channel,
    normalize_core_dims,
    select_data_array,
    summarize_statistics,
)
from datakit.validation.structure import (
    check_channels,
    check_coordinates,
    check_dimensions,
    check_same_layout,
)

from ..reports import to_jsonable


def load_channel_statistic_from_zarr(
    path: str | Path,
    name: str,
    channels: Sequence[Any],
) -> xr.DataArray:
    """Load one channel statistic vector from Zarr storage.

    Statistics are resolved from top-level arrays (``mean``/``std``).
    """

    import zarr

    candidate = Path(path).expanduser() / name
    if not candidate.exists():
        raise FileNotFoundError(f"Cannot find zarr statistic {name!r} in {path}")
    node = zarr.open(str(candidate), mode="r")
    if isinstance(node, zarr.Group):
        keys = list(node.array_keys())
        if not keys:
            raise FileNotFoundError(f"Cannot find zarr statistic {name!r} in {path}")
        node = node[keys[0]]
    values = np.asarray(node[...]).reshape(-1)
    if len(values) != len(channels):
        raise ValueError(
            f"{name} length {len(values)} does not match channel count {len(channels)}"
        )
    return xr.DataArray(values, dims=("channel",), coords={"channel": list(channels)})


def denormalize_zarr_array(
    array: xr.DataArray,
    zarr_path: str | Path,
    *,
    log_channels: Sequence[str] = (),
) -> xr.DataArray:
    """Denormalize a Zarr-backed channel array using Zarr mean/std statistics."""

    channels = array.channel.values.tolist()
    mean = load_channel_statistic_from_zarr(zarr_path, "mean", channels)
    std = load_channel_statistic_from_zarr(zarr_path, "std", channels)
    return denormalize_xarray(
        array.astype("float32"),
        mean,
        std,
        channel_dim="channel",
        log_channels=log_channels,
    )


def save_report_json(path: str | Path, payload: dict[str, object]) -> None:
    """Save a validation report as UTF-8 JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
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

    zarr_opened = nc_opened = None
    try:
        zarr_opened = dataset_io.open(zarr_path).read()
        nc_opened = dataset_io.open(nc_path).read()
        zarr_data = normalize_core_dims(
            select_data_array(zarr_opened, variable, source_name="zarr")
        )
        nc_data = normalize_core_dims(
            select_data_array(nc_opened, variable, source_name="netcdf")
        )

        report["checks"] = {}
        for label, array in (("zarr", zarr_data), ("netcdf", nc_data)):
            checks = (
                check_dimensions(array)
                + check_coordinates(array)
                + check_channels(array)
            )
            report["checks"][label] = to_jsonable([asdict(check) for check in checks])
            failures = [check for check in checks if check.status == "fail"]
            if failures:
                raise ValueError("; ".join(str(check.evidence) for check in failures))

        zarr_data = denormalize_zarr_array(
            zarr_data, zarr_path, log_channels=log_channels
        )

        zarr_for_compare = zarr_data.transpose("time", "channel", "lat", "lon")
        nc_for_compare = nc_data.transpose("time", "channel", "lat", "lon")

        report["structure"] = {
            "zarr_denorm": collect_structure(zarr_for_compare),
            "netcdf": collect_structure(nc_for_compare),
        }

        coord_result = compare_coordinates(
            zarr_for_compare, nc_for_compare, atol=coord_atol
        )
        report["coordinates"] = coord_result

        channel_result = compare_channels(zarr_for_compare, nc_for_compare)
        report["channels"] = channel_result

        if not all(coord_result.values()):
            raise ValueError("Coordinate mismatch found between Zarr and NetCDF")
        if not bool(channel_result["count_equal"]) or not bool(
            channel_result["order_equal"]
        ):
            raise ValueError("Channel mismatch found between Zarr and NetCDF")

        layout_checks = check_same_layout(nc_for_compare, zarr_for_compare)
        report["checks"]["layout"] = to_jsonable(
            [asdict(check) for check in layout_checks]
        )
        failures = [check for check in layout_checks if check.status == "fail"]
        if failures:
            raise ValueError(failures[0].evidence)
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
            "diff_netcdf_minus_zarr_denorm": build_statistics_diff(
                nc_stats, zarr_stats
            ),
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
