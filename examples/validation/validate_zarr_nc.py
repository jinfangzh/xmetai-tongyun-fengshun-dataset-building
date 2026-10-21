"""Validate converted NetCDF against source Zarr with structured checks.

Checks:
1) Structure: time/channel/lat/lon dimensions.
2) Coordinates: time/lat/lon consistency.
3) Channel: names, order, and count.
4) Elementwise value check: bias mean over (time, channel, lat, lon).
5) Statistics: min/max/mean/std/NaN/Inf for both sides and differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.dataset import validate_zarr_netcdf_consistency


def _print_structure(label: str, structure: dict[str, object]) -> None:
	print("\n" + "=" * 60)
	print(f"{label} | 结构校验")
	print("=" * 60)
	print(f"dims: {structure['dims']}")
	print(f"sizes: {structure['sizes']}")


def _print_coordinate_check(results: dict[str, bool]) -> None:
	print("\n" + "=" * 60)
	print("坐标校验")
	print("=" * 60)
	print(f"time equal: {results['time']}")
	print(f"lat  equal: {results['lat']}")
	print(f"lon  equal: {results['lon']}")


def _print_channel_check(results: dict[str, object]) -> None:
	print("\n" + "=" * 60)
	print("Channel 校验")
	print("=" * 60)
	print(f"count equal : {results['count_equal']}")
	print(f"order equal : {results['order_equal']}")
	print(f"left count  : {results['left_count']}")
	print(f"right count : {results['right_count']}")
	print(f"missing in nc   : {results['missing_in_right']}")
	print(f"missing in zarr : {results['missing_in_left']}")


def _print_stats(label: str, stats: dict[str, float]) -> None:
	print("\n" + "=" * 60)
	print(f"{label} | 统计校验")
	print("=" * 60)
	print(f"min      : {stats['min']:.8e}")
	print(f"max      : {stats['max']:.8e}")
	print(f"mean     : {stats['mean']:.8e}")
	print(f"std      : {stats['std']:.8e}")
	print(f"NaN      : {int(stats['nan_count'])}")
	print(f"Inf      : {int(stats['inf_count'])}")


def _print_stats_diff(left: dict[str, float], right: dict[str, float]) -> None:
	print("\n" + "=" * 60)
	print("统计差异 (nc - zarr_denorm)")
	print("=" * 60)
	for name in ("min", "max", "mean", "std"):
		diff = left[name] - right[name]
		print(f"{name:8}: {diff:.8e}")
	print(f"nan_count: {int(left['nan_count'] - right['nan_count'])}")
	print(f"inf_count: {int(left['inf_count'] - right['inf_count'])}")


def _print_channel_bias_mean(channel_bias: dict[str, float]) -> None:
	print("\n" + "=" * 60)
	print("逐 channel 逐元素 bias_mean (nc - zarr_denorm)")
	print("=" * 60)
	for channel, value in channel_bias.items():
		print(f"{channel:>20}: {value:.8e}")


def get_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Validate denormalized Zarr and NetCDF consistency")
	parser.add_argument("--zarr-path", type=Path, required=True, help="Input normalized Zarr path")
	parser.add_argument("--nc-path", type=Path, required=True, help="Converted NetCDF path")
	parser.add_argument("--variable", type=str, default="data", help="Variable name to compare")
	parser.add_argument(
		"--report-json",
		type=Path,
		default=Path("./output/validate_zarr_nc_report.json"),
		help="Path to save all validation results as JSON",
	)
	parser.add_argument(
		"--coord-atol",
		type=float,
		default=0.0,
		help="Absolute tolerance for latitude/longitude coordinate comparison",
	)
	return parser.parse_args()


def main() -> None:
	args = get_args()
	report = validate_zarr_netcdf_consistency(
		args.zarr_path,
		args.nc_path,
		variable=args.variable,
		coord_atol=args.coord_atol,
		report_json=args.report_json,
	)

	structure = report["structure"]
	if not isinstance(structure, dict):
		raise TypeError("Invalid report structure: expected dict")

	_print_structure("ZARR(denorm)", structure["zarr_denorm"])
	_print_structure("NETCDF", structure["netcdf"])

	coord_result = report["coordinates"]
	if not isinstance(coord_result, dict):
		raise TypeError("Invalid report coordinates: expected dict")
	_print_coordinate_check(coord_result)

	channel_result = report["channels"]
	if not isinstance(channel_result, dict):
		raise TypeError("Invalid report channels: expected dict")
	_print_channel_check(channel_result)

	elementwise = report["elementwise"]
	if not isinstance(elementwise, dict):
		raise TypeError("Invalid report elementwise: expected dict")
	bias = float(elementwise["bias_mean"])
	channel_bias = elementwise["bias_mean_by_channel"]
	if not isinstance(channel_bias, dict):
		raise TypeError("Invalid report channel bias: expected dict")
	print("\n" + "=" * 60)
	print("逐元素校验 (time, channel, lat, lon)")
	print("=" * 60)
	print(f"bias_mean (nc - zarr_denorm): {bias:.8e}")
	_print_channel_bias_mean(channel_bias)

	statistics = report["statistics"]
	if not isinstance(statistics, dict):
		raise TypeError("Invalid report statistics: expected dict")
	zarr_stats = statistics["zarr_denorm"]
	nc_stats = statistics["netcdf"]
	if not isinstance(zarr_stats, dict) or not isinstance(nc_stats, dict):
		raise TypeError("Invalid report statistics content: expected dict")
	_print_stats("ZARR(denorm)", zarr_stats)
	_print_stats("NETCDF", nc_stats)
	_print_stats_diff(nc_stats, zarr_stats)

	print(f"\nSaved validation report: {args.report_json}")


if __name__ == "__main__":
	main()

