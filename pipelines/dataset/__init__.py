"""Dataset creation, mutation, export and validation operations."""

from importlib import import_module

__all__ = [
    "build_zarr",
    "BuildReport",
    "append_zarr",
    "validate_zarr",
    "convert_to_zarr",
    "extract_zarr_to_netcdf",
    "extract_raw_truth_day",
    "split_netcdf_by_time_and_channel",
    "validate_zarr_netcdf_consistency",
    "validate_standard_conversion",
    "run_conversion_validation",
]

_MODULES = {
    "build_zarr": ".build",
    "BuildReport": ".reports",
    "append_zarr": ".append",
    "validate_zarr": ".validate",
    "convert_to_zarr": ".old.conversion",
    "extract_zarr_to_netcdf": ".old.conversion",
    "extract_raw_truth_day": ".old.raw_truth",
    "split_netcdf_by_time_and_channel": ".old.split",
    "validate_zarr_netcdf_consistency": ".old.netcdf_comparison",
    "validate_standard_conversion": ".old.conversion_validation",
    "run_conversion_validation": ".old.conversion_validation",
}


def __getattr__(name: str):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
