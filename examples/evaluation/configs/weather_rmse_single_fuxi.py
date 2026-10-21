"""Evaluation configuration with overridable input and output paths."""

import os
from pipelines.evaluation.config import EvalConfig, metric_options_from_var_metrics

VARS = [
    "z500",
    "q700",
    "t700",
    "t2m",
    "t850",
    "msl",
    "u850",
    "v850",
    "u10m",
    "v10m",
    "ws10m",
    "ws850",
]
VAR_METRICS = {
    "z500": ["rmse", "zonal_spectrum", "spherical_bands", "acc", "activity"],
    "q700": ["rmse"],
    "t700": ["rmse", "zonal_spectrum", "spherical_bands"],
    "t2m": ["rmse", "zonal_spectrum", "spherical_bands"],
    "t850": ["rmse"],
    "msl": ["rmse", "zonal_spectrum", "spherical_bands"],
    "u850": ["rmse", "activity", "zonal_spectrum", "spherical_bands"],
    "v850": ["rmse", "activity", "zonal_spectrum", "spherical_bands"],
    "u10m": ["rmse", "activity"],
    "v10m": ["rmse", "activity"],
}
METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}
_ERA5_STORE_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
cfg = EvalConfig(
    name="weather_rmse_single_fuxi",
    description="确定性连续量检验（FuXi × ERA5）：12 要素逐变量的 RMSE / 谱 / ACC / 活跃度",
    pipeline="weather_field_scores",
    forecast_reader={
        "type": "fuxi_phys",
        "root_dir": os.environ.get(
            "FUXI_OUTPUT", "/workspace/data/shenzw/fuxi_single_output"
        ),
        "variables": VARS,
        "step_hours": 6.0,
    },
    observation_reader={
        "type": "era5_zarr",
        "stores": {
            "pl": os.environ.get(
                "ERA5_PL_STORE",
                f"{_ERA5_STORE_ROOT}/era5_pl_2025.01-2026.07.c84.p25.h6.zarr",
            ),
            "sfc": os.environ.get(
                "ERA5_SFC_STORE",
                f"{_ERA5_STORE_ROOT}/era5_sfc_2025.01-2026.07.c15.p25.h6.zarr",
            ),
        },
        "variables": VARS,
    },
    reference_reader={
        "type": "daily_climatology",
        "root_dir": os.environ.get(
            "ERA5_CLIMO", "/workspace/data/worm/era5_clim_phys_14.nc"
        ),
        "smooth_days": 15,
    },
    start_date=os.environ.get("START_DATE", "20250102"),
    end_date=os.environ.get("END_DATE", "20251215"),
    limit=None,
    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/weather_rmse_single_fuxi",
    ),
    metric_options=METRIC_OPTIONS,
    log_level="INFO",
    writers=["csv_long", "spectrum"],
    options={
        "sample_by": "init_lead",
        "regions": {
            "tropics": {"lat_min": -20, "lat_max": 20},
            "nh_extratropics": {"lat_min": 20, "lat_max": 90},
            "sh_extratropics": {"lat_min": -90, "lat_max": -20},
        },
    },
    execution={
        "mode": "processes",
        "n_workers": 24,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {"observation": "window:16", "reference": "resident"},
        "resume": True,
    },
)
