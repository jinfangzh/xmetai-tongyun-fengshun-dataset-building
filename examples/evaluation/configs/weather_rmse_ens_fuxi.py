"""Migrated evaluation configuration; paths may be overridden by the tests."""

import os
from pipelines.evaluation.config import EvalConfig, metric_options_from_var_metrics

VARS = ["z500", "msl", "u200", "v200", "ws200"]
VAR_METRICS = {
    "z500": [
        "rmse",
        "crps",
        "spread_error",
        "acc",
        "activity",
        "zonal_spectrum",
        "spherical_bands",
    ],
    "msl": ["rmse", "spread_error", "acc"],
    "u200": ["rmse", "spread_error", "activity", "zonal_spectrum", "spherical_bands"],
    "v200": ["rmse", "spread_error", "activity", "zonal_spectrum", "spherical_bands"],
    "ws200": ["rmse", "spread_error", "activity", "zonal_spectrum", "spherical_bands"],
}
METRIC_OPTIONS = metric_options_from_var_metrics(VAR_METRICS)
METRIC_OPTIONS["acc"] = {**METRIC_OPTIONS["acc"], "centered": False}
METRIC_OPTIONS["zonal_spectrum"] = {
    **METRIC_OPTIONS["zonal_spectrum"],
    "max_wavenumber": 720,
}
METRIC_OPTIONS["spread_error"] = {**METRIC_OPTIONS["spread_error"], "ddof": 0}
_ERA5_STORE_ROOT = "/workspace/data/liujunjie/era5_foundation_store2"
cfg = EvalConfig(
    name="weather_rmse_ens_fuxi",
    description="集合场检验（FuXi 集合 × ERA5）：z500/msl/u200/v200/ws200 的 RMSE / CRPS / ACC / 活跃度 / 谱 / 球谐带功率",
    pipeline="weather_ens_field_scores",
    forecast_reader={
        "type": "fuxi_ens_phys",
        "root_dir": os.environ.get(
            "FUXI_ENS_OUTPUT", "/workspace/data/shenzw/fuxi_ens_output"
        ),
        "variables": VARS,
        "step_hours": 6.0,
        "lead_times": None,
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
    end_date=os.environ.get("END_DATE", "20250104"),
    limit=None,
    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/weather_rmse_ens_fuxi",
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
