"""Evaluation configuration with overridable input and output paths.
"""

import os
from pipelines.evaluation.config import EvalConfig

TS_THRESHOLDS = [0.1, 10.0, 25.0, 50.0, 100.0, 250.0]
cfg = EvalConfig(
    name="weather_ts_single_fgvp",
    description="FGVP 确定性降水 TS 评估（与站点观测对比）",
    pipeline="weather_ts_det",
    forecast_reader={
        "type": "fuxi",
        "root_dir": os.environ.get("FGVP_OUTPUT", "/workspace/data/shenzw/tongyun_single"),
        "variable": "tp",
        "step_hours": 6.0,
        "lead_times": None,
    },
    observation_reader={
        "type": "station",
        "root_dir": os.environ.get("STATION_OBS", "/workspace/data/worm/r0/2025"),
        "variable": "precipitation",
        "station_list": os.environ.get(
            "STATION_LIST", "/workspace/data/worm/r0/zd_sta_10285.dat"
        ),
    },
    transform_options={"time_window_accumulator": {"window_hours": 24}},
    metric_options={"ts_score": {"thresholds": TS_THRESHOLDS}},
    writers=["csv_long", "categorical_wide"],
    options={
        "local_utc_offset_hours": 8,
        "regions": {
            "南方": {"lat_min": 0, "lat_max": 25},
            "长江中下游": {"lat_min": 25, "lat_max": 32},
            "华北": {"lat_min": 32, "lat_max": 40},
            "东北": {"lat_min": 40, "lat_max": 90},
        },
    },
    limit=None,
    output_dir=os.environ.get(
        "EVAL_OUTPUT", "/workspace/szwCode/evaluation_results/weather_ts_single_tongyun"
    ),
    log_level="INFO",
    execution={
        "mode": "threads",
        "n_workers": 24,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {"observation": "resident"},
        "resume": True,
    },
    start_date="20251001",
    end_date="20251031",
)
