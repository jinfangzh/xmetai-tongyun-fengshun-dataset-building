"""Evaluation configuration with overridable input and output paths."""

import os
from pipelines.evaluation.config import EvalConfig

BSS_REF_DEFAULT = os.environ.get("BSS_REF", "/workspace/data/worm/r0/ref")
cfg = EvalConfig(
    name="weather_ts_ens_fuxi",
    description="FuXi 集合降水评估：24h 集合平均 TS + 6h 逐成员概率 AROC/BS/BSS",
    pipeline=["weather_ts_ens", "weather_ts_ens_prob"],
    forecast_reader={
        "type": "fuxi_ens",
        "root_dir": os.environ.get(
            "FUXI_ENS_OUTPUT", "/workspace/data/shenzw/fuxi_ens_output"
        ),
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
    reference_reader={"type": "ref_probability", "root_dir": BSS_REF_DEFAULT},
    start_date=os.environ.get("START_DATE", "20250101"),
    end_date=os.environ.get("END_DATE", "20251215"),
    limit=None,
    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/xmetai-evaluate/evaluation_results/ts_multi_fuxi_ens",
    ),
    log_level="INFO",
    options={
        "local_utc_offset_hours": 8,
        "regions": {
            "南方": {"lat_min": 0, "lat_max": 25},
            "长江中下游": {"lat_min": 25, "lat_max": 32},
            "华北": {"lat_min": 32, "lat_max": 40},
            "东北": {"lat_min": 40, "lat_max": 90},
        },
    },
    execution={
        "mode": "processes",
        "n_workers": 24,
        "chunk_days": 1,
        "lead_chunk_days": 1,
        "loads": {"observation": "resident"},
        "resume": True,
    },
)
