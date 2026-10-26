"""Evaluation configuration with overridable input and output paths."""

import os
from pipelines.evaluation.config import EvalConfig

cfg = EvalConfig(
    name="weather_typhoon_single_fuxi",
    description="台风路径与强度检验（FuXi 确定性 × BABJ 报文）",
    pipeline="weather_typhoon_det",
    forecast_reader={
        "type": "fuxi_phys",
        "root_dir": os.environ.get(
            "FUXI_OUTPUT", "/workspace/data/shenzw/fuxi_single_output"
        ),
        "variables": ["msl", "u10m", "v10m"],
        "step_hours": 6.0,
        "lead_times": None,
    },
    observation_reader={
        "type": "babj",
        "root_dir": os.environ.get("BABJ_DIR", "/workspace/data/LiuJs/babj"),
    },
    options={
        "center_half_deg": 3.0,
        "early_half_deg": 4.0,
        "early_hours": 12.0,
        "intensity_half_deg": 5.0,
        "wind_variables": ["u10m", "v10m"],
        "seed_min_offset_hours": 6.0,
    },
    start_date=os.environ.get("START_DATE", "20250101"),
    end_date=os.environ.get("END_DATE", "20251231"),
    limit=None,
    output_dir=os.environ.get(
        "EVAL_OUTPUT",
        "/workspace/szwCode/evaluation_results/weather_typhoon_single_fuxi",
    ),
    log_level="INFO",
    writers=["csv_long", "typhoon_cases"],
    execution={
        "mode": "processes",
        "n_workers": 8,
        "chunk_days": 1,
        "lead_chunk_days": 0,
        "loads": {"observation": "slice"},
        "resume": True,
    },
)
