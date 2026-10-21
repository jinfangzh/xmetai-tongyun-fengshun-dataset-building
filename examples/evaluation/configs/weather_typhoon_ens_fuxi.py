"""Migrated evaluation configuration; paths may be overridden by the tests."""

from pipelines.evaluation.config import EvalConfig

cfg = EvalConfig(
    name="weather_typhoon_ens_fuxi",
    description="台风路径与强度检验（FuXi 集合 × BABJ 报文）",
    pipeline="weather_typhoon_ens",
    forecast_reader={
        "type": "fuxi_ens_phys",
        "root_dir": "/workspace/data/shenzw/fuxi_ens_output",
        "variables": ["msl", "u10m", "v10m"],
        "step_hours": 6.0,
        "lead_times": None,
    },
    observation_reader={"type": "babj", "root_dir": "/workspace/data/LiuJs/babj"},
    options={
        "center_half_deg": 3.0,
        "early_half_deg": 4.0,
        "early_hours": 12.0,
        "intensity_half_deg": 5.0,
        "wind_variables": ["u10m", "v10m"],
        "seed_min_offset_hours": 6.0,
    },
    start_date="20250101",
    end_date="20251231",
    limit=None,
    output_dir="/workspace/szwCode/evaluation_results/weather_typhoon_ens_fuxi",
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
