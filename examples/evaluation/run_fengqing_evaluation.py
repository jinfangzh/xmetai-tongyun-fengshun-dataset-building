"""Run a Fengqing evaluation with real input data.

Example from the repository root::

    python -m examples.evaluation.run_fengqing_evaluation \
        --forecast-root /data/fengqing \
        --observation-root /data/cra_root \
        --reference-root /data/cli_fake \
        --output-dir ./results/fengqing

The selected config supplies the pipeline, variables, metrics, and time range.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Direct execution (including VS Code's file debugger) needs the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.evaluation.cli import run_evaluation
from pipelines.evaluation.config import EvalConfig, load_config
from pipelines.evaluation.reporting import build_report


def run_example(
    config_name: str,
    *,
    forecast_root: Path | None = None,
    observation_root: Path | None = None,
    reference_root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run evaluation and reporting, returning the results directory."""
    config = load_config(config_name)
    if not isinstance(config, EvalConfig):
        raise ValueError(f"{config_name} must define a single EvalConfig")

    for label, reader, override in (
        ("forecast", config.forecast_reader, forecast_root),
        ("observation", config.observation_reader, observation_root),
        ("reference", config.reference_reader, reference_root),
    ):
        if reader is None:
            if override is not None:
                raise ValueError(f"{label} root was provided, but the config has no {label} reader")
            continue
        if override is not None:
            reader["root_dir"] = str(override)
        if "root_dir" in reader and not Path(reader["root_dir"]).is_dir():
            raise FileNotFoundError(f"{label} data directory does not exist: {reader['root_dir']}")

    if output_dir is not None:
        config.output_dir = str(output_dir)
    
    print(config)
        
    run_evaluation(config)

    results = build_report(
        Path(config.output_dir),
        out_dir=Path(config.output_dir) / "report",
        model_name="FengQing",
    )
    print(f"报告已生成：{results['report']}")
    return Path(config.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-data Fengqing evaluation")
    parser.add_argument(
        "--config", default="fdp_rmse_single_fengqing", help="config name or Python file"
    )
    parser.add_argument("--forecast-root", type=Path, help="forecast data directory")
    parser.add_argument("--observation-root", type=Path, help="observation data directory")
    parser.add_argument("--reference-root", type=Path, help="reference data directory")
    parser.add_argument("--output-dir", type=Path, help="result directory")
    args = parser.parse_args()

    result_dir = run_example(
        args.config,
        forecast_root=args.forecast_root,
        observation_root=args.observation_root,
        reference_root=args.reference_root,
        output_dir=args.output_dir,
    )
    print(f"评测完成，结果目录：{result_dir}")

if __name__ == "__main__":
    main()
