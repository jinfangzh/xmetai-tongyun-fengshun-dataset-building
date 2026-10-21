"""Run any single or batch evaluation configuration and optionally build reports.

Example from the repository root::

    python -m examples.evaluation.run_evaluation \
        --config weather_ts_single_tongyun

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
from pipelines.evaluation.reporting.report import IMPLEMENTED
from pipelines.evaluation.spec import pipeline_names
from datakit.runtime.logging import configure_logging


def run_example(
    config_name: str,
    *,
    forecast_root: Path | None = None,
    observation_root: Path | None = None,
    reference_root: Path | None = None,
    output_dir: Path | None = None,
    era5_pl_store: Path | None = None,
    era5_sfc_store: Path | None = None,
    report: bool = False,
    model_name: str | None = None,
) -> list[Path]:
    """Run cfg or cfgs sequentially; batch output overrides use one folder per name.

    Input paths are validated by each reader: root_dir may be a file or a directory.
    """
    loaded = load_config(config_name)
    configs = loaded if isinstance(loaded, list) else [loaded]
    if not configs or not all(isinstance(cfg, EvalConfig) for cfg in configs):
        raise ValueError(f"{config_name} must define cfg or a nonempty cfgs list")
    if report:
        # Check every task before evaluating any of them. Keep the original
        # report dispatch strict; unsupported reports must not be substituted.
        for config in configs:
            # Runner records the first segment as resolved_config.pipeline.
            # Later segments may supply probability tables consumed by a TS report.
            primary = pipeline_names(config)[0]
            if primary not in IMPLEMENTED:
                raise ValueError(
                    f"{config.name}: 不支持生成报告的主流程: {primary}。"
                    "请去掉 --report，仅执行评估。"
                )
    if output_dir is not None and len(configs) > 1:
        names = [cfg.name for cfg in configs]
        if len(set(names)) != len(names) or any(
            name in ("", ".", "..") or "/" in name or "\\" in name for name in names
        ):
            raise ValueError("Batch output override requires unique config names without path separators")

    # Apply overrides before starting any evaluation, so invalid options fail early.
    for config in configs:
        for label, reader, override in (
            ("forecast", config.forecast_reader, forecast_root),
            ("observation", config.observation_reader, observation_root),
            ("reference", config.reference_reader, reference_root),
        ):
            if override is None:
                continue
            if reader is None:
                raise ValueError(f"{label} root was provided, but {config.name} has no {label} reader")
            if reader.get("stores"):
                raise ValueError(f"{label} uses stores; override ERA5 paths with --era5-pl-store/--era5-sfc-store")
            reader["root_dir"] = str(override)
        for key, override in (("pl", era5_pl_store), ("sfc", era5_sfc_store)):
            if override is not None:
                if config.observation_reader.get("type") != "era5_zarr":
                    raise ValueError(f"{config.name} does not use an era5_zarr observation reader")
                config.observation_reader.setdefault("stores", {})[key] = str(override)
        if output_dir is not None:
            target = output_dir / config.name if len(configs) > 1 else output_dir
            config.output_dir = str(target)

    result_dirs = []
    for config in configs:
        configure_logging(config.log_level)
        print(f"开始评测：{config.name}")
        run_evaluation(config)
        target = Path(config.output_dir)
        if report:
            results = build_report(target, out_dir=target / "report", model_name=model_name)
            print(f"报告已生成：{results.get('report') or results['REPORT']}")
        result_dirs.append(target)
    return result_dirs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run a single or batch evaluation configuration")
    parser.add_argument(
        "--config", default="weather_ts_single_tongyun", help="config name or Python file"
    )
    parser.add_argument("--forecast-root", type=Path, help="forecast data directory")
    parser.add_argument("--observation-root", type=Path, help="observation data directory")
    parser.add_argument("--reference-root", type=Path, help="reference data directory")
    parser.add_argument("--output-dir", type=Path, help="result directory")
    parser.add_argument("--era5-pl-store", type=Path, help="ERA5 pressure-level Zarr store")
    parser.add_argument("--era5-sfc-store", type=Path, help="ERA5 surface Zarr store")
    report_options = parser.add_mutually_exclusive_group()
    report_options.add_argument("--report", action="store_true", help="评估后生成报告，运行前校验模板支持范围")
    report_options.add_argument("--no-report", dest="report", action="store_false", help="兼容旧命令：仅评估（现在为默认行为）")
    parser.set_defaults(report=False)
    parser.add_argument("--model-name", help="report title; defaults to the recorded run ID")
    args = parser.parse_args(argv)

    result_dirs = run_example(
        args.config,
        forecast_root=args.forecast_root,
        observation_root=args.observation_root,
        reference_root=args.reference_root,
        output_dir=args.output_dir,
        era5_pl_store=args.era5_pl_store,
        era5_sfc_store=args.era5_sfc_store,
        report=args.report,
        model_name=args.model_name,
    )
    for result_dir in result_dirs:
        print(f"评测完成，结果目录：{result_dir}")

if __name__ == "__main__":
    main()
