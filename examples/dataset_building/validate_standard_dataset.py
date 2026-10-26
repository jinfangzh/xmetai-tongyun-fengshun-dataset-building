"""Strict local validation CLI. Run with python -m from the repository root."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
from pipelines.dataset_building.workflows.dataset import run_dataset_validation
from pipelines.dataset_building.support.reports import report_exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按独立产品标准严格校验完整数据集；输入只读")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--schema", type=Path, nargs="+", help="一份或多份独立 YAML 标准")
    selection.add_argument("--schema-directory", type=Path, help="显式选择目录中的全部 YAML 标准")
    selection.add_argument("--run-config", type=Path, nargs="+", help="独立执行配置，引用 YAML 和运行证据")
    parser.add_argument(
        "--data-root", type=Path, help="数据根目录，按 YAML 冻结的目录名定位各套数据"
    )
    parser.add_argument("--dataset-path", type=Path, help="单套路径覆盖，仅适用于一份 YAML")
    parser.add_argument("--auxiliary-directory", type=Path)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--sample-blocks", type=int, default=3)
    parser.add_argument("--max-blocks", type=int, help="诊断限量；full 会明确报告未完成")
    parser.add_argument("--memory-budget-mib", type=int, default=2048)
    parser.add_argument(
        "--output", type=Path, required=True, help="报告目录，必须位于所有输入数据目录之外"
    )
    args = parser.parse_args(argv)
    try:
        schemas = args.schema or (
            sorted(
                [*args.schema_directory.glob("*.yaml"), *args.schema_directory.glob("*.yml")]
            ) if args.schema_directory else []
        )
        interrupted = False

        def interrupt(_signal, _frame):
            nonlocal interrupted
            interrupted = True

        previous = signal.signal(signal.SIGINT, interrupt)
        try:
            result = run_dataset_validation(
                args.output, schemas=schemas, run_configs=args.run_config or (),
                data_root=args.data_root, dataset_path=args.dataset_path,
                auxiliary_directory=args.auxiliary_directory, mode=args.mode,
                sample_blocks=args.sample_blocks, max_blocks=args.max_blocks,
                memory_budget_mib=args.memory_budget_mib,
                stop_requested=lambda: interrupted,
                on_progress=lambda message: print(message, flush=True),
            )
        finally:
            signal.signal(signal.SIGINT, previous)
        return 2 if result.interrupted else report_exit_code(result.reports)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Configuration/output error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
