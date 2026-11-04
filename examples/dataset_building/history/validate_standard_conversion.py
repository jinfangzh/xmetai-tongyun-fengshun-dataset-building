"""Run an explicit source/target comparison and write a bound test receipt."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal

from datakit.validation.zarr_inspection import file_sha256
from pipelines.dataset.old.conversion_validation import run_conversion_validation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicit frozen conversion mapping; no alias inference or repeated unit conversion"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--memory-budget-mib", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    interrupted = False

    def interrupt(_signal, _frame):
        nonlocal interrupted
        interrupted = True

    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        receipt = run_conversion_validation(
            args.config, args.output, mode=args.mode, max_blocks=args.max_blocks,
            memory_budget_mib=args.memory_budget_mib,
            stop_requested=lambda: interrupted,
        )
    finally:
        signal.signal(signal.SIGINT, previous)
    print(
        f"receipt={args.output} sha256={file_sha256(args.output)} comparison_complete={receipt['comparison_complete']}"
    )
    return (
        0
        if receipt["comparison_complete"]
        and all(c["status"] == "pass" for c in receipt["checks"].values())
        else 2 if interrupted or args.mode == "quick" or args.max_blocks else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
