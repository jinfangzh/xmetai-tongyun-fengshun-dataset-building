"""Run an explicit source/target comparison and write a bound test receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal

from datakit.validation.zarr_inspection import file_sha256
from pipelines.dataset import validate_standard_conversion
from datakit.schema import ConversionChannelRule, ConversionValidationSpec, EvidenceReference
from datakit.validation import to_jsonable
from ._standard_config import load_profile, read_json


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
    config = read_json(args.config)
    spec, document = load_profile(Path(config["target_profile"]))
    source = Path(config["source_path"])
    target = Path(config.get("target_path") or document["dataset_path"])
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (source, target)):
        raise ValueError("Comparison output must be outside both input datasets")
    values = dict(config["conversion"])
    values["rules"] = tuple(ConversionChannelRule(**r) for r in values["rules"])
    values["source_channels"] = tuple(values["source_channels"])
    if values.get("source_statistics"):
        reference = values["source_statistics"]
        values["source_statistics"] = EvidenceReference(
            Path(reference["path"]), reference["sha256"]
        )
    if "source_dimensions" in values:
        values["source_dimensions"] = tuple(values["source_dimensions"])
    conversion = ConversionValidationSpec(**values)
    interrupted = False

    def interrupt(_signal, _frame):
        nonlocal interrupted
        interrupted = True

    previous = signal.signal(signal.SIGINT, interrupt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.with_suffix(".blocks.jsonl").open("w", encoding="utf-8") as stream:

            def record(entry):
                stream.write(
                    json.dumps(to_jsonable(entry), ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()

            receipt = validate_standard_conversion(
                source,
                target,
                spec,
                conversion,
                mode=args.mode,
                max_blocks=args.max_blocks,
                memory_budget_mib=args.memory_budget_mib,
                on_block=record,
                stop_requested=lambda: interrupted,
            )
    finally:
        signal.signal(signal.SIGINT, previous)
    receipt["config_sha256"] = file_sha256(args.config)
    args.output.write_text(
        json.dumps(to_jsonable(receipt), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
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
