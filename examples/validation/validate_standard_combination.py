"""Joint-use checks require independently successful single-dataset reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datakit.io import file_sha256
from pipelines.dataset import validate_standard_combination
from datakit.validation import to_jsonable
from ._standard_config import load_profile, read_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicit combination grid/time/channel/shared-array checks"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    paths, specs, reports = [], [], []
    for member in config["members"]:
        spec, document = load_profile(Path(member["profile"]))
        path = Path(member.get("path") or document["dataset_path"])
        if args.output.resolve().is_relative_to(path.resolve()):
            raise ValueError("Combination report must be outside every input")
        reference = member["individual_report"]
        if file_sha256(reference["path"]) != reference["sha256"]:
            raise ValueError("Frozen individual report hash mismatch")
        paths.append(path)
        specs.append(spec)
        reports.append(read_json(Path(reference["path"])))
    result = validate_standard_combination(
        paths,
        specs,
        policy=config["time_policy"],
        allow_intersection_loss=config.get("allow_intersection_loss", False),
        shared_arrays=tuple(config.get("shared_arrays", [])),
        individual_reports=reports,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(to_jsonable(result), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
