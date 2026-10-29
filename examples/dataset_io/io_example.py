"""Read a dataset from a YAML declaration, optionally normalize and export it.

Run with python -m examples.dataset_io.io_example --help.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datakit import io
from datakit.schema.mapping import normalize_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Dataset description YAML"
    )
    parser.add_argument(
        "--path", type=Path, required=True, help="Source root or single file"
    )
    parser.add_argument("--time", nargs="+", help="Source observation/analysis times")
    parser.add_argument("--init-time", nargs="+", help="Forecast initialization times")
    parser.add_argument("--lead-time", nargs="+", type=int, help="Forecast lead hours")
    parser.add_argument("--member", nargs="+", type=int, help="Forecast member numbers")
    parser.add_argument(
        "--cycle-time", nargs="+", help="Month/day/hour labels, e.g. 010106"
    )
    parser.add_argument(
        "--all-files", action="store_true", help="Allow reading the whole collection"
    )
    parser.add_argument(
        "--variables", nargs="+", help="Explicitly standardize selected variables"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional output; existing output is refused"
    )
    args = parser.parse_args(argv)

    description = io.get_description(args.config)
    selection = {
        name: value
        for name in ("time", "init_time", "lead_time", "member", "cycle_time")
        if (value := getattr(args, name)) is not None
    }
    source = io.open_dataset(args.path, description, selection=selection)
    with source.read(all_files=args.all_files) as raw:
        result = (
            normalize_dataset(raw, args.variables, description)
            if args.variables
            else raw
        )
        if result is None:
            raise ValueError("None of the requested variables exist in this dataset")
        print(result)
        if args.output is not None:
            print("Written:", io.open(args.output).write(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
