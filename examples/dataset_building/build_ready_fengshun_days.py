"""Consume Fengshun's download watermark using the Tongyun publication flow."""

import argparse
from pathlib import Path

from examples.dataset_building import build_ready_tongyun_days as consumer
from examples.dataset_building import example_build_fengshun_daily as daily


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-days", type=int, default=1, help="Build at most N dates per run")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--dry-run", dest="write", action="store_false")
    parser.set_defaults(write=False)
    args = parser.parse_args(argv)
    if args.max_days < 1:
        parser.error("--max-days must be positive")
    return consumer.run(consumer.load_config(args.config), write=args.write,
                        max_days=args.max_days, builder=daily)


if __name__ == "__main__":
    raise SystemExit(main())
