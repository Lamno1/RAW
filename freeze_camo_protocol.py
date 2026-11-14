"""CLI for freezing CAMO-Pyramid S0 artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from camo_build.protocol import assert_external_temp, freeze_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--basetemp",
        type=Path,
        help="optional pytest scratch directory; must be outside --data-dir",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.basetemp is not None:
        assert_external_temp(args.data_dir, args.basetemp)
    output = args.output or args.data_dir / "protocol" / "s0"
    protocol = freeze_protocol(args.data_dir, output, overwrite=args.overwrite)
    print(json.dumps({"output": str(output), "version": protocol["version"]}, indent=2))


if __name__ == "__main__":
    main()
