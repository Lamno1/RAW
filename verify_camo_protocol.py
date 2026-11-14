"""Verify that the workspace still matches the frozen S0 manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from camo_build.protocol import verify_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--protocol-dir", type=Path)
    args = parser.parse_args()
    protocol_dir = args.protocol_dir or args.data_dir / "protocol" / "s0"
    result = verify_protocol(args.data_dir, protocol_dir)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
