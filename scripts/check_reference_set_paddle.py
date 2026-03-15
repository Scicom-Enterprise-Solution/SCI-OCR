#!/usr/bin/env python3

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_reference_common import run_reference_set


def main() -> int:
    return run_reference_set("paddle")


if __name__ == "__main__":
    raise SystemExit(main())
