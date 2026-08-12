#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shift_ocr.export import select_mobile_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-benchmarks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = json.loads(args.validation_benchmarks.read_text(encoding="utf-8"))
    result = select_mobile_profile(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
