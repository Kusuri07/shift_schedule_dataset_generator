#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shift_ocr.shards import verify_compatible_annotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", required=True, type=Path)
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_compatible_annotations(
        args.annotations_dir / "objects.jsonl", args.annotations_dir / "objects.csv", args.shard_dir
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
