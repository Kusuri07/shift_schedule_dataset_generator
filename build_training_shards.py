#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shift_ocr.master_split import MasterSplit
from shift_ocr.shards import build_parquet_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--master-split", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-schedules", type=int, default=500)
    parser.add_argument("--max-objects", type=int, default=250_000)
    args = parser.parse_args()
    split = MasterSplit.load(args.master_split)
    result = build_parquet_shards(
        args.objects, split, args.output_dir,
        max_schedules=args.max_schedules, max_objects=args.max_objects,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
