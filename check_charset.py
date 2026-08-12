#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from shift_ocr.charset import coverage_report, load_charset, write_coverage_report
from shift_ocr.shards import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--charset", type=Path, default=Path("data/korean_charset_v1.txt"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = coverage_report(iter_jsonl(args.objects), load_charset(args.charset))
    write_coverage_report(report, args.output)


if __name__ == "__main__":
    main()
