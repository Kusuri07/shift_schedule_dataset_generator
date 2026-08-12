#!/usr/bin/env python3
"""Create the only allowed schedule-level train/validation/test/OOD split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shift_ocr.master_split import create_master_split, write_master_split


def read_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schedules = payload.get("schedules")
    if not isinstance(schedules, list):
        raise ValueError(f"manifest has no schedules array: {path}")
    return schedules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ood-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--capture-target", type=int, default=0)
    args = parser.parse_args()

    schedules = read_manifest(args.manifest)
    for item in schedules:
        item.setdefault("capture_target", args.capture_target)
    ood = read_manifest(args.ood_manifest) if args.ood_manifest else []
    config = {
        "source_manifest": str(args.manifest.resolve()),
        "ood_manifest": str(args.ood_manifest.resolve()) if args.ood_manifest else None,
        "seed": args.seed,
        "ratios": [0.70, 0.15, 0.15],
        "cv": "train-only-3-fold",
    }
    records, metadata = create_master_split(
        schedules, seed=args.seed, config=config, ood_schedules=ood
    )
    write_master_split(records, metadata, args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
