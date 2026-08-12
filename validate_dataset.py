#!/usr/bin/env python3
"""Validate rendered image counts, cell cardinality and final-PNG geometry."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from shift_ocr.geometry import aabb
from shift_ocr.shards import iter_jsonl


def positive_overlap(first, second) -> bool:
    a = aabb(first)
    b = aabb(second)
    return min(a[2], b[2]) - max(a[0], b[0]) > 0.5 and min(a[3], b[3]) - max(a[1], b[1]) > 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    args = parser.parse_args()
    annotation_dir = args.dataset_dir / "annotations"
    manifest = json.loads((annotation_dir / "manifest.json").read_text(encoding="utf-8"))
    cells = list(iter_jsonl(annotation_dir / "cells.jsonl"))
    schedules = {item["schedule_id"]: item for item in manifest["schedules"]}
    images = list((args.dataset_dir / "images").glob("*.png"))
    if len(images) != len(schedules):
        raise ValueError(f"PNG count {len(images)} != schedule count {len(schedules)}")
    observed = Counter(str(item["schedule_id"]) for item in cells)
    for schedule_id, schedule in schedules.items():
        expected = int(schedule["people_count"]) * int(schedule["day_count"])
        if observed[schedule_id] != expected:
            raise ValueError(f"{schedule_id}: {observed[schedule_id]} cells != {expected}")
    seen = set()
    grouped = {}
    for item in cells:
        schedule = schedules[str(item["schedule_id"])]
        width, height = schedule["image_size"]
        key = (item["schedule_id"], item["row_id"], int(item["day"]))
        if key in seen:
            raise ValueError(f"duplicate cell key: {key}")
        seen.add(key)
        if item["display_text"] != item["display_code"]:
            raise ValueError(f"display_text mismatch: {key}")
        if unicodedata.normalize("NFC", item["display_text"]) != item["display_text"]:
            raise ValueError(f"non-NFC display text: {key}")
        for field in ("cell_polygon", "text_polygon"):
            polygon = item[field]
            if len(polygon) != 4 or any(not (0 <= float(x) <= width and 0 <= float(y) <= height) for x, y in polygon):
                raise ValueError(f"invalid {field}: {key}")
        grouped.setdefault((item["schedule_id"], item["row_id"]), []).append(item)
    for row_key, row in grouped.items():
        ordered = sorted(row, key=lambda item: int(item["day"]))
        for first, second in zip(ordered, ordered[1:]):
            if positive_overlap(first["cell_polygon"], second["cell_polygon"]):
                raise ValueError(f"adjacent cells overlap: {row_key}")
    result = {
        "schedule_count": len(schedules), "png_count": len(images), "cell_count": len(cells),
        "object_count": sum(1 for _ in iter_jsonl(annotation_dir / "objects.jsonl")),
        "all_geometry_inside_images": True, "adjacent_positive_overlap": False,
        "display_text_matches_display_code": True, "nfc": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
