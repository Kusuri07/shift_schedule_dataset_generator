"""DBNet → recognizer → geometric clustering → header alignment fallback."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


DATE_PATTERN = re.compile(r"^(?:[1-9]|[12]\d|3[01])$")


def center(box):
    return (float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2


def cluster_rows(items: Iterable[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for source in sorted(items, key=lambda item: center(item["bbox"])[1]):
        item = dict(source)
        _x, y = center(item["bbox"])
        height = max(1.0, float(item["bbox"][3]) - float(item["bbox"][1]))
        best = None
        best_distance = float("inf")
        for row in rows:
            row_y = statistics.median(center(member["bbox"])[1] for member in row)
            tolerance = max(height, statistics.median(float(member["bbox"][3]) - float(member["bbox"][1]) for member in row)) * 0.65
            if abs(y - row_y) <= tolerance and abs(y - row_y) < best_distance:
                best, best_distance = row, abs(y - row_y)
        if best is None:
            rows.append([item])
        else:
            best.append(item)
    for row in rows:
        row.sort(key=lambda item: center(item["bbox"])[0])
    return sorted(rows, key=lambda row: statistics.median(center(item["bbox"])[1] for item in row))


def _header_candidate(row: list[Mapping[str, Any]]) -> tuple[int, float]:
    dates = [int(str(item.get("text", ""))) for item in row if DATE_PATTERN.match(str(item.get("text", "")))]
    if not dates:
        return 0, 0.0
    ordered = sorted(set(dates))
    continuity = sum(second - first == 1 for first, second in zip(ordered, ordered[1:])) / max(1, len(ordered) - 1)
    return len(ordered), continuity


def align_to_headers(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = cluster_rows(items)
    if not rows:
        return {"cells": [], "confidence": 0.0, "failure": "no_rows"}
    header_index, header_row = max(enumerate(rows), key=lambda pair: _header_candidate(pair[1]))
    date_items = [item for item in header_row if DATE_PATTERN.match(str(item.get("text", "")))]
    if len(date_items) < 2:
        return {"cells": [], "confidence": 0.0, "failure": "date_header_not_found"}
    date_items.sort(key=lambda item: int(str(item["text"])))
    anchors = {int(str(item["text"])): center(item["bbox"])[0] for item in date_items}
    spacings = [b - a for a, b in zip(sorted(anchors.values()), sorted(anchors.values())[1:]) if b > a]
    median_spacing = statistics.median(spacings) if spacings else 1.0
    for day in range(1, 32):
        if day not in anchors:
            nearest = min(anchors, key=lambda known: abs(known - day))
            anchors[day] = anchors[nearest] + (day - nearest) * median_spacing
    first_x = anchors[min(anchors)]
    body_rows = rows[header_index + 1:]
    cells: list[dict[str, Any]] = []
    aligned = 0
    expected = 0
    for output_row, row in enumerate(body_rows, start=1):
        row_y = statistics.median(center(item["bbox"])[1] for item in row)
        left = [item for item in row if center(item["bbox"])[0] < first_x - median_spacing * 0.35]
        name = max(left, key=lambda item: center(item["bbox"])[0], default=None)
        if name:
            cells.append({"row": output_row, "col": 0, "text": name.get("text", ""), "bbox": name["bbox"], "object_type": "name"})
        for item in row:
            x, _y = center(item["bbox"])
            if x < first_x - median_spacing * 0.35:
                continue
            day = min(anchors, key=lambda value: abs(anchors[value] - x))
            expected += 1
            distance = abs(anchors[day] - x)
            if distance <= median_spacing * 0.55:
                aligned += 1
                cells.append({"row": output_row, "col": day, "text": item.get("text", ""), "bbox": item["bbox"], "object_type": "shift_code"})
    header_confidence = min(1.0, len(date_items) / 20.0)
    alignment_confidence = aligned / max(1, expected)
    return {
        "cells": cells,
        "confidence": 0.55 * header_confidence + 0.45 * alignment_confidence,
        "header_count": len(date_items),
        "median_column_spacing": median_spacing,
        "missing_columns": sorted(day for day in range(min(anchors), max(anchors) + 1) if day not in {int(str(item["text"])) for item in date_items}),
    }
