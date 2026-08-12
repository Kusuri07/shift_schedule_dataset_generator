"""OCR/table metrics, schedule-level bootstrap confidence intervals and route selection."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for row, char_a in enumerate(first, start=1):
        current = [row]
        for column, char_b in enumerate(second, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (char_a != char_b),
            ))
        previous = current
    return previous[-1]


def polygon_iou(first, second) -> float:
    try:
        from shapely.geometry import Polygon
        a, b = Polygon(first), Polygon(second)
        if not a.is_valid or not b.is_valid:
            return 0.0
        return float(a.intersection(b).area / max(a.union(b).area, 1e-9))
    except ImportError:
        from .geometry import aabb
        left_a, top_a, right_a, bottom_a = aabb(first)
        left_b, top_b, right_b, bottom_b = aabb(second)
        intersection = max(0, min(right_a, right_b) - max(left_a, left_b)) * max(0, min(bottom_a, bottom_b) - max(top_a, top_b))
        union = (right_a - left_a) * (bottom_a - top_a) + (right_b - left_b) * (bottom_b - top_b) - intersection
        return intersection / max(union, 1e-9)


def _key(item: Mapping[str, Any]):
    return str(item["schedule_id"]), int(item.get("row_index") or 0), int(item.get("day") or item.get("col") or 0)


def _polygon(item: Mapping[str, Any], preferred: str):
    if item.get(preferred):
        return item[preferred]
    box = item.get("bbox") or item.get("bbox_px")
    if box is None and all(key in item for key in ("left", "top", "right", "bottom")):
        box = [item["left"], item["top"], item["right"], item["bottom"]]
    if box is None:
        return None
    left, top, right, bottom = map(float, box)
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def detection_hmean(ground_truth: Iterable[Mapping[str, Any]], predictions: Iterable[Mapping[str, Any]], preferred: str) -> float:
    truth_by_image: dict[str, list[Any]] = defaultdict(list)
    prediction_by_image: dict[str, list[Any]] = defaultdict(list)
    for item in ground_truth:
        polygon = _polygon(item, preferred)
        if polygon is not None and not item.get("ignore", False):
            truth_by_image[str(item.get("image_path", item["schedule_id"]))].append(polygon)
    for item in predictions:
        polygon = _polygon(item, preferred)
        if polygon is not None:
            prediction_by_image[str(item.get("image_path", item["schedule_id"]))].append(polygon)
    true_positive = false_positive = false_negative = 0
    for image_id in set(truth_by_image) | set(prediction_by_image):
        expected = truth_by_image[image_id]
        used = set()
        for predicted in prediction_by_image[image_id]:
            candidates = [(index, polygon_iou(target, predicted)) for index, target in enumerate(expected) if index not in used]
            if candidates and max(candidates, key=lambda item: item[1])[1] >= 0.5:
                used.add(max(candidates, key=lambda item: item[1])[0]); true_positive += 1
            else:
                false_positive += 1
        false_negative += len(expected) - len(used)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return 2 * precision * recall / max(1e-9, precision + recall)


def evaluate_cells(ground_truth: Iterable[Mapping[str, Any]], predictions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    ground_truth = list(ground_truth)
    predictions = list(predictions)
    truth = {_key(item): item for item in ground_truth if not item.get("ignore", False)}
    predicted = {_key(item): item for item in predictions}
    by_schedule: dict[str, list[tuple[str, str, int, bool]]] = defaultdict(list)
    polygon_tp = 0
    polygon_fp = 0
    polygon_fn = 0
    for key, expected in truth.items():
        actual = predicted.get(key)
        expected_text = str(expected.get("display_text", expected.get("display_code", "")))
        actual_text = str(actual.get("text", actual.get("display_text", ""))) if actual else ""
        distance = edit_distance(expected_text, actual_text)
        exact = actual is not None and expected_text == actual_text
        by_schedule[key[0]].append((expected_text, actual_text, distance, exact))
        if actual and expected.get("cell_polygon") and actual.get("cell_polygon"):
            if polygon_iou(expected["cell_polygon"], actual["cell_polygon"]) >= 0.5:
                polygon_tp += 1
            else:
                polygon_fp += 1
                polygon_fn += 1
        elif expected.get("cell_polygon"):
            polygon_fn += 1
    polygon_fp += sum(key not in truth for key in predicted)
    cell_count = len(truth)
    exact_count = sum(exact for rows in by_schedule.values() for *_prefix, exact in rows)
    total_characters = sum(len(expected) for rows in by_schedule.values() for expected, *_rest in rows)
    total_distance = sum(distance for rows in by_schedule.values() for _a, _b, distance, _exact in rows)

    truth_rows: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for key, expected in truth.items():
        actual = predicted.get(key)
        truth_rows[(key[0], key[1])].append(
            actual is not None and str(expected.get("display_text", expected.get("display_code", ""))) == str(actual.get("text", actual.get("display_text", "")))
        )
    row_accuracy = sum(all(values) for values in truth_rows.values()) / max(1, len(truth_rows))
    schedule_exact = {schedule_id: all(item[3] for item in rows) for schedule_id, rows in by_schedule.items()}
    precision = polygon_tp / max(1, polygon_tp + polygon_fp)
    recall = polygon_tp / max(1, polygon_tp + polygon_fn)
    return {
        "cell_count": cell_count,
        "cell_exact_accuracy": exact_count / max(1, cell_count),
        "cer": total_distance / max(1, total_characters),
        "row_level_accuracy": row_accuracy,
        "full_schedule_exact_accuracy": sum(schedule_exact.values()) / max(1, len(schedule_exact)),
        "cell_polygon_f1": 2 * precision * recall / max(1e-9, precision + recall),
        "text_detection_hmean": detection_hmean(ground_truth, predictions, "text_polygon"),
        "schedule_values": {
            schedule_id: {
                "cell_exact_accuracy": sum(item[3] for item in rows) / max(1, len(rows)),
                "cer": sum(item[2] for item in rows) / max(1, sum(len(item[0]) for item in rows)),
                "full_schedule_exact_accuracy": float(schedule_exact[schedule_id]),
            }
            for schedule_id, rows in by_schedule.items()
        },
    }


def bootstrap_schedule_ci(
    schedule_values: Mapping[str, Mapping[str, float]], metric: str, *,
    iterations: int = 2000, seed: int = 20260723,
) -> dict[str, float]:
    ids = sorted(schedule_values)
    if not ids:
        return {"estimate": 0.0, "low": 0.0, "high": 0.0, "iterations": iterations}
    values = [float(schedule_values[schedule_id][metric]) for schedule_id in ids]
    estimate = sum(values) / len(values)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(iterations))
    return {
        "estimate": estimate,
        "low": samples[math.floor(0.025 * (iterations - 1))],
        "high": samples[math.ceil(0.975 * (iterations - 1))],
        "iterations": iterations,
    }


def add_confidence_intervals(metrics: Mapping[str, Any], iterations: int = 2000) -> dict[str, Any]:
    output = dict(metrics)
    values = output.pop("schedule_values", {})
    output["confidence_intervals"] = {
        metric: bootstrap_schedule_ci(values, metric, iterations=iterations)
        for metric in ("cell_exact_accuracy", "cer", "full_schedule_exact_accuracy")
    }
    return output


def choose_mobile_route(results: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Choose A/B/C/D using real Validation only, with the specified D guardrail."""
    required = {"A", "B", "C", "D"}
    if set(results) != required:
        raise ValueError(f"route results must contain exactly {sorted(required)}")
    if any(str(value.get("split")) != "validation" for value in results.values()):
        raise ValueError("route selection accepts real Validation results only")
    reference = results["D"]
    eligible = []
    for route in ("A", "C"):
        value = results[route]
        accuracy_loss = float(reference["full_schedule_exact_accuracy"]) - float(value["full_schedule_exact_accuracy"])
        latency_reduction = 1 - float(value["p95_latency_ms"]) / max(1e-9, float(reference["p95_latency_ms"]))
        memory_reduction = 1 - float(value["peak_memory_mb"]) / max(1e-9, float(reference["peak_memory_mb"]))
        if accuracy_loss <= 0.005 and max(latency_reduction, memory_reduction) >= 0.15:
            eligible.append((route, value, latency_reduction, memory_reduction))
    if eligible:
        selected = min(eligible, key=lambda item: (float(item[1]["p95_latency_ms"]), float(item[1]["peak_memory_mb"])))
        route = selected[0]
        reason = "D 대비 정확도 손실 ≤0.5%p, latency 또는 memory ≥15% 절감"
    else:
        route = max(results, key=lambda key: (
            float(results[key]["full_schedule_exact_accuracy"]),
            -float(results[key]["p95_latency_ms"]),
            -float(results[key]["peak_memory_mb"]),
        ))
        reason = "guardrail 미충족으로 Validation 정확도 우선 선택"
    return {"selected_route": route, "reason": reason, "locked_before_test": True, "results": results}
