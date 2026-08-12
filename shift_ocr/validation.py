"""Model-family Validation metrics used for checkpoint selection only."""

from __future__ import annotations

from .evaluation import edit_distance
from .models import TABLE_CANDIDATE_TOP_K, decode_table_candidates


def _decode_ctc(logits, charset):
    indices = logits.argmax(dim=-1).detach().cpu().tolist()
    decoded = []
    for sequence in indices:
        previous = -1
        characters = []
        for index in sequence:
            if index != previous and index > 0 and index - 1 < len(charset):
                characters.append(charset[index - 1])
            previous = index
        decoded.append("".join(characters))
    return decoded


def validate_recognizer(model, loader, charset, device) -> dict[str, float]:
    import torch

    exact = 0
    distance = 0
    characters = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            predictions = _decode_ctc(logits, charset)
            for expected, predicted in zip(batch["display_text"], predictions):
                exact += expected == predicted
                distance += edit_distance(expected, predicted)
                characters += len(expected)
                count += 1
    return {"cell_exact_accuracy": exact / max(1, count), "cer": distance / max(1, characters)}


def _component_boxes(mask):
    import cv2
    import numpy as np

    contours, _hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) >= 2]


def _box_iou(first, second):
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return intersection / max(1, w1 * h1 + w2 * h2 - intersection)


def validate_dbnet(model, loader, device) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    true_positive = false_positive = false_negative = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to(device))["probability"]
            output = functional.interpolate(output, size=batch["probability_target"].shape[-2:], mode="bilinear", align_corners=False)
            for predicted, expected in zip(output[:, 0].cpu().numpy() > 0.3, batch["probability_target"][:, 0].numpy() > 0.5):
                predicted_boxes = _component_boxes(predicted)
                expected_boxes = _component_boxes(expected)
                used = set()
                for box in predicted_boxes:
                    matches = [(index, _box_iou(box, target)) for index, target in enumerate(expected_boxes) if index not in used]
                    if matches and max(matches, key=lambda item: item[1])[1] >= 0.5:
                        index = max(matches, key=lambda item: item[1])[0]
                        used.add(index); true_positive += 1
                    else:
                        false_positive += 1
                false_negative += len(expected_boxes) - len(used)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {"text_polygon_hmean_iou_0_5": 2 * precision * recall / max(1e-9, precision + recall)}


def _partition_exact(expected_labels, predicted_groups) -> bool:
    """Compare partitions without assuming that group IDs share a namespace."""

    if len(expected_labels) != len(predicted_groups):
        return False
    expected_to_predicted = {}
    predicted_to_expected = {}
    for expected, predicted in zip(expected_labels, predicted_groups):
        if expected in expected_to_predicted and expected_to_predicted[expected] != predicted:
            return False
        if predicted in predicted_to_expected and predicted_to_expected[predicted] != expected:
            return False
        expected_to_predicted[expected] = predicted
        predicted_to_expected[predicted] = expected
    return True


def _quad_iou(first, second) -> float:
    """Return convex-quad IoU, treating invalid predictions as non-matches."""

    import cv2
    import numpy as np

    first = np.asarray(first, dtype=np.float32).reshape(4, 2)
    second = np.asarray(second, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return 0.0
    first_contour = first.reshape(-1, 1, 2)
    second_contour = second.reshape(-1, 1, 2)
    if not cv2.isContourConvex(first_contour) or not cv2.isContourConvex(second_contour):
        return 0.0
    first_area = abs(float(cv2.contourArea(first_contour)))
    second_area = abs(float(cv2.contourArea(second_contour)))
    if first_area <= 1e-6 or second_area <= 1e-6:
        return 0.0
    intersection, _polygon = cv2.intersectConvexConvex(first, second)
    return float(intersection) / max(1e-9, first_area + second_area - float(intersection))


def _match_quads(predicted_quads, expected_quads, threshold: float = 0.5) -> dict[int, int]:
    """Greedy global one-to-one IoU matching (expected index -> prediction).

    Axis-aligned bounds cheaply reject the overwhelmingly non-overlapping
    pairs before the exact convex-polygon intersection.  This keeps validation
    bounded for the 1,500-candidate decoder ceiling without changing the
    IoU>=threshold contract.
    """

    import cv2
    import numpy as np

    predicted = np.asarray(predicted_quads, dtype=np.float32).reshape(-1, 4, 2)
    expected = np.asarray(expected_quads, dtype=np.float32).reshape(-1, 4, 2)
    if len(predicted) == 0 or len(expected) == 0:
        return {}
    expected_min = expected.min(axis=1)
    expected_max = expected.max(axis=1)
    expected_area = np.asarray(
        [abs(float(cv2.contourArea(quad.reshape(-1, 1, 2)))) for quad in expected],
        dtype=np.float32,
    )
    candidates = []
    for predicted_index, quad in enumerate(predicted):
        if not np.isfinite(quad).all():
            continue
        predicted_area = abs(float(cv2.contourArea(quad.reshape(-1, 1, 2))))
        if predicted_area <= 1e-6:
            continue
        intersection_size = np.maximum(
            0.0,
            np.minimum(quad.max(axis=0), expected_max) - np.maximum(quad.min(axis=0), expected_min),
        )
        bbox_intersection = intersection_size[:, 0] * intersection_size[:, 1]
        denominator = predicted_area + expected_area - bbox_intersection
        possible = (bbox_intersection > 0) & (
            (denominator <= 0) | (bbox_intersection / np.maximum(denominator, 1e-9) >= threshold)
        )
        for expected_index in np.flatnonzero(possible):
            iou = _quad_iou(quad, expected[expected_index])
            if iou >= threshold:
                candidates.append((iou, predicted_index, int(expected_index)))
    candidates.sort(reverse=True)
    used_predictions: set[int] = set()
    used_expected: set[int] = set()
    matches: dict[int, int] = {}
    for iou, predicted_index, expected_index in candidates:
        if iou < threshold:
            break
        if predicted_index in used_predictions or expected_index in used_expected:
            continue
        used_predictions.add(predicted_index)
        used_expected.add(expected_index)
        matches[expected_index] = predicted_index
    return matches


def validate_table(model, loader, device) -> dict[str, float]:
    """Validate decoded cell polygons and learned relation partitions.

    ``row_level_accuracy`` remains a schedule-level exact metric, but now a
    schedule passes only when every expected cell quad has a one-to-one
    IoU>=0.5 match and the learned row-embedding partition is exactly
    equivalent to the row labels.  Column and joint relation accuracy use the
    same semantics for the column head and both heads together, respectively.
    """

    import torch
    true_positive = false_positive = false_negative = 0
    row_exact = 0
    column_exact = 0
    relation_exact = 0
    schedule_count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device))
            target_size = batch["cell_heatmap_target"].shape[-2:]
            network = model.module if hasattr(model, "module") else model
            top_k = int(getattr(network, "top_k", TABLE_CANDIDATE_TOP_K))
            decoded_batches = decode_table_candidates(
                outputs, threshold=0.2, top_k=top_k, target_size=target_size,
            )
            for candidates, corner_target, relation, valid in zip(
                decoded_batches, batch["corner_target"], batch["relation_target"],
                batch["corner_valid"][:, 0],
            ):
                predicted_quads = candidates.quads.detach().float().cpu().numpy()
                predicted_row_groups = candidates.row_groups.detach().cpu().tolist()
                predicted_column_groups = candidates.column_groups.detach().cpu().tolist()
                expected_points = (valid > 0.5).nonzero().tolist()
                expected_quads = []
                for y, x in expected_points:
                    offsets = corner_target[:, y, x].reshape(4, 2).float()
                    center_xy = torch.tensor([x, y], dtype=offsets.dtype)
                    expected_quads.append((offsets + center_xy).numpy())
                matched_predictions = _match_quads(predicted_quads, expected_quads)
                matches = len(matched_predictions)
                true_positive += matches
                false_positive += len(predicted_quads) - matches
                false_negative += len(expected_quads) - matches

                complete = matches == len(expected_quads) == len(predicted_quads)
                row_matches = column_matches = False
                if complete:
                    ordered_predictions = [matched_predictions[index] for index in range(len(expected_points))]
                    expected_rows = [int(relation[0, y, x]) for y, x in expected_points]
                    expected_columns = [int(relation[1, y, x]) for y, x in expected_points]
                    row_matches = _partition_exact(
                        expected_rows, [predicted_row_groups[index] for index in ordered_predictions],
                    )
                    column_matches = _partition_exact(
                        expected_columns, [predicted_column_groups[index] for index in ordered_predictions],
                    )
                row_exact += row_matches
                column_exact += column_matches
                relation_exact += row_matches and column_matches
                schedule_count += 1
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "cell_polygon_f1": 2 * precision * recall / max(1e-9, precision + recall),
        "row_level_accuracy": row_exact / max(1, schedule_count),
        "column_level_accuracy": column_exact / max(1, schedule_count),
        "table_relation_accuracy": relation_exact / max(1, schedule_count),
    }
