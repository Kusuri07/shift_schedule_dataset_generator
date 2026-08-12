"""Model-family Validation metrics used for checkpoint selection only."""

from __future__ import annotations

from collections import defaultdict

from .evaluation import edit_distance


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


def validate_table(model, loader, device) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    true_positive = false_positive = false_negative = 0
    row_exact = 0
    schedule_count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device))
            heatmap = functional.interpolate(outputs["cell_heatmap"], size=batch["cell_heatmap_target"].shape[-2:], mode="bilinear", align_corners=False).sigmoid()
            for predicted, expected, relation, valid in zip(
                heatmap[:, 0].cpu(), batch["cell_heatmap_target"][:, 0], batch["relation_target"], batch["corner_valid"][:, 0]
            ):
                local = functional.max_pool2d(predicted[None, None], 3, stride=1, padding=1)[0, 0]
                peak_mask = (predicted >= local) & (predicted > 0.2)
                peak_points = peak_mask.nonzero()
                if len(peak_points) > 1500:
                    peak_scores = predicted[peak_mask]
                    keep = torch.topk(peak_scores, 1500).indices
                    peak_points = peak_points[keep]
                predicted_points = peak_points.tolist()
                expected_points = (valid > 0.5).nonzero().tolist()
                used = set()
                for y, x in predicted_points:
                    candidates = [(index, (y - target_y) ** 2 + (x - target_x) ** 2) for index, (target_y, target_x) in enumerate(expected_points) if index not in used]
                    if candidates and min(candidates, key=lambda item: item[1])[1] <= 9:
                        used.add(min(candidates, key=lambda item: item[1])[0]); true_positive += 1
                    else:
                        false_positive += 1
                false_negative += len(expected_points) - len(used)
                expected_rows = defaultdict(int)
                for y, x in expected_points:
                    expected_rows[int(relation[0, y, x])] += 1
                predicted_y = sorted(y for y, _x in predicted_points)
                tolerance = 3
                predicted_rows = []
                for y in predicted_y:
                    if not predicted_rows or y - predicted_rows[-1][-1] > tolerance:
                        predicted_rows.append([y])
                    else:
                        predicted_rows[-1].append(y)
                row_exact += sorted(expected_rows.values()) == sorted(len(row) for row in predicted_rows)
                schedule_count += 1
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "cell_polygon_f1": 2 * precision * recall / max(1e-9, precision + recall),
        "row_level_accuracy": row_exact / max(1, schedule_count),
    }
