"""Leakage-safe datasets and recognition crop sampling."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .augmentation import jitter_quad, rectify_cell, sample_recipe, augment_image_and_objects
from .charset import normalize_transcription
from .master_split import MasterSplit
from .shards import iter_jsonl


RARE_CODES = {
    "보", "보건", "보건휴가", "경", "경조", "경조사", "병", "병가", "출", "출산",
    "출산휴가", "육", "육아", "육아휴직", "노", "노조", "노조휴가", "휴직",
}


def load_records(path: Path, master_split: MasterSplit, *, purpose: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    split_by_schedule: dict[str, str] = {}
    target_split = {
        "train": "train", "cv": "train", "select": "validation", "threshold": "validation",
        "quantize": "validation", "route": "validation", "test": "test", "ood": "ood_layout",
    }.get(purpose)
    if target_split is None:
        raise ValueError(f"unknown dataset purpose: {purpose}")
    for item in iter_jsonl(path):
        schedule_id = str(item["schedule_id"])
        record = master_split.require(schedule_id, item.get("split"))
        previous = split_by_schedule.setdefault(schedule_id, record.split)
        if previous != record.split:
            raise ValueError(f"derived records disagree on split for {schedule_id}")
        if record.split != target_split:
            continue
        master_split.authorize(schedule_id, purpose, item.get("split"))
        item["split"] = record.split
        item["cv_fold"] = record.cv_fold
        records.append(item)
    return records


def train_cv_partition(records: Sequence[Mapping[str, Any]], validation_fold: int):
    if validation_fold not in {0, 1, 2}:
        raise ValueError("validation_fold must be 0, 1 or 2")
    if any(item.get("split") != "train" for item in records):
        raise ValueError("grouped CV accepts Train records only")
    train = [item for item in records if int(item["cv_fold"]) != validation_fold]
    fold_validation = [item for item in records if int(item["cv_fold"]) == validation_fold]
    return train, fold_validation


class RareCodeCropSampler:
    """Class-aware recognition sampler with a per-schedule repetition cap."""

    def __init__(
        self, records: Sequence[Mapping[str, Any]], *, rare_weight: float = 3.0,
        max_per_schedule: int = 128, seed: int = 0,
    ) -> None:
        self.records = records
        self.weights = [rare_weight if item.get("canonical_code") in RARE_CODES else 1.0 for item in records]
        self.max_per_schedule = max_per_schedule
        self.rng = random.Random(seed)

    def sample_indices(self, count: int) -> list[int]:
        accepted: list[int] = []
        schedule_counts: Counter[str] = Counter()
        candidates = list(range(len(self.records)))
        attempts = 0
        while len(accepted) < count and attempts < count * 50:
            index = self.rng.choices(candidates, weights=self.weights, k=1)[0]
            schedule_id = str(self.records[index]["schedule_id"])
            attempts += 1
            if schedule_counts[schedule_id] >= self.max_per_schedule:
                continue
            accepted.append(index)
            schedule_counts[schedule_id] += 1
        if len(accepted) < count:
            raise RuntimeError("per-schedule sampler cap is too low for requested epoch size")
        return accepted


class RecognitionCropDataset:
    """Rectified crops using GT, simulated detector errors and table predictions."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        image_root: Path,
        *,
        charset: Sequence[str],
        training: bool,
        table_prediction_records: Mapping[tuple[str, str, int], Mapping[str, Any]] | None = None,
        seed: int = 0,
    ) -> None:
        self.records = [item for item in records if item.get("object_type") in {"shift_code", "name"}]
        self.image_root = image_root
        self.training = training
        self.charset = {char: index + 1 for index, char in enumerate(charset)}  # 0 is CTC blank
        self.table_predictions = table_prediction_records or {}
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def _mode(self, index: int) -> str:
        value = random.Random(self.seed + index * 104729).random()
        if self.table_predictions:
            return "gt" if value < 0.50 else "jitter" if value < 0.80 else "predicted"
        return "gt" if value < 0.60 else "jitter" if value < 0.90 else "strong"

    def __getitem__(self, index: int):
        import cv2
        import numpy as np
        import torch

        item = self.records[index]
        image = cv2.imread(str(self.image_root / str(item["image_path"])), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.image_root / str(item["image_path"]))
        quad = item["cell_polygon"]
        mode = self._mode(index) if self.training else "gt"
        prediction_key = (str(item["schedule_id"]), str(item.get("row_id")), int(item.get("day") or 0))
        if mode == "predicted" and prediction_key in self.table_predictions:
            prediction = self.table_predictions[prediction_key]
            if float(prediction.get("gt_iou", 0)) >= 0.45 and float(prediction.get("confidence", 0)) >= 0.35:
                quad = prediction["cell_polygon"]
            else:
                mode = "jitter"
        if mode in {"jitter", "strong"}:
            quad = jitter_quad(
                quad,
                seed=self.seed + index,
                jitter_ratio=(0.02, 0.08 if mode == "jitter" else 0.12),
                margin_ratio=(0.0, 0.12),
                rotation_deg=3.0,
            )
            if mode == "strong":
                points = np.asarray(quad, dtype=np.float32)
                fraction = random.Random(self.seed + index * 31).uniform(0.0, 0.05)
                points[0] += (points[3] - points[0]) * fraction
                points[1] += (points[2] - points[1]) * fraction
                quad = points.tolist()
        crop, bucket, content_width, _matrix = rectify_cell(image, quad)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(crop).permute(2, 0, 1)
        text = normalize_transcription(str(item["display_text"]))
        labels = torch.tensor([self.charset[char] for char in text], dtype=torch.long)
        return {
            "image": tensor,
            "labels": labels,
            "label_length": len(labels),
            "bucket": bucket,
            "content_width": content_width,
            "display_text": text,
            "canonical_code": item.get("canonical_code"),
            "crop_source": mode,
            "schedule_id": item["schedule_id"],
        }


class OnTheFlyScheduleDataset:
    def __init__(self, images: Sequence[Path], objects_by_image: Mapping[str, Sequence[Mapping[str, Any]]], seed: int = 0):
        self.images = images
        self.objects_by_image = objects_by_image
        self.seed = seed

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import cv2

        path = self.images[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        recipe = sample_recipe(self.seed + index)
        return augment_image_and_objects(image, self.objects_by_image[str(path)], recipe)


class DenseScheduleDataset:
    """Rasterize DBNet or table targets from schedule-level polygons."""

    def __init__(
        self, records: Sequence[Mapping[str, Any]], image_root: Path, *,
        kind: str, training: bool, long_side: int = 1280, seed: int = 0,
    ) -> None:
        if kind not in {"dbnet", "table"}:
            raise ValueError("kind must be dbnet or table")
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in records:
            grouped[str(item["image_path"])].append(item)
        self.images = sorted(grouped)
        self.grouped = grouped
        self.image_root = image_root
        self.kind = kind
        self.training = training
        self.long_side = long_side
        self.seed = seed

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import cv2
        import numpy as np
        import torch

        image_path = self.images[index]
        image = cv2.imread(str(self.image_root / image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.image_root / image_path)
        objects = [dict(item) for item in self.grouped[image_path]]
        if self.training:
            image, objects, _matrix, recipe = augment_image_and_objects(
                image, objects, sample_recipe(self.seed + index * 15485863)
            )
        else:
            recipe = None
        height, width = image.shape[:2]
        scale = self.long_side / max(height, width)
        resized_width, resized_height = max(4, round(width * scale)), max(4, round(height * scale))
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        # A fixed square canvas keeps DBNet/table batches stackable while the
        # schedule aspect ratio and polygons retain their common scale.
        canvas_height = ((self.long_side + 31) // 32) * 32
        canvas_width = canvas_height
        canvas = np.full((canvas_height, canvas_width, 3), 255, np.uint8)
        canvas[:resized_height, :resized_width] = image
        target_height, target_width = canvas_height // 4, canvas_width // 4
        mask = np.zeros((target_height, target_width), np.float32)
        threshold = np.zeros_like(mask)
        corners = np.zeros((8, target_height, target_width), np.float32)
        valid = np.zeros_like(mask)
        relation = np.zeros((2, target_height, target_width), np.float32)
        for item in objects:
            if item.get("ignore"):
                continue
            if self.kind == "dbnet" and "registration_high_confidence" in item and not item.get("registration_high_confidence"):
                continue
            polygon_key = "text_polygon" if self.kind == "dbnet" else "cell_polygon"
            polygon = np.asarray(item.get(polygon_key), np.float32) * (scale / 4.0)
            if len(polygon) < 3:
                continue
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1.0)
            if self.kind == "dbnet":
                expanded = cv2.boxPoints(cv2.minAreaRect(polygon)).astype(np.int32)
                cv2.fillPoly(threshold, [expanded], 1.0)
            else:
                center = polygon.mean(axis=0)
                x, y = int(round(center[0])), int(round(center[1]))
                if 0 <= x < target_width and 0 <= y < target_height:
                    cv2.circle(mask, (x, y), 2, 1.0, -1)
                    valid[y, x] = 1.0
                    quad = polygon[:4]
                    corners[:, y, x] = (quad - center).reshape(-1)
                    relation[0, y, x] = float(item.get("row_index") or 0)
                    relation[1, y, x] = float(item.get("day") or 0)
        tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1)
        return {
            "image": tensor,
            "probability_target": torch.from_numpy(mask[None]),
            "threshold_target": torch.from_numpy(threshold[None]),
            "cell_heatmap_target": torch.from_numpy(mask[None]),
            "corner_target": torch.from_numpy(corners),
            "corner_valid": torch.from_numpy(valid[None]),
            "relation_target": torch.from_numpy(relation),
            "schedule_id": str(objects[0]["schedule_id"]) if objects else image_path,
            "augmentation_recipe": recipe,
        }


def recognition_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    widths = {int(item["bucket"]) for item in batch}
    if len(widths) != 1:
        raise ValueError("recognizer micro-batch must contain one width bucket")
    labels = torch.cat([item["labels"] for item in batch])
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "labels": labels,
        "label_lengths": torch.tensor([item["label_length"] for item in batch], dtype=torch.long),
        "bucket": next(iter(widths)),
        "display_text": [item["display_text"] for item in batch],
        "schedule_id": [item["schedule_id"] for item in batch],
        "crop_source": [item["crop_source"] for item in batch],
    }


class WidthBucketBatchSampler:
    """Group 160/320/640 crops while retaining batch=1 as a legal fallback."""

    def __init__(
        self, dataset: RecognitionCropDataset, batch_sizes: Mapping[int, int], shuffle: bool = True,
        seed: int = 0, indices: Sequence[int] | None = None,
    ):
        self.dataset = dataset
        self.batch_sizes = {int(width): max(1, int(size)) for width, size in batch_sizes.items()}
        self.shuffle = shuffle
        self.seed = seed
        buckets: dict[int, list[int]] = defaultdict(list)
        # Bounding-box aspect gives a cheap deterministic bucket prediction.
        for index in (list(indices) if indices is not None else range(len(dataset.records))):
            item = dataset.records[index]
            polygon = item["cell_polygon"]
            width = max(abs(float(polygon[1][0]) - float(polygon[0][0])), 1.0)
            height = max(abs(float(polygon[3][1]) - float(polygon[0][1])), 1.0)
            buckets[160 if 48 * width / height <= 160 else 320 if 48 * width / height <= 320 else 640].append(index)
        self.buckets = buckets

    def __iter__(self):
        rng = random.Random(self.seed)
        batches = []
        for width, indices in self.buckets.items():
            values = list(indices)
            if self.shuffle:
                rng.shuffle(values)
            batch_size = self.batch_sizes.get(width, 1)
            batches.extend(values[offset:offset + batch_size] for offset in range(0, len(values), batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        import math
        return sum(math.ceil(len(indices) / self.batch_sizes.get(width, 1)) for width, indices in self.buckets.items())
