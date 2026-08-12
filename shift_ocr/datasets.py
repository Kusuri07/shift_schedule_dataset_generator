"""Leakage-safe datasets and recognition crop sampling."""

from __future__ import annotations

import json
import hashlib
import math
import multiprocessing
import random
from array import array
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .augmentation import jitter_quad, rectify_cell, sample_recipe, augment_image_and_objects
from .charset import normalize_transcription
from .master_split import MasterSplit
from .shards import (
    ParquetImageStore, iter_jsonl, iter_verified_index_rows,
    recognition_width_bucket, verify_parquet_index,
)


RARE_CODES = {
    "보", "보건", "보건휴가", "경", "경조", "경조사", "병", "병가", "출", "출산",
    "출산휴가", "육", "육아", "육아휴직", "노", "노조", "노조휴가", "휴직",
}


# A relation head has one label at each detected cell centre.  The source
# annotations' ``row_index`` and ``day`` fields describe the nursing roster,
# not the physical table: headers have no row index, while group/name/summary
# columns have no day.  Derive the relation labels from the cell geometry
# instead.  The thresholds below deliberately compare the complete projected
# interval (centre, overlap and extent), so a row/column-spanning merged cell
# cannot bridge the bands it spans.
_TOPOLOGY_MIN_EXTENT_RATIO = 0.67
_TOPOLOGY_MIN_COMMON_OVERLAP_RATIO = 0.60
_TOPOLOGY_MAX_CENTER_SPREAD_RATIO = 0.25
_TOPOLOGY_ABSOLUTE_TOLERANCE_PX = 1.5


def _polygon_projection(
    polygon: Sequence[Sequence[float]] | None, axis: int,
) -> tuple[float, float] | None:
    """Return the axis-aligned projection of a valid cell polygon."""

    if not polygon or len(polygon) < 3:
        return None
    try:
        values = [float(point[axis]) for point in polygon if len(point) > axis]
    except (TypeError, ValueError):
        return None
    if len(values) < 3 or not all(value == value and abs(value) != float("inf") for value in values):
        return None
    low, high = min(values), max(values)
    if high - low <= 1e-6:
        return None
    return low, high


def _polygon_linear_projection(
    polygon: Sequence[Sequence[float]] | None, covector: tuple[float, float],
) -> tuple[float, float] | None:
    """Project a polygon onto one coordinate of an estimated table basis."""

    if not polygon or len(polygon) < 3:
        return None
    try:
        values = [
            float(point[0]) * covector[0] + float(point[1]) * covector[1]
            for point in polygon if len(point) >= 2
        ]
    except (TypeError, ValueError):
        return None
    if len(values) < 3 or not all(math.isfinite(value) for value in values):
        return None
    low, high = min(values), max(values)
    if high - low <= 1e-6:
        return None
    return low, high


def _mean_undirected_axis(vectors: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """Return a robust mean direction where an axis and its negation are equal."""

    cosine_sum = 0.0
    sine_sum = 0.0
    directed_x = 0.0
    directed_y = 0.0
    count = 0
    for x_value, y_value in vectors:
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            continue
        if math.hypot(x_value, y_value) <= 1e-6:
            continue
        magnitude = math.hypot(x_value, y_value)
        angle = math.atan2(y_value, x_value)
        cosine_sum += math.cos(2.0 * angle)
        sine_sum += math.sin(2.0 * angle)
        directed_x += x_value / magnitude
        directed_y += y_value / magnitude
        count += 1
    if count == 0 or math.hypot(cosine_sum, sine_sum) <= 1e-9:
        return None
    angle = 0.5 * math.atan2(sine_sum, cosine_sum)
    axis = (math.cos(angle), math.sin(angle))
    # Polygon vertices are ordered in reference-grid order, so use their mean
    # directed edge to resolve the unavoidable 180-degree axis ambiguity.  IDs
    # then retain top-to-bottom and left-to-right ordering after registration.
    if axis[0] * directed_x + axis[1] * directed_y < 0:
        axis = (-axis[0], -axis[1])
    return axis


def _table_coordinate_covectors(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Estimate affine row/column coordinates from ordered cell edges.

    Registered phone photos may rotate or shear the whole grid.  Projecting
    their cells onto screen x/y splits one physical row across the image.  The
    first/fourth polygon edges retain the reference grid directions through
    registration, so their dual basis removes global rotation and affine shear
    before interval clustering.  Axis-aligned coordinates remain the fallback
    for malformed or nearly singular geometry.
    """

    horizontal_edges: list[tuple[float, float]] = []
    vertical_edges: list[tuple[float, float]] = []
    for item in records:
        polygon = item.get("cell_polygon")
        if not polygon or len(polygon) < 4:
            continue
        try:
            points = [(float(point[0]), float(point[1])) for point in polygon[:4]]
        except (TypeError, ValueError, IndexError):
            continue
        if not all(math.isfinite(value) for point in points for value in point):
            continue
        horizontal_edges.extend((
            (points[1][0] - points[0][0], points[1][1] - points[0][1]),
            (points[2][0] - points[3][0], points[2][1] - points[3][1]),
        ))
        vertical_edges.extend((
            (points[3][0] - points[0][0], points[3][1] - points[0][1]),
            (points[2][0] - points[1][0], points[2][1] - points[1][1]),
        ))

    horizontal = _mean_undirected_axis(horizontal_edges)
    vertical = _mean_undirected_axis(vertical_edges)
    if horizontal is None or vertical is None:
        return (0.0, 1.0), (1.0, 0.0)
    determinant = horizontal[0] * vertical[1] - horizontal[1] * vertical[0]
    if abs(determinant) <= 0.10:
        return (0.0, 1.0), (1.0, 0.0)

    # Inverse of the basis matrix [horizontal vertical].  Its first row is the
    # column covector and its second row the row covector.
    column_covector = (vertical[1] / determinant, -vertical[0] / determinant)
    row_covector = (-horizontal[1] / determinant, horizontal[0] / determinant)
    return row_covector, column_covector


def _can_join_topology_band(
    cluster: Mapping[str, float], interval: tuple[float, float], *, base_extent: float,
) -> bool:
    """Test a projected interval against a cluster without transitive joins."""

    low, high = interval
    extent = high - low
    center = (low + high) / 2.0
    min_extent = min(float(cluster["min_extent"]), extent)
    max_extent = max(float(cluster["max_extent"]), extent)
    if min_extent / max_extent < _TOPOLOGY_MIN_EXTENT_RATIO:
        return False

    min_center = min(float(cluster["min_center"]), center)
    max_center = max(float(cluster["max_center"]), center)
    geometry_tolerance = max(
        _TOPOLOGY_ABSOLUTE_TOLERANCE_PX,
        _TOPOLOGY_MAX_CENTER_SPREAD_RATIO * base_extent,
    )
    if max_center - min_center > geometry_tolerance:
        return False

    # A relative comparison alone is too permissive for very broad merged
    # cells: e.g. a title spanning 35 columns and a notice spanning 37 columns
    # overlap by more than 90%, yet are different physical spans.  Compare
    # both projected boundaries against a tolerance based on the schedule's
    # ordinary cell size, not on the merged cell's own width/height.
    if max(float(cluster["max_interval_low"]), low) - min(
        float(cluster["min_interval_low"]), low,
    ) > geometry_tolerance:
        return False
    if max(float(cluster["max_interval_high"]), high) - min(
        float(cluster["min_interval_high"]), high,
    ) > geometry_tolerance:
        return False

    # Requiring a common intersection across the whole cluster avoids a
    # chain of individually-overlapping intervals joining neighbouring rows.
    max_low = max(float(cluster["max_low"]), low)
    min_high = min(float(cluster["min_high"]), high)
    common_overlap = max(0.0, min_high - max_low)
    return common_overlap / max_extent >= _TOPOLOGY_MIN_COMMON_OVERLAP_RATIO


def _cluster_projected_intervals(
    projections: Sequence[tuple[float, float] | None],
) -> list[int | None]:
    """Cluster near-identical projected spans into stable ordered band IDs."""

    # Synthetic schedules repeat the exact same row/column span hundreds of
    # times.  Collapsing exact projections before clustering keeps topology
    # construction effectively proportional to the number of table bands,
    # which matters for a 10,000-image dataset.
    occurrences: dict[tuple[float, float], list[int]] = defaultdict(list)
    for index, projection in enumerate(projections):
        if projection is not None:
            occurrences[(round(projection[0], 6), round(projection[1], 6))].append(index)

    weighted_extents = sorted(
        projection[1] - projection[0]
        for projection in projections
        if projection is not None
    )
    base_extent = (
        weighted_extents[len(weighted_extents) // 2]
        if weighted_extents else 1.0
    )

    unique_intervals = sorted(
        occurrences,
        key=lambda interval: (
            (interval[0] + interval[1]) / 2.0,
            interval[1] - interval[0],
            interval[0],
        ),
    )
    clusters: list[dict[str, float]] = []
    interval_cluster: dict[tuple[float, float], int] = {}
    for interval in unique_intervals:
        low, high = interval
        extent = high - low
        center = (low + high) / 2.0
        compatible = [
            (abs(float(cluster["mean_center"]) - center), cluster_index)
            for cluster_index, cluster in enumerate(clusters)
            if _can_join_topology_band(cluster, interval, base_extent=base_extent)
        ]
        if compatible:
            _, cluster_index = min(compatible)
            cluster = clusters[cluster_index]
            weight = float(len(occurrences[interval]))
            total_weight = float(cluster["weight"]) + weight
            cluster["mean_center"] = (
                float(cluster["mean_center"]) * float(cluster["weight"]) + center * weight
            ) / total_weight
            cluster["weight"] = total_weight
            cluster["min_extent"] = min(float(cluster["min_extent"]), extent)
            cluster["max_extent"] = max(float(cluster["max_extent"]), extent)
            cluster["min_center"] = min(float(cluster["min_center"]), center)
            cluster["max_center"] = max(float(cluster["max_center"]), center)
            cluster["max_low"] = max(float(cluster["max_low"]), low)
            cluster["min_high"] = min(float(cluster["min_high"]), high)
            cluster["min_interval_low"] = min(float(cluster["min_interval_low"]), low)
            cluster["max_interval_low"] = max(float(cluster["max_interval_low"]), low)
            cluster["min_interval_high"] = min(float(cluster["min_interval_high"]), high)
            cluster["max_interval_high"] = max(float(cluster["max_interval_high"]), high)
        else:
            cluster_index = len(clusters)
            clusters.append({
                "mean_center": center,
                "weight": float(len(occurrences[interval])),
                "min_extent": extent,
                "max_extent": extent,
                "min_center": center,
                "max_center": center,
                "max_low": low,
                "min_high": high,
                "min_interval_low": low,
                "max_interval_low": low,
                "min_interval_high": high,
                "max_interval_high": high,
            })
        interval_cluster[interval] = cluster_index

    # Cluster creation is already centre-ordered in the normal case.  Sort
    # explicitly to make IDs deterministic even if a broad, shifted interval
    # joined an earlier representative.
    ordered_clusters = sorted(
        range(len(clusters)),
        key=lambda index: (float(clusters[index]["mean_center"]), index),
    )
    normalized_id = {cluster_index: order + 1 for order, cluster_index in enumerate(ordered_clusters)}
    labels: list[int | None] = [None] * len(projections)
    for interval, indices in occurrences.items():
        label = normalized_id[interval_cluster[interval]]
        for index in indices:
            labels[index] = label
    return labels


def derive_table_topology_ids(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[int | None, int | None]]:
    """Derive physical ``(row, column)`` IDs from every cell polygon.

    Rows are clusters of vertical polygon projections and columns are
    clusters of horizontal projections.  Equal IDs therefore mean equal
    physical bands across header, body and summary cells; merged spans receive
    their own band instead of collapsing the rows/columns they cover.
    """

    row_covector, column_covector = _table_coordinate_covectors(records)
    row_projections = [
        _polygon_linear_projection(item.get("cell_polygon"), row_covector)
        for item in records
    ]
    column_projections = [
        _polygon_linear_projection(item.get("cell_polygon"), column_covector)
        for item in records
    ]
    row_ids = _cluster_projected_intervals(row_projections)
    column_ids = _cluster_projected_intervals(column_projections)
    return list(zip(row_ids, column_ids))


class _SharedAbsoluteEpoch:
    """Small spawn-safe epoch cell shared with persistent DataLoader workers.

    Windows workers receive a spawned copy of the dataset, so a plain integer
    set in the parent remains frozen at worker-start time.  A synchronized
    multiprocessing value keeps the absolute epoch visible without rebuilding
    workers every epoch.
    """

    def __init__(self) -> None:
        # RawValue is reconstructed through the Windows spawn reduction path
        # without a SemLock.  Aligned 64-bit reads/writes are atomic on the
        # supported desktop platforms and set_epoch only runs between epochs.
        self._value = multiprocessing.RawValue("q", 0)

    def get(self) -> int:
        return int(self._value.value)

    def set(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("absolute dataset epoch cannot be negative")
        self._value.value = epoch


class _EpochAwareAugmentation:
    """Shared absolute-epoch API for deterministic on-the-fly augmentation."""

    _EPOCH_STRIDE = 1_000_000_007
    _INDEX_STRIDE = 15_485_863
    _STREAM_STRIDE = 32_452_843

    def _initialize_epoch_state(self) -> None:
        self._shared_absolute_epoch = _SharedAbsoluteEpoch()

    @property
    def absolute_epoch(self) -> int:
        return self._shared_absolute_epoch.get()

    def set_epoch(self, epoch: int) -> None:
        """Select the absolute (resume-aware) epoch used by future samples."""

        self._shared_absolute_epoch.set(epoch)

    def _augmentation_seed(self, index: int, *, stream: int = 0) -> int:
        # Distinct strides avoid collisions such as epoch=1/index=0 and
        # epoch=0/index=1 while preserving the base+epoch+index contract.
        return (
            int(self.seed)
            + self.absolute_epoch * self._EPOCH_STRIDE
            + int(index) * self._INDEX_STRIDE
            + int(stream) * self._STREAM_STRIDE
        )


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


@dataclass(frozen=True, slots=True)
class ParquetImageEntry:
    """Compact, spawn-safe address of one image row group.

    Only image-level metadata is copied into Windows workers.  Object
    annotations remain in Parquet and are loaded into a worker-local bounded
    cache when an image is actually sampled.
    """

    shard_dir_index: int
    image_path: str
    schedule_id: str
    split: str
    cv_fold: int
    source_domain: str
    object_count: int


def load_parquet_image_entries(
    shard_dirs: Sequence[Path], master_split: MasterSplit, *, purpose: str,
    source_domain: str | None = None, include_cv_fold: int | None = None,
    exclude_cv_fold: int | None = None,
) -> list[ParquetImageEntry]:
    """Read the compact image index without materializing object annotations."""

    if include_cv_fold is not None and exclude_cv_fold is not None:
        raise ValueError("include_cv_fold and exclude_cv_fold are mutually exclusive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - training dependency
        raise RuntimeError("lazy Parquet datasets require pyarrow") from exc

    entries: list[ParquetImageEntry] = []
    target_split = {
        "train": "train", "cv": "train", "select": "validation",
        "threshold": "validation", "quantize": "validation",
        "route": "validation", "test": "test", "ood": "ood_layout",
    }.get(purpose)
    if target_split is None:
        raise ValueError(f"unknown dataset purpose: {purpose}")
    for shard_dir_index, shard_dir in enumerate(shard_dirs):
        # Authenticate the compact index before exposing even its first row.
        # This costs one streaming pass and prevents a forged early batch from
        # entering training before an end-of-iteration checksum check.
        verify_parquet_index(Path(shard_dir), "image")
        index_path = Path(shard_dir) / "image_index.parquet"
        parquet = pq.ParquetFile(index_path)
        available = set(parquet.schema_arrow.names)
        required = {"image_path", "schedule_id", "split", "object_count"}
        missing = required - available
        if missing:
            raise ValueError(f"image index is missing fields {sorted(missing)}: {index_path}")
        columns = sorted(required | ({"cv_fold", "source_domain", "master_split_sha256"} & available))
        for batch in parquet.iter_batches(columns=columns, batch_size=16_384):
            for item in batch.to_pylist():
                schedule_id = str(item["schedule_id"])
                declared_split = str(item["split"])
                split_record = master_split.require(schedule_id, declared_split)
                if split_record.split != target_split:
                    continue
                master_split.authorize(schedule_id, purpose, declared_split)
                declared_master = item.get("master_split_sha256")
                if declared_master not in (None, "", master_split.metadata.get("split_sha256")):
                    raise ValueError(f"image index master split hash mismatch: {schedule_id}")
                cv_fold = int(item.get("cv_fold", split_record.cv_fold))
                if cv_fold != split_record.cv_fold:
                    raise ValueError(f"image index CV fold mismatch: {schedule_id}")
                if include_cv_fold is not None and cv_fold != include_cv_fold:
                    continue
                if exclude_cv_fold is not None and cv_fold == exclude_cv_fold:
                    continue
                domain = str(item.get("source_domain") or "synthetic")
                if source_domain is not None and domain != source_domain:
                    continue
                entries.append(ParquetImageEntry(
                    shard_dir_index=shard_dir_index,
                    image_path=str(item["image_path"]),
                    schedule_id=schedule_id,
                    split=declared_split,
                    cv_fold=cv_fold,
                    source_domain=domain,
                    object_count=int(item["object_count"]),
                ))
    entries.sort(key=lambda item: (
        item.schedule_id, item.image_path, item.shard_dir_index,
    ))
    return entries


class _WorkerLocalParquetImages:
    """Mixin that opens ParquetImageStore lazily inside each worker process."""

    def _initialize_parquet_images(
        self, shard_dirs: Sequence[Path], master_split: MasterSplit, *, purpose: str,
    ) -> None:
        self.shard_dirs = tuple(Path(path) for path in shard_dirs)
        self.master_split = master_split
        self.parquet_purpose = purpose
        self._parquet_stores: dict[int, ParquetImageStore] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        # ParquetFile handles are not spawn-pickleable and should never be
        # inherited across processes.  Each worker opens only the shards it
        # samples and ParquetImageStore keeps a bounded file LRU.
        state["_parquet_stores"] = {}
        return state

    def _parquet_records(self, entry: ParquetImageEntry) -> list[dict[str, Any]]:
        store = self._parquet_stores.get(entry.shard_dir_index)
        if store is None:
            store = ParquetImageStore(
                self.shard_dirs[entry.shard_dir_index], self.master_split,
            )
            self._parquet_stores[entry.shard_dir_index] = store
        records = store.load_image(entry.image_path, purpose=self.parquet_purpose)
        if len(records) != entry.object_count:
            raise ValueError(
                f"image index object count mismatch for {entry.image_path}: "
                f"index={entry.object_count}, rows={len(records)}"
            )
        if any(str(item.get("schedule_id")) != entry.schedule_id for item in records):
            raise ValueError(f"image row group schedule mismatch: {entry.image_path}")
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


class RecognitionCropDataset(_EpochAwareAugmentation):
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
        self._initialize_epoch_state()

    def __len__(self) -> int:
        return len(self.records)

    def _mode(self, index: int) -> str:
        value = random.Random(self._augmentation_seed(index)).random()
        if self.table_predictions:
            return "gt" if value < 0.50 else "jitter" if value < 0.80 else "predicted"
        return "gt" if value < 0.60 else "jitter" if value < 0.90 else "strong"

    def __getitem__(self, index: int):
        return self._build_recognition_item(self.records[index], index)

    def _build_recognition_item(self, item: Mapping[str, Any], index: int):
        import cv2
        import numpy as np
        import torch

        image = cv2.imread(str(self.image_root / str(item["image_path"])), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.image_root / str(item["image_path"]))
        quad = item["cell_polygon"]
        augmentation_seed = self._augmentation_seed(index)
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
                seed=augmentation_seed,
                jitter_ratio=(0.02, 0.08 if mode == "jitter" else 0.12),
                margin_ratio=(0.0, 0.12),
                rotation_deg=3.0,
            )
            if mode == "strong":
                points = np.asarray(quad, dtype=np.float32)
                fraction = random.Random(
                    self._augmentation_seed(index, stream=1)
                ).uniform(0.0, 0.05)
                points[0] += (points[3] - points[0]) * fraction
                points[1] += (points[2] - points[1]) * fraction
                quad = points.tolist()
        indexed_bucket = recognition_width_bucket(item)
        crop, bucket, content_width, _matrix = rectify_cell(
            image, quad, target_width=indexed_bucket,
        )
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
            "augmentation_epoch": self.absolute_epoch,
            "augmentation_seed": augmentation_seed,
            "schedule_id": item["schedule_id"],
        }


class OnTheFlyScheduleDataset(_EpochAwareAugmentation):
    def __init__(self, images: Sequence[Path], objects_by_image: Mapping[str, Sequence[Mapping[str, Any]]], seed: int = 0):
        self.images = images
        self.objects_by_image = objects_by_image
        self.seed = seed
        self._initialize_epoch_state()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import cv2

        path = self.images[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        recipe = sample_recipe(self._augmentation_seed(index))
        image, objects, matrix, recipe_details = augment_image_and_objects(
            image, self.objects_by_image[str(path)], recipe,
        )
        return image, objects, matrix, {
            "kind": "sampled", "epoch": self.absolute_epoch, **recipe_details,
        }


class DenseScheduleDataset(_EpochAwareAugmentation):
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
        topology_grouped: dict[str, tuple[array, array]] = {}
        if kind == "table":
            for image_path, image_records in grouped.items():
                topology = derive_table_topology_ids(image_records)
                # Compact arrays avoid duplicating millions of annotation
                # dictionaries when the 10,000-schedule dataset is loaded.
                # Zero is reserved for an invalid polygon; real IDs start at 1.
                topology_grouped[image_path] = (
                    array("I", (row_id or 0 for row_id, _column_id in topology)),
                    array("I", (column_id or 0 for _row_id, column_id in topology)),
                )
        self.images = sorted(grouped)
        self.grouped = grouped
        self.table_topology = topology_grouped
        self.image_root = image_root
        self.kind = kind
        self.training = training
        self.long_side = long_side
        self.seed = seed
        self._initialize_epoch_state()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        objects = [dict(item) for item in self.grouped[image_path]]
        if self.kind == "table":
            row_ids, column_ids = self.table_topology[image_path]
            for item, row_id, column_id in zip(objects, row_ids, column_ids):
                item["_table_topology_row_id"] = int(row_id) or None
                item["_table_topology_column_id"] = int(column_id) or None
        return self._build_dense_item(image_path, objects, index)

    def _build_dense_item(
        self, image_path: str, objects: list[dict[str, Any]], index: int,
    ):
        import cv2
        import numpy as np
        import torch

        image = cv2.imread(str(self.image_root / image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.image_root / image_path)
        if self.training:
            augmentation_seed = self._augmentation_seed(index)
            image, objects, _matrix, recipe = augment_image_and_objects(
                image, objects, sample_recipe(augmentation_seed)
            )
            recipe = {"kind": "sampled", "epoch": self.absolute_epoch, **recipe}
        else:
            # ``None`` cannot be collated by PyTorch's default collate
            # function.  Keep this metadata JSON-serializable and the same
            # mapping type in train/validation so dense validation batches can
            # use an ordinary DataLoader.
            recipe = {"kind": "identity", "epoch": self.absolute_epoch}
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
            # The explicit Arrow schema represents fields that do not apply to
            # synthetic rows as null.  Only an explicit low-confidence real
            # registration should suppress DBNet supervision; absent/null are
            # ordinary synthetic (or legacy) annotations and remain valid.
            if self.kind == "dbnet" and item.get("registration_high_confidence") is False:
                continue
            polygon_key = "text_polygon" if self.kind == "dbnet" else "cell_polygon"
            polygon = np.asarray(item.get(polygon_key), np.float32) * (scale / 4.0)
            if len(polygon) < 3:
                continue
            if self.kind == "dbnet":
                # DBNet is the only dense segmentation target.  The table
                # head is a CenterNet-style detector and must not inherit this
                # full-polygon plateau (which would create many equal local
                # maxima for one cell).
                cv2.fillPoly(mask, [polygon.astype(np.int32)], 1.0)
                expanded = cv2.boxPoints(cv2.minAreaRect(polygon)).astype(np.int32)
                cv2.fillPoly(threshold, [expanded], 1.0)
            else:
                center = polygon.mean(axis=0)
                x, y = int(round(center[0])), int(round(center[1]))
                if 0 <= x < target_width and 0 <= y < target_height:
                    # A compact Gaussian gives exactly one peak per cell while
                    # still supervising neighbouring pixels.  Radius is
                    # bounded so adjacent narrow schedule cells do not merge.
                    cell_width = float(np.ptp(polygon[:, 0]))
                    cell_height = float(np.ptp(polygon[:, 1]))
                    radius = max(1, min(4, int(round(min(cell_width, cell_height) * 0.15))))
                    diameter = radius * 2 + 1
                    coordinates = np.arange(diameter, dtype=np.float32) - radius
                    gaussian = np.exp(
                        -(coordinates[:, None] ** 2 + coordinates[None, :] ** 2)
                        / (2.0 * max(radius / 3.0, 1.0 / 3.0) ** 2)
                    ).astype(np.float32)
                    left, right = min(x, radius), min(target_width - x - 1, radius)
                    top, bottom = min(y, radius), min(target_height - y - 1, radius)
                    target_patch = mask[y - top:y + bottom + 1, x - left:x + right + 1]
                    gaussian_patch = gaussian[
                        radius - top:radius + bottom + 1,
                        radius - left:radius + right + 1,
                    ]
                    np.maximum(target_patch, gaussian_patch, out=target_patch)
                    valid[y, x] = 1.0
                    quad = polygon[:4]
                    # Offsets are relative to the integer heatmap sample where
                    # they are supervised and later decoded, not the fractional
                    # polygon centroid.  This prevents a sub-pixel translation
                    # error after round(center).
                    corners[:, y, x] = (quad - np.asarray([x, y], np.float32)).reshape(-1)
                    # Roster semantics (row_index/day) leave every header and
                    # every non-day column at zero.  The geometry-derived IDs
                    # preserve the actual header/body/summary topology and
                    # survive on-the-fly homographies as object metadata.
                    row_id = item.get("_table_topology_row_id")
                    column_id = item.get("_table_topology_column_id")
                    if row_id is not None and column_id is not None:
                        relation[0, y, x] = float(row_id)
                        relation[1, y, x] = float(column_id)
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


class LazyParquetDenseScheduleDataset(_WorkerLocalParquetImages, DenseScheduleDataset):
    """Dense targets backed by image-indexed, worker-local Parquet reads."""

    def __init__(
        self, entries: Sequence[ParquetImageEntry], shard_dirs: Sequence[Path],
        master_split: MasterSplit, image_root: Path, *, purpose: str, kind: str,
        training: bool, long_side: int = 1280, seed: int = 0,
    ) -> None:
        if kind not in {"dbnet", "table"}:
            raise ValueError("kind must be dbnet or table")
        self.entries = tuple(entries)
        self.image_root = image_root
        self.kind = kind
        self.training = training
        self.long_side = long_side
        self.seed = seed
        self._initialize_epoch_state()
        self._initialize_parquet_images(shard_dirs, master_split, purpose=purpose)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        objects = [dict(item) for item in self._parquet_records(entry)]
        if self.kind == "table":
            for item, (row_id, column_id) in zip(objects, derive_table_topology_ids(objects)):
                item["_table_topology_row_id"] = row_id
                item["_table_topology_column_id"] = column_id
        return self._build_dense_item(entry.image_path, objects, index)


@dataclass(frozen=True, slots=True)
class ParquetRecognitionEntry:
    """Compact address and sampling metadata for one recognition object."""

    shard_dir_index: int
    image_path: str
    schedule_id: str
    split: str
    cv_fold: int
    source_domain: str
    row_group_object_index: int
    image_object_index: int
    object_type: str
    canonical_code: str | None
    width_bucket: int
    master_split_sha256: str


class ParquetRecognitionBatchSampler:
    """Stream width-homogeneous crop batches from recognition_index.parquet.

    The sampler buffers at most a few hundred already-formed batches.  It does
    not construct a Python object entry for every crop in the 10,000-schedule
    corpus, while still retaining deterministic epoch shuffling, rare-code
    weighting and the per-schedule repetition cap.
    """

    _REQUIRED_FIELDS = {
        "image_path", "schedule_id", "split", "cv_fold", "source_domain",
        "row_group_object_index", "image_object_index", "object_type",
        "canonical_code", "width_bucket",
        "master_split_sha256",
    }

    def __init__(
        self, shard_dirs: Sequence[Path], master_split: MasterSplit, *, purpose: str,
        batch_sizes: Mapping[int, int], training: bool, source_domain: str | None = None,
        include_cv_fold: int | None = None, exclude_cv_fold: int | None = None,
        shuffle: bool = True, seed: int = 0, max_per_schedule: int = 128,
        shuffle_buffer_batches: int = 256,
    ) -> None:
        if include_cv_fold is not None and exclude_cv_fold is not None:
            raise ValueError("include_cv_fold and exclude_cv_fold are mutually exclusive")
        self.shard_dirs = tuple(Path(path) for path in shard_dirs)
        self.master_split = master_split
        self.purpose = purpose
        self.batch_sizes = {int(key): max(1, int(value)) for key, value in batch_sizes.items()}
        self.training = training
        self.source_domain = source_domain
        self.include_cv_fold = include_cv_fold
        self.exclude_cv_fold = exclude_cv_fold
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.max_per_schedule = max(1, int(max_per_schedule))
        self.shuffle_buffer_batches = max(1, int(shuffle_buffer_batches))
        self._declared_record_count = 0
        for shard_dir in self.shard_dirs:
            manifest_path = shard_dir / "shards.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._declared_record_count += int(manifest.get("recognition_index_record_count", 0))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        # This is a conservative progress-length estimate.  The iterator is
        # authoritative because per-schedule weighted sampling and width
        # buckets determine exact partial-batch counts lazily.
        minimum_batch = min(self.batch_sizes.values(), default=1)
        count = self._declared_record_count
        return max(1, math.ceil(count / minimum_batch)) if count else 0

    def _iter_entries(self) -> Iterator[ParquetRecognitionEntry]:
        target_split = {
            "train": "train", "cv": "train", "select": "validation",
            "threshold": "validation", "quantize": "validation",
            "route": "validation", "test": "test", "ood": "ood_layout",
        }.get(self.purpose)
        if target_split is None:
            raise ValueError(f"unknown dataset purpose: {self.purpose}")
        for shard_dir_index, shard_dir in enumerate(self.shard_dirs):
            path = shard_dir / "recognition_index.parquet"
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:  # pragma: no cover - training dependency
                raise RuntimeError("lazy recognition sampling requires pyarrow") from exc
            available = set(pq.ParquetFile(path).schema_arrow.names)
            missing = self._REQUIRED_FIELDS - available
            if missing:
                raise ValueError(f"recognition index is missing fields {sorted(missing)}: {path}")
            # iter_verified_index_rows authenticates the full index before
            # yielding its first row, so an early forged batch can never be
            # consumed by a DataLoader.
            for item in iter_verified_index_rows(shard_dir, "recognition"):
                schedule_id = str(item["schedule_id"])
                split = str(item["split"])
                split_record = self.master_split.require(schedule_id, split)
                if split_record.split != target_split:
                    continue
                self.master_split.authorize(schedule_id, self.purpose, split)
                cv_fold = int(item["cv_fold"])
                if cv_fold != split_record.cv_fold:
                    raise ValueError(f"recognition index CV fold mismatch: {schedule_id}")
                master_sha = str(item["master_split_sha256"])
                if master_sha != str(self.master_split.metadata.get("split_sha256")):
                    raise ValueError(f"recognition index master split hash mismatch: {schedule_id}")
                if self.include_cv_fold is not None and cv_fold != self.include_cv_fold:
                    continue
                if self.exclude_cv_fold is not None and cv_fold == self.exclude_cv_fold:
                    continue
                domain = str(item.get("source_domain") or "synthetic")
                if self.source_domain is not None and domain != self.source_domain:
                    continue
                row_group_object_index = int(item["row_group_object_index"])
                image_object_index = int(item["image_object_index"])
                yield ParquetRecognitionEntry(
                    shard_dir_index=shard_dir_index,
                    image_path=str(item["image_path"]),
                    schedule_id=schedule_id,
                    split=split,
                    cv_fold=cv_fold,
                    source_domain=domain,
                    row_group_object_index=row_group_object_index,
                    image_object_index=image_object_index,
                    object_type=str(item["object_type"]),
                    canonical_code=(
                        None if item.get("canonical_code") in (None, "")
                        else str(item["canonical_code"])
                    ),
                    width_bucket=int(item["width_bucket"]),
                    master_split_sha256=master_sha,
                )

    def _schedule_batches(
        self, schedule_id: str, entries: list[ParquetRecognitionEntry], rng: random.Random,
    ) -> list[list[ParquetRecognitionEntry]]:
        if self.training:
            count = min(len(entries), self.max_per_schedule)
            weights = [3.0 if item.canonical_code in RARE_CODES else 1.0 for item in entries]
            # Preserve the original RareCodeCropSampler contract: weighted
            # draws are with replacement, capped by schedule per epoch.
            selected = rng.choices(entries, weights=weights, k=count)
        else:
            selected = list(entries)
        grouped: dict[tuple[str, int], list[ParquetRecognitionEntry]] = defaultdict(list)
        for item in selected:
            if item.width_bucket not in {160, 320, 640}:
                raise ValueError(
                    f"invalid recognition width bucket {item.width_bucket}: {schedule_id}"
                )
            grouped[(item.image_path, item.width_bucket)].append(item)
        batches: list[list[ParquetRecognitionEntry]] = []
        for (_image_path, width), values in sorted(grouped.items()):
            if self.shuffle:
                rng.shuffle(values)
            batch_size = self.batch_sizes.get(width, 1)
            batches.extend(
                values[offset:offset + batch_size]
                for offset in range(0, len(values), batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        current_schedule_key: tuple[int, str] | None = None
        schedule_entries: list[ParquetRecognitionEntry] = []
        completed_schedules: set[tuple[int, str]] = set()
        batch_buffer: list[list[ParquetRecognitionEntry]] = []

        def flush_schedule():
            nonlocal schedule_entries, batch_buffer
            if current_schedule_key is not None and schedule_entries:
                batch_buffer.extend(self._schedule_batches(
                    current_schedule_key[1], schedule_entries, rng,
                ))
            schedule_entries = []

        for entry in self._iter_entries():
            schedule_key = (entry.shard_dir_index, entry.schedule_id)
            if current_schedule_key is None:
                current_schedule_key = schedule_key
            elif schedule_key != current_schedule_key:
                completed_schedules.add(current_schedule_key)
                flush_schedule()
                if schedule_key in completed_schedules:
                    raise ValueError(
                        "recognition index must keep each schedule contiguous for streaming sampling"
                    )
                current_schedule_key = schedule_key
            schedule_entries.append(entry)
            if len(batch_buffer) >= self.shuffle_buffer_batches:
                if self.shuffle:
                    rng.shuffle(batch_buffer)
                yield from batch_buffer
                batch_buffer = []
        flush_schedule()
        if self.shuffle:
            rng.shuffle(batch_buffer)
        yield from batch_buffer


class LazyParquetRecognitionCropDataset(_WorkerLocalParquetImages, RecognitionCropDataset):
    """Recognition crops loaded from Parquet only for sampled image batches."""

    def __init__(
        self, shard_dirs: Sequence[Path], master_split: MasterSplit, image_root: Path, *,
        purpose: str, charset: Sequence[str], training: bool,
        table_prediction_records: Mapping[tuple[str, str, int], Mapping[str, Any]] | None = None,
        seed: int = 0, image_cache_size: int = 2,
    ) -> None:
        self.image_root = image_root
        self.training = training
        self.charset = {char: index + 1 for index, char in enumerate(charset)}
        self.table_predictions = table_prediction_records or {}
        self.seed = seed
        self.image_cache_size = max(1, int(image_cache_size))
        self._recognition_image_cache: OrderedDict[
            tuple[int, str], list[tuple[int, dict[str, Any]]]
        ] = OrderedDict()
        self._initialize_epoch_state()
        self._initialize_parquet_images(shard_dirs, master_split, purpose=purpose)

    def __len__(self) -> int:
        total = 0
        for shard_dir in self.shard_dirs:
            manifest = json.loads((shard_dir / "shards.manifest.json").read_text(encoding="utf-8"))
            total += int(manifest.get("recognition_index_record_count", 0))
        return total

    def __getstate__(self):
        state = _WorkerLocalParquetImages.__getstate__(self)
        state["_recognition_image_cache"] = OrderedDict()
        return state

    def __getitem__(self, entry: ParquetRecognitionEntry):
        cache_key = (entry.shard_dir_index, entry.image_path)
        indexed_records = self._recognition_image_cache.get(cache_key)
        if indexed_records is None:
            store = self._parquet_store(entry.shard_dir_index)
            indexed_records = store.load_indexed_image(
                entry.image_path, purpose=self.parquet_purpose,
            )
            self._recognition_image_cache[cache_key] = indexed_records
            if len(self._recognition_image_cache) > self.image_cache_size:
                self._recognition_image_cache.popitem(last=False)
        else:
            self._recognition_image_cache.move_to_end(cache_key)
        if not 0 <= entry.image_object_index < len(indexed_records):
            raise IndexError(f"recognition image object index is out of range: {entry}")
        raw_index, item = indexed_records[entry.image_object_index]
        self._validate_recognition_entry(entry, raw_index, item)
        stable_key = (
            f"{entry.schedule_id}:{entry.image_path}:{entry.row_group_object_index}"
        )
        stable_index = int.from_bytes(hashlib.sha256(stable_key.encode()).digest()[:8], "big")
        return self._build_recognition_item(item, stable_index)

    def _parquet_store(self, shard_dir_index: int) -> ParquetImageStore:
        store = self._parquet_stores.get(shard_dir_index)
        if store is None:
            store = ParquetImageStore(
                self.shard_dirs[shard_dir_index], self.master_split,
            )
            self._parquet_stores[shard_dir_index] = store
        return store

    def _validate_recognition_entry(
        self, entry: ParquetRecognitionEntry, raw_index: int,
        item: Mapping[str, Any],
    ) -> None:
        """Bind every sampling/index field to the authenticated object row."""

        expected_master = str(self.master_split.metadata.get("split_sha256") or "")
        actual_canonical = (
            None if item.get("canonical_code") in (None, "")
            else str(item.get("canonical_code"))
        )
        checks = {
            "row_group_object_index": (int(raw_index), entry.row_group_object_index),
            "schedule_id": (str(item.get("schedule_id")), entry.schedule_id),
            "image_path": (str(item.get("image_path")), entry.image_path),
            "split": (str(item.get("split")), entry.split),
            "cv_fold": (int(item.get("cv_fold")), entry.cv_fold),
            "master_split_sha256": (
                str(item.get("master_split_sha256") or ""), entry.master_split_sha256,
            ),
            "source_domain": (
                str(item.get("source_domain") or "synthetic"), entry.source_domain,
            ),
            "object_type": (str(item.get("object_type")), entry.object_type),
            "canonical_code": (actual_canonical, entry.canonical_code),
            "width_bucket": (recognition_width_bucket(item), entry.width_bucket),
        }
        if entry.master_split_sha256 != expected_master:
            raise ValueError(
                f"recognition index master split hash mismatch: {entry.schedule_id}"
            )
        for field, (actual, expected) in checks.items():
            if actual != expected:
                raise ValueError(
                    f"recognition index {field} mismatch for {entry.schedule_id}: "
                    f"index={expected!r}, row={actual!r}"
                )


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
        self.base_seed = seed
        self.epoch = 0
        buckets: dict[int, list[int]] = defaultdict(list)
        # Bounding-box aspect gives a cheap deterministic bucket prediction.
        for index in (list(indices) if indices is not None else range(len(dataset.records))):
            item = dataset.records[index]
            buckets[recognition_width_bucket(item)].append(index)
        self.buckets = buckets

    def __iter__(self):
        rng = random.Random(self.base_seed + self.epoch)
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

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        import math
        return sum(math.ceil(len(indices) / self.batch_sizes.get(width, 1)) for width, indices in self.buckets.items())
