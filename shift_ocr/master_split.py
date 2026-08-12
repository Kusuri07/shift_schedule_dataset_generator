"""Immutable schedule-level split creation and leakage guards.

Every image, augmentation recipe and crop is joined to this file by
``schedule_id``.  Dataset loaders are deliberately not allowed to invent a
split so that synthetic, registered real photos and their derivatives cannot
cross evaluation boundaries.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPLITS = {"train", "validation", "test", "ood_layout"}
SCHEMA_VERSION = "master_split_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SplitRecord:
    schedule_id: str
    split: str
    cv_fold: int
    template_id: str
    layout_family: str
    seed: int
    capture_target: int
    month: int | None = None
    people_count: int | None = None

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"invalid split {self.split!r} for {self.schedule_id}")
        if self.split == "train" and self.cv_fold not in {0, 1, 2}:
            raise ValueError(f"train record must have cv_fold 0..2: {self.schedule_id}")
        if self.split != "train" and self.cv_fold != -1:
            raise ValueError(f"{self.split} record must have cv_fold=-1: {self.schedule_id}")


def _people_bucket(people_count: int | None) -> str:
    if people_count is None:
        return "unknown"
    if people_count <= 20:
        return "small"
    if people_count <= 27:
        return "medium"
    return "large"


def _assign_stratified(
    groups: Mapping[tuple[str, int | None, str], list[Mapping[str, Any]]], seed: int,
) -> dict[str, str]:
    """Hamilton allocation preserves global 70/15/15 without small-stratum bias."""
    split_names = ("train", "validation", "test")
    ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    total = sum(len(items) for items in groups.values())
    targets = {
        "train": round(total * ratios["train"]),
        "validation": round(total * ratios["validation"]),
    }
    targets["test"] = total - targets["train"] - targets["validation"]
    allocations: dict[tuple[str, int | None, str], dict[str, int]] = {}
    slots: dict[tuple[str, int | None, str], int] = {}
    fractions: dict[tuple[tuple[str, int | None, str], str], float] = {}
    for key, items in groups.items():
        allocations[key] = {}
        used = 0
        for split in split_names:
            exact = len(items) * ratios[split]
            value = int(exact)
            allocations[key][split] = value
            fractions[(key, split)] = exact - value
            used += value
        slots[key] = len(items) - used
    remaining = {
        split: targets[split] - sum(value[split] for value in allocations.values())
        for split in split_names
    }
    candidates = sorted(
        ((fractions[(key, split)], hashlib.sha256(f"{seed}:{key}:{split}".encode()).hexdigest(), key, split)
         for key in groups for split in split_names),
        reverse=True,
    )
    while any(value > 0 for value in remaining.values()):
        progressed = False
        for _fraction, _tie, key, split in candidates:
            if slots[key] <= 0 or remaining[split] <= 0:
                continue
            allocations[key][split] += 1
            slots[key] -= 1
            remaining[split] -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("unable to allocate stratified split targets")

    assignment: dict[str, str] = {}
    for index, key in enumerate(sorted(groups, key=str)):
        ordered = sorted(groups[key], key=lambda item: str(item["schedule_id"]))
        random.Random(seed + index * 7919).shuffle(ordered)
        cursor = 0
        for split in split_names:
            for item in ordered[cursor:cursor + allocations[key][split]]:
                assignment[str(item["schedule_id"])] = split
            cursor += allocations[key][split]
    return assignment


def create_master_split(
    schedules: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    config: Mapping[str, Any],
    ood_schedules: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[SplitRecord], dict[str, Any]]:
    """Create one deterministic split before any derived data is generated."""
    ids = [str(item["schedule_id"]) for item in [*schedules, *ood_schedules]]
    if len(ids) != len(set(ids)):
        raise ValueError("schedule_id values must be globally unique")

    groups: dict[tuple[str, int | None, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in schedules:
        key = (
            str(item.get("template_id", "unknown")),
            int(item["month"]) if item.get("month") is not None else None,
            _people_bucket(int(item["people_count"])) if item.get("people_count") is not None else "unknown",
        )
        groups[key].append(item)

    assignment = _assign_stratified(groups, seed)

    records: list[SplitRecord] = []
    for item in schedules:
        schedule_id = str(item["schedule_id"])
        split = assignment[schedule_id]
        fold_hash = int(hashlib.sha256(f"{seed}:{schedule_id}".encode()).hexdigest()[:8], 16)
        record = SplitRecord(
            schedule_id=schedule_id,
            split=split,
            cv_fold=fold_hash % 3 if split == "train" else -1,
            template_id=str(item.get("template_id", "unknown")),
            layout_family=str(item.get("layout_family", item.get("template_id", "unknown"))),
            seed=int(item.get("seed", seed)),
            capture_target=int(item.get("capture_target", 0)),
            month=int(item["month"]) if item.get("month") is not None else None,
            people_count=int(item["people_count"]) if item.get("people_count") is not None else None,
        )
        record.validate()
        records.append(record)

    for item in ood_schedules:
        record = SplitRecord(
            schedule_id=str(item["schedule_id"]),
            split="ood_layout",
            cv_fold=-1,
            template_id=str(item.get("template_id", "ood")),
            layout_family=str(item.get("layout_family", item.get("template_id", "ood"))),
            seed=int(item.get("seed", seed)),
            capture_target=int(item.get("capture_target", 0)),
            month=int(item["month"]) if item.get("month") is not None else None,
            people_count=int(item["people_count"]) if item.get("people_count") is not None else None,
        )
        record.validate()
        records.append(record)

    records.sort(key=lambda item: item.schedule_id)
    serializable = [asdict(item) for item in records]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "config_sha256": sha256_value(config),
        "split_sha256": sha256_value(serializable),
        "record_count": len(records),
        "counts": {name: sum(item.split == name for item in records) for name in sorted(SPLITS)},
    }
    return records, metadata


def write_master_split(records: Sequence[SplitRecord], metadata: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in records]
    expected = sha256_value(rows)
    if metadata.get("split_sha256") != expected:
        raise ValueError("split metadata hash does not match records")
    split_path = output_dir / "master_split.jsonl"
    split_path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    manifest = dict(metadata)
    manifest["master_split_file_sha256"] = sha256_file(split_path)
    (output_dir / "master_split.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class MasterSplit:
    def __init__(self, records: Iterable[SplitRecord], metadata: Mapping[str, Any]) -> None:
        self.records = {item.schedule_id: item for item in records}
        self.metadata = dict(metadata)
        if len(self.records) != int(self.metadata.get("record_count", -1)):
            raise ValueError("master split record count mismatch")
        for item in self.records.values():
            item.validate()

    @classmethod
    def load(cls, split_path: Path, manifest_path: Path | None = None) -> "MasterSplit":
        manifest_path = manifest_path or split_path.with_name("master_split.manifest.json")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256_file(split_path) != metadata.get("master_split_file_sha256"):
            raise ValueError("master split file SHA-256 mismatch")
        rows = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line]
        if sha256_value(rows) != metadata.get("split_sha256"):
            raise ValueError("master split content SHA-256 mismatch")
        return cls((SplitRecord(**row) for row in rows), metadata)

    def require(self, schedule_id: str, declared_split: str | None = None) -> SplitRecord:
        try:
            record = self.records[schedule_id]
        except KeyError as exc:
            raise ValueError(f"unknown schedule_id: {schedule_id}") from exc
        if declared_split is not None and record.split != declared_split:
            raise ValueError(
                f"split mismatch for {schedule_id}: master={record.split}, data={declared_split}"
            )
        return record

    def authorize(self, schedule_id: str, purpose: str, declared_split: str | None = None) -> SplitRecord:
        record = self.require(schedule_id, declared_split)
        if purpose in {"train", "cv"} and record.split != "train":
            raise ValueError(f"{record.split} schedule cannot enter {purpose}: {schedule_id}")
        if purpose in {"select", "threshold", "quantize", "route"} and record.split != "validation":
            raise ValueError(f"only validation is allowed for {purpose}: {schedule_id}")
        if purpose == "cv" and record.cv_fold not in {0, 1, 2}:
            raise ValueError(f"invalid CV fold for {schedule_id}")
        return record
