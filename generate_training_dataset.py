#!/usr/bin/env python3
"""Resumable production generator for synthetic OCR schedules.

The 200-schedule pilot and 10,000-schedule production run use this exact code
path, renderer, annotation schema, master split and Parquet builder.  Completed
Excel/PNG chunks are content-addressed in the cache so a later invocation
resumes at the first incomplete chunk instead of restarting the dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Mapping

import generate_dataset as generator
from shift_ocr.charset import coverage_report, load_charset
from shift_ocr.geometry import is_convex_quad
from shift_ocr.master_split import MasterSplit, create_master_split, sha256_file, write_master_split
from shift_ocr.paths import DEFAULT_PATH_CONFIG, StorageLayout, load_storage_layout
from shift_ocr.shards import build_parquet_shards, iter_jsonl, verify_compatible_annotations


ROW_FIELDS = [
    "schedule_id", "split", "cv_fold", "master_split_sha256", "template_id", "layout_family",
    "page_number", "page_count", "sheet_name", "row_id", "row_index", "excel_row", "group",
    "name", "surname", "surname_rank", "surname_population", "surname_hanja_variants",
    "surname_source_method", "surname_source_url", "given_name", "birth_year", "gender",
    "day_count", "codes_canonical", "codes_display", "name_cell", "image_path",
]
CELL_FIELDS = [
    "schedule_id", "split", "cv_fold", "master_split_sha256", "template_id", "layout_family",
    "row_id", "row_index", "name", "surname", "surname_rank", "surname_population", "birth_year",
    "gender", "group", "day", "date", "canonical_code", "display_code", "display_text",
    "object_type", "excel_cell", "bbox_px", "cell_polygon", "text_polygon", "text_polygon_source",
    "text_polygon_validation_max_error_px", "visibility", "ignore", "name_bbox_px",
    "name_cell_polygon", "image_path", "image_width", "image_height",
]
OBJECT_FIELDS = [
    "schedule_id", "split", "cv_fold", "master_split_sha256", "template_id", "layout_family",
    "object_type", "display_text", "canonical_code", "row_id", "row_index", "day", "bbox_px",
    "cell_polygon", "text_polygon", "text_polygon_source", "visibility", "ignore", "image_path",
    "text_polygon_validation_max_error_px", "image_width", "image_height",
]
JSON_FIELDS = {
    "codes_canonical", "codes_display", "bbox_px", "cell_polygon", "text_polygon",
    "name_bbox_px", "name_cell_polygon",
}


class StreamingAnnotations:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.files: dict[str, Any] = {}
        self.csv_writers: dict[str, csv.DictWriter] = {}
        self.counts = {"rows": 0, "cells": 0, "objects": 0}
        for name, fields in (("rows", ROW_FIELDS), ("cells", CELL_FIELDS), ("objects", OBJECT_FIELDS)):
            jsonl = self.stack.enter_context((directory / f"{name}.jsonl").open("w", encoding="utf-8"))
            csv_file = self.stack.enter_context(
                (directory / f"{name}.csv").open("w", newline="", encoding="utf-8-sig")
            )
            writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            self.files[name] = jsonl
            self.csv_writers[name] = writer

    def write(self, name: str, item: Mapping[str, Any]) -> None:
        record = {field: item.get(field) for field in self.csv_writers[name].fieldnames or []}
        self.files[name].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        csv_record = dict(record)
        for field in JSON_FIELDS & csv_record.keys():
            csv_record[field] = json.dumps(csv_record[field], ensure_ascii=False, separators=(",", ":"))
        self.csv_writers[name].writerow(csv_record)
        self.counts[name] += 1

    def close(self) -> None:
        self.stack.close()


class PeakRssMonitor:
    """Sample total Python process-tree RSS without retaining dataset objects."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        import psutil

        self.psutil = psutil
        self.process = psutil.Process(os.getpid())
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            total = 0
            try:
                total += self.process.memory_info().rss
                for child in self.process.children(recursive=True):
                    # Node is reported separately; exclude it from Python RAM.
                    if "python" in child.name().lower():
                        total += child.memory_info().rss
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                pass
            self.peak_bytes = max(self.peak_bytes, total)
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "PeakRssMonitor":
        self._thread.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / 1_000_000


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def schedule_seed(root_seed: int, schedule_id: str) -> int:
    return int(hashlib.sha256(f"{root_seed}:{schedule_id}".encode()).hexdigest()[:15], 16)


def plan_family(
    count: int, root_seed: int, templates: list[str], prefix: str, *, ood: bool,
) -> list[dict[str, Any]]:
    plans = []
    for index in range(1, count + 1):
        schedule_id = f"{prefix}_{index:04d}"
        seed = schedule_seed(root_seed, schedule_id)
        rng = random.Random(seed)
        template = templates[(index - 1) % len(templates)]
        plans.append({
            "schedule_id": schedule_id,
            "template_id": template,
            "layout_family": template,
            "month": rng.randint(1, 12),
            "people_count": rng.randint(18, 40 if ood else 32),
            "seed": seed,
            "capture_target": 0,
            "schedule_index": index,
        })
    return plans


def assign_real_capture_targets(
    plans: list[dict[str, Any]], schedule_count: int, image_target: int, seed: int,
) -> None:
    if schedule_count > len(plans) or image_target < schedule_count:
        raise ValueError("capture targets require image_target >= schedule_count <= planned schedules")
    selected = sorted(
        plans,
        key=lambda item: hashlib.sha256(f"capture:{seed}:{item['schedule_id']}".encode()).digest(),
    )[:schedule_count]
    if not selected:
        return
    base, remainder = divmod(image_target, schedule_count)
    for index, item in enumerate(selected):
        item["capture_target"] = base + (1 if index < remainder else 0)


def make_schedule(plan: Mapping[str, Any], config: generator.GeneratorConfig, names, surnames):
    return generator.generate_schedule(
        int(plan["schedule_index"]), config, random.Random(int(plan["seed"])), names, surnames,
        forced_template=str(plan["template_id"]),
    )


def write_schedule_annotations(schedule, split: MasterSplit, writers: StreamingAnnotations) -> None:
    split_record = split.require(schedule.schedule_id)
    common = {
        "split": split_record.split,
        "cv_fold": split_record.cv_fold,
        "master_split_sha256": split.metadata["split_sha256"],
        "layout_family": split_record.layout_family,
        "image_path": schedule.clean_image_path,
        "image_width": schedule.image_width,
        "image_height": schedule.image_height,
    }
    rows = {row.row_id: row for row in schedule.rows}
    for row_index, row in enumerate(schedule.rows, start=1):
        writers.write("rows", {
            **common, "schedule_id": schedule.schedule_id, "template_id": schedule.template_id,
            "page_number": schedule.page_number, "page_count": schedule.page_count,
            "sheet_name": schedule.sheet_name, "row_id": row.row_id, "row_index": row_index,
            "excel_row": row.excel_row, "group": row.group, "name": row.name, "surname": row.surname,
            "surname_rank": row.surname_rank, "surname_population": row.surname_population,
            "surname_hanja_variants": row.surname_hanja_variants,
            "surname_source_method": row.surname_source_method,
            "surname_source_url": row.surname_source_url,
            "given_name": row.given_name, "birth_year": row.birth_year, "gender": row.gender,
            "day_count": schedule.day_count, "codes_canonical": row.codes_canonical,
            "codes_display": row.codes_display, "name_cell": row.name_cell,
        })
    for annotation in schedule.cell_annotations:
        row = rows[annotation["row_id"]]
        excel_cell = (
            f"{generator.excel_column_name(generator.schedule_day_start_col_1based(schedule) + int(annotation['day']) - 1)}"
            f"{row.excel_row}"
        )
        writers.write("cells", {**annotation, **common, "excel_cell": excel_cell})
    for annotation in schedule.training_objects:
        writers.write("objects", {**annotation, **common})


def ensure_master_split(
    base_plans: list[dict[str, Any]], ood_plans: list[dict[str, Any]], *, seed: int,
    config: Mapping[str, Any], split_dir: Path, cache_dir: Path,
) -> MasterSplit:
    records, metadata = create_master_split(
        base_plans, seed=seed, config=config, ood_schedules=ood_plans,
    )
    split_path = split_dir / "master_split.jsonl"
    manifest_path = split_dir / "master_split.manifest.json"
    existing = (split_path.exists(), manifest_path.exists())
    if any(existing):
        if not all(existing):
            raise RuntimeError("master split is incomplete; it will not be overwritten automatically")
        loaded = MasterSplit.load(split_path, manifest_path)
        if loaded.metadata.get("split_sha256") != metadata["split_sha256"]:
            raise ValueError("existing immutable master split does not match this dataset plan")
        return loaded

    # The schedule plan is now final.  Write the split once in a temporary
    # cache directory, validate it, then publish both immutable files.
    attempt = cache_dir / "master_split_publish"
    attempt.mkdir(parents=True, exist_ok=True)
    write_master_split(records, metadata, attempt)
    MasterSplit.load(attempt / "master_split.jsonl")
    split_dir.mkdir(parents=True, exist_ok=True)
    if split_path.exists() or manifest_path.exists():
        raise FileExistsError("master split appeared concurrently and will not be overwritten")
    os.replace(attempt / "master_split.jsonl", split_path)
    os.replace(attempt / "master_split.manifest.json", manifest_path)
    return MasterSplit.load(split_path, manifest_path)


def _chunk_name(family: str, start: int, end: int) -> str:
    return f"{family}_{start:05d}-{end:05d}"


def _verified_completed_chunk(marker: Path) -> dict[str, Any] | None:
    if not marker.exists():
        return None
    state = json.loads(marker.read_text(encoding="utf-8"))
    annotation_dir = Path(state["annotation_dir"])
    for name in ("rows", "cells", "objects"):
        for suffix in ("jsonl", "csv"):
            if not (annotation_dir / f"{name}.{suffix}").exists():
                return None
    for item in state.get("output_files", []):
        path = Path(item["path"])
        if not path.exists() or path.stat().st_size != int(item["size_bytes"]):
            return None
        if sha256_file(path) != item["sha256"]:
            return None
    return state


def render_chunk(
    *, family: str, plans: list[dict[str, Any]], family_start: int,
    family_count: int, config: generator.GeneratorConfig, names, surname_entries,
    surname_pool, dataset_dir: Path, cache_dir: Path, log_dir: Path,
    master: MasterSplit, retries: int,
) -> dict[str, Any]:
    first, last = family_start + 1, family_start + len(plans)
    chunk_root = cache_dir / "chunks" / _chunk_name(family, first, last)
    marker = chunk_root / "chunk.complete.json"
    completed = _verified_completed_chunk(marker)
    if completed is not None:
        completed["resumed_from_cache"] = True
        return completed

    chunk_root.mkdir(parents=True, exist_ok=True)
    failure_log = log_dir / "chunk_failures.jsonl"
    for retry in range(retries + 1):
        attempt_dir = chunk_root / f"attempt_{int(time.time() * 1000)}_{retry + 1}"
        annotation_dir = attempt_dir / "annotations"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        try:
            schedules = [make_schedule(plan, config, names, surname_pool) for plan in plans]
            for offset, (schedule, plan) in enumerate(zip(schedules, plans)):
                if schedule.schedule_id != plan["schedule_id"]:
                    raise ValueError("schedule regeneration diverged from immutable plan")
                schedule.page_number = family_start + offset + 1
                schedule.page_count = family_count
                schedule.show_page_number = config.show_page_numbers
            if family == "schedule" and family_start == 0 and config.ensure_all_codes:
                generator.ensure_code_coverage(
                    schedules, random.Random(config.seed), config.case_mutation_probability,
                )
            workbook_relative = (
                f"workbooks/synthetic_shift_dataset_{family}_{first:05d}-{last:05d}.xlsx"
            )
            (dataset_dir / "workbooks").mkdir(parents=True, exist_ok=True)
            renderer_metrics = generator.render_dataset_workbook(
                schedules, names, surname_entries, surname_pool, dataset_dir,
                export_workbook=True, workbook_name=workbook_relative,
                workbook_profile="training_chunk",
            )
            writers = StreamingAnnotations(annotation_dir)
            manifest_rows = []
            try:
                for schedule in schedules:
                    write_schedule_annotations(schedule, master, writers)
                    split_record = master.require(schedule.schedule_id)
                    manifest_rows.append({
                        "schedule_id": schedule.schedule_id,
                        "template_id": schedule.template_id,
                        "layout_family": split_record.layout_family,
                        "split": split_record.split,
                        "cv_fold": split_record.cv_fold,
                        "year": schedule.year,
                        "month": schedule.month,
                        "day_count": schedule.day_count,
                        "people_count": len(schedule.rows),
                        "image_path": schedule.clean_image_path,
                        "image_size": [schedule.image_width, schedule.image_height],
                        "workbook_path": workbook_relative,
                    })
            finally:
                writers.close()
            output_paths = [dataset_dir / workbook_relative] + [
                dataset_dir / row["image_path"] for row in manifest_rows
            ]
            output_files = [{
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            } for path in output_paths]
            state = {
                "schema_version": "production_chunk_v1",
                "family": family,
                "first": first,
                "last": last,
                "schedule_count": len(schedules),
                "annotation_dir": str(annotation_dir.resolve()),
                "annotation_counts": writers.counts,
                "manifest_rows": manifest_rows,
                "renderer": renderer_metrics,
                "chunk_duration_seconds": time.perf_counter() - started,
                "retry_number": retry,
                "output_files": output_files,
                "resumed_from_cache": False,
            }
            atomic_json(marker, state)
            return state
        except Exception as exc:
            failure_log.parent.mkdir(parents=True, exist_ok=True)
            with failure_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "chunk": chunk_root.name,
                    "attempt": retry + 1,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": time.time(),
                }, ensure_ascii=False) + "\n")
            if retry >= retries:
                raise
    raise AssertionError("unreachable")


def merge_chunk_annotations(states: list[dict[str, Any]], output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: sum(int(state["annotation_counts"][name]) for state in states)
              for name in ("rows", "cells", "objects")}
    for name in ("rows", "cells", "objects"):
        json_partial = output_dir / f"{name}.jsonl.partial"
        with json_partial.open("wb") as destination:
            for state in states:
                with (Path(state["annotation_dir"]) / f"{name}.jsonl").open("rb") as source:
                    shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
        os.replace(json_partial, output_dir / f"{name}.jsonl")

        csv_partial = output_dir / f"{name}.csv.partial"
        with csv_partial.open("wb") as destination:
            for index, state in enumerate(states):
                with (Path(state["annotation_dir"]) / f"{name}.csv").open("rb") as source:
                    if index:
                        source.readline()  # skip BOM + duplicate header
                    shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
        os.replace(csv_partial, output_dir / f"{name}.csv")
    return counts


def build_or_verify_shards(
    objects_jsonl: Path, objects_csv: Path, master: MasterSplit,
    shard_dir: Path, cache_dir: Path,
) -> tuple[dict[str, Any], float, bool]:
    if shard_dir.exists() and any(shard_dir.iterdir()):
        verification = verify_compatible_annotations(objects_jsonl, objects_csv, shard_dir)
        manifest = json.loads((shard_dir / "shards.manifest.json").read_text(encoding="utf-8"))
        return {**manifest, "verification": verification}, 0.0, True
    if shard_dir.exists():
        shard_dir.rmdir()  # only an explicitly resolved empty target directory
    attempt = cache_dir / f"shard_build_{int(time.time() * 1000)}"
    started = time.perf_counter()
    result = build_parquet_shards(objects_jsonl, master, attempt)
    duration = time.perf_counter() - started
    verification = verify_compatible_annotations(objects_jsonl, objects_csv, attempt)
    shard_dir.parent.mkdir(parents=True, exist_ok=True)
    if shard_dir.exists():
        raise FileExistsError(f"shard target appeared concurrently: {shard_dir}")
    os.replace(attempt, shard_dir)
    return {**result, "verification": verification}, duration, False


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _valid_quad(value: Any, width: int, height: int) -> tuple[bool, str | None]:
    try:
        if not is_convex_quad(value):
            return False, "not_convex_quad"
        coordinates = [float(number) for point in value for number in point]
        if not all(math.isfinite(number) for number in coordinates):
            return False, "non_finite"
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        if min(xs) < -1 or min(ys) < -1 or max(xs) > width + 1 or max(ys) > height + 1:
            return False, "outside_image"
        return True, None
    except (TypeError, ValueError, IndexError):
        return False, "malformed"


def validate_production_dataset(
    *, manifest_rows: list[dict[str, Any]], expected_base_count: int,
    expected_ood_count: int, dataset_dir: Path, shard_dir: Path,
    master: MasterSplit, annotation_counts: Mapping[str, int],
) -> dict[str, Any]:
    from PIL import Image

    ids = [str(item["schedule_id"]) for item in manifest_rows]
    base_ids = [value for value in ids if value.startswith("schedule_")]
    ood_ids = [value for value in ids if value.startswith("ood_schedule_")]
    duplicate_ids = sorted(value for value, count in Counter(ids).items() if count > 1)
    missing_png: list[str] = []
    corrupt_png: list[dict[str, str]] = []
    png_sizes: list[int] = []
    for item in manifest_rows:
        image_path = dataset_dir / str(item["image_path"])
        if not image_path.exists():
            missing_png.append(str(item["schedule_id"]))
            continue
        png_sizes.append(image_path.stat().st_size)
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            corrupt_png.append({"schedule_id": str(item["schedule_id"]), "error": str(exc)})

    annotation_dir = dataset_dir / "annotations"
    schedule_sets: dict[str, set[str]] = {}
    invalid_polygon_count = 0
    invalid_polygon_details: list[dict[str, Any]] = []
    code_distribution: Counter[str] = Counter()
    for name in ("rows", "cells", "objects"):
        seen: set[str] = set()
        for record in iter_jsonl(annotation_dir / f"{name}.jsonl"):
            schedule_id = str(record.get("schedule_id") or "")
            seen.add(schedule_id)
            if name == "objects":
                if record.get("object_type") == "shift_code":
                    code_distribution[str(record.get("canonical_code") or "")] += 1
                width = int(record.get("image_width") or 0)
                height = int(record.get("image_height") or 0)
                for field in ("cell_polygon", "text_polygon"):
                    valid, reason = _valid_quad(record.get(field), width, height)
                    if not valid:
                        invalid_polygon_count += 1
                        if len(invalid_polygon_details) < 100:
                            invalid_polygon_details.append({
                                "schedule_id": schedule_id,
                                "object_type": record.get("object_type"),
                                "field": field,
                                "reason": reason,
                            })
        schedule_sets[name] = seen
    missing_annotation_schedules = {
        name: sorted(set(ids) - values) for name, values in schedule_sets.items()
    }

    charset = load_charset(Path(__file__).resolve().parent / "data" / "korean_charset_v1.txt")
    coverage = coverage_report(iter_jsonl(annotation_dir / "objects.jsonl"), charset)
    atomic_json(annotation_dir / "charset_coverage.json", coverage)
    checksum_mismatches = 0
    checksum_error = None
    try:
        compatibility = verify_compatible_annotations(
            annotation_dir / "objects.jsonl", annotation_dir / "objects.csv", shard_dir,
        )
    except Exception as exc:
        checksum_mismatches = 1
        checksum_error = str(exc)
        compatibility = None

    split_errors: list[str] = []
    split_distribution: Counter[str] = Counter()
    for item in manifest_rows:
        try:
            record = master.require(str(item["schedule_id"]), str(item["split"]))
            split_distribution[record.split] += 1
            if int(item["cv_fold"]) != record.cv_fold:
                split_errors.append(f"cv_fold mismatch: {item['schedule_id']}")
        except Exception as exc:
            split_errors.append(str(exc))
    workbook_paths = sorted({dataset_dir / str(item["workbook_path"]) for item in manifest_rows})
    workbook_sizes = [path.stat().st_size for path in workbook_paths if path.exists()]
    distributions = {
        "template": dict(sorted(Counter(str(item["template_id"]) for item in manifest_rows).items())),
        "month": dict(sorted(Counter(str(item["month"]) for item in manifest_rows).items())),
        "people_count": dict(sorted(Counter(str(item["people_count"]) for item in manifest_rows).items())),
        "canonical_code": dict(sorted(code_distribution.items())),
        "split": dict(sorted(split_distribution.items())),
    }
    fatal_conditions = {
        "base_schedule_count_mismatch": len(base_ids) != expected_base_count,
        "ood_schedule_count_mismatch": len(ood_ids) != expected_ood_count,
        "duplicate_schedule_ids": bool(duplicate_ids),
        "missing_png": bool(missing_png),
        "corrupt_png": bool(corrupt_png),
        "annotation_missing": any(missing_annotation_schedules.values()),
        "invalid_polygons": invalid_polygon_count != 0,
        "dictionary_oov": int(coverage["oov_count"]) != 0,
        "checksum_mismatch": checksum_mismatches != 0,
        "split_leakage_or_metadata_error": bool(split_errors),
    }
    return {
        "passed": not any(fatal_conditions.values()),
        "fatal_conditions": fatal_conditions,
        "expected_base_schedule_count": expected_base_count,
        "expected_ood_schedule_count": expected_ood_count,
        "successful_schedule_count": len(ids),
        "unique_schedule_id_count": len(set(ids)),
        "base_schedule_id_count": len(set(base_ids)),
        "ood_schedule_id_count": len(set(ood_ids)),
        "duplicate_schedule_ids": duplicate_ids[:100],
        "missing_png_count": len(missing_png),
        "corrupt_png_count": len(corrupt_png),
        "png_average_mb": statistics.fmean(png_sizes) / 1_000_000 if png_sizes else 0.0,
        "png_max_mb": max(png_sizes, default=0) / 1_000_000,
        "excel_workbook_count": len(workbook_sizes),
        "excel_average_mb": statistics.fmean(workbook_sizes) / 1_000_000 if workbook_sizes else 0.0,
        "annotation_counts": dict(annotation_counts),
        "missing_annotation_schedules": {
            name: values[:100] for name, values in missing_annotation_schedules.items()
        },
        "invalid_polygon_count": invalid_polygon_count,
        "invalid_polygon_details": invalid_polygon_details,
        "dictionary_coverage": coverage,
        "checksum_mismatch_count": checksum_mismatches,
        "checksum_error": checksum_error,
        "compatibility": compatibility,
        "split_error_count": len(split_errors),
        "split_errors": split_errors[:100],
        "distributions": distributions,
    }


def resolve_generation_paths(
    args: argparse.Namespace, layout: StorageLayout,
) -> tuple[Path, Path, Path, Path]:
    legacy_output = getattr(args, "output_dir", None)
    if args.dataset_dir and legacy_output:
        raise ValueError("use only one of --dataset-dir and legacy --output-dir")
    dataset_dir = (args.dataset_dir or legacy_output or layout.dataset(args.dataset_name)).resolve()
    shard_dir = (args.shard_dir or layout.shard_set(args.dataset_name)).resolve()
    cache_dir = (args.cache_dir or layout.dataset_cache(args.dataset_name)).resolve()
    log_dir = (args.log_dir or layout.dataset_logs(args.dataset_name)).resolve()
    return dataset_dir, shard_dir, cache_dir, log_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--ood-count", type=int, default=200)
    parser.add_argument("--dataset-name", default="synthetic_10000")
    parser.add_argument("--path-config", type=Path, default=DEFAULT_PATH_CONFIG)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Legacy alias for --dataset-dir")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--render-chunk-size", type=int, default=25,
        help="Artifact-tool schedules per workbook/process; 25 is the production-tested default",
    )
    parser.add_argument(
        "--chunk-retries", type=int, default=2,
        help="Retries for only the failed chunk after the initial attempt",
    )
    parser.add_argument("--real-schedule-target", type=int)
    parser.add_argument("--real-photo-target", type=int)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.count < 1 or args.ood_count < 0:
        raise ValueError("count must be positive and ood-count nonnegative")
    if args.render_chunk_size < 1 or args.chunk_retries < 0:
        raise ValueError("chunk size must be positive and retries nonnegative")
    layout = load_storage_layout(args.path_config)
    dataset_dir, shard_dir, cache_dir, log_dir = resolve_generation_paths(args, layout)
    for directory in (dataset_dir, cache_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "production_generation_report.json"
    completion_path = dataset_dir / "dataset.complete.json"
    if completion_path.exists():
        prior = json.loads(completion_path.read_text(encoding="utf-8"))
        if prior.get("plan", {}).get("count") != args.count or prior.get("plan", {}).get("ood_count") != args.ood_count:
            raise ValueError("completed dataset path belongs to a different plan")
        return prior

    started = time.perf_counter()
    with PeakRssMonitor() as memory_monitor:
        base_plans = plan_family(args.count, args.seed, generator.TEMPLATE_IDS, "schedule", ood=False)
        ood_plans = plan_family(
            args.ood_count, args.seed + 1, generator.OOD_TEMPLATE_IDS, "ood_schedule", ood=True,
        )
        real_schedule_target = (
            args.real_schedule_target if args.real_schedule_target is not None else min(300, args.count)
        )
        real_photo_target = args.real_photo_target if args.real_photo_target is not None else (
            1000 if real_schedule_target >= 300 else max(
                real_schedule_target, round(real_schedule_target * 10 / 3),
            )
        )
        assign_real_capture_targets(
            base_plans, real_schedule_target, real_photo_target, args.seed,
        )
        for item in sorted(
            ood_plans, key=lambda value: str(value["schedule_id"]),
        )[:min(100, len(ood_plans))]:
            item["capture_target"] = 1
        plan_config = {
            "schema_version": "production_dataset_plan_v1",
            "count": args.count,
            "ood_count": args.ood_count,
            "seed": args.seed,
            "render_chunk_size": args.render_chunk_size,
            "real_schedule_target": real_schedule_target,
            "real_photo_target": real_photo_target,
            "templates": generator.TEMPLATE_IDS,
            "ood_templates": generator.OOD_TEMPLATE_IDS,
        }
        atomic_json(cache_dir / "dataset_plan.json", plan_config)
        master = ensure_master_split(
            base_plans, ood_plans, seed=args.seed, config=plan_config,
            split_dir=dataset_dir / "splits", cache_dir=cache_dir,
        )

        names = generator.build_name_dictionary(generator.MIN_BIRTH_YEAR, generator.MAX_BIRTH_YEAR)
        surname_entries = generator.build_surname_dictionary(scrape=False)
        surname_pool = generator.aggregate_surnames(surname_entries)
        generator.export_dictionaries(names, surname_entries, dataset_dir)
        base_config = generator.GeneratorConfig(
            count=args.count, seed=args.seed, output_dir=str(dataset_dir),
            template_ids=generator.TEMPLATE_IDS.copy(), schedule_id_prefix="schedule",
            scrape_surnames=False,
        )
        ood_config = generator.GeneratorConfig(
            count=args.ood_count, seed=args.seed + 1, output_dir=str(dataset_dir),
            template_ids=generator.OOD_TEMPLATE_IDS.copy(), schedule_id_prefix="ood_schedule",
            scrape_surnames=False, min_people=18, max_people=40,
        )
        chunk_states: list[dict[str, Any]] = []
        failures = 0
        for family, plans, config in (
            ("schedule", base_plans, base_config),
            ("ood_schedule", ood_plans, ood_config),
        ):
            for start in range(0, len(plans), args.render_chunk_size):
                selected = plans[start:start + args.render_chunk_size]
                try:
                    state = render_chunk(
                        family=family, plans=selected, family_start=start,
                        family_count=len(plans), config=config, names=names,
                        surname_entries=surname_entries, surname_pool=surname_pool,
                        dataset_dir=dataset_dir, cache_dir=cache_dir, log_dir=log_dir,
                        master=master, retries=args.chunk_retries,
                    )
                    chunk_states.append(state)
                    print(json.dumps({
                        "progress": {
                            "completed_schedules": sum(item["schedule_count"] for item in chunk_states),
                            "total_schedules": args.count + args.ood_count,
                            "chunk": _chunk_name(family, start + 1, start + len(selected)),
                            "resumed": state.get("resumed_from_cache", False),
                            "node_peak_rss_mb": state["renderer"].get("node_peak_rss_mb"),
                        }
                    }, ensure_ascii=False), flush=True)
                except Exception:
                    failures += len(selected)
                    raise

        annotation_dir = dataset_dir / "annotations"
        annotation_counts = merge_chunk_annotations(chunk_states, annotation_dir)
        manifest_rows = [row for state in chunk_states for row in state["manifest_rows"]]
        manifest = {
            "dataset_version": "2.1-production-resumable",
            "master_split_sha256": master.metadata["split_sha256"],
            "schedule_count": args.count,
            "ood_layout_schedule_count": args.ood_count,
            "annotation_counts": annotation_counts,
            "schedules": manifest_rows,
        }
        atomic_json(annotation_dir / "manifest.json", manifest)
        shard_result, shard_seconds, shard_resumed = build_or_verify_shards(
            annotation_dir / "objects.jsonl", annotation_dir / "objects.csv",
            master, shard_dir, cache_dir,
        )
        qa = validate_production_dataset(
            manifest_rows=manifest_rows, expected_base_count=args.count,
            expected_ood_count=args.ood_count, dataset_dir=dataset_dir,
            shard_dir=shard_dir, master=master, annotation_counts=annotation_counts,
        )
        total_seconds = time.perf_counter() - started
        generation_seconds = sum(float(state["chunk_duration_seconds"]) for state in chunk_states)
        successful = int(qa["successful_schedule_count"])
        dataset_bytes = directory_size(dataset_dir)
        shard_bytes = directory_size(shard_dir)
        annotation_bytes = directory_size(annotation_dir)
        workbook_bytes = directory_size(dataset_dir / "workbooks")
        cache_bytes = directory_size(cache_dir)
        log_bytes = directory_size(log_dir)
        node_peak_mb = max(
            (float(state["renderer"].get("node_peak_rss_mb") or 0) for state in chunk_states),
            default=0.0,
        )
        scale = 10_000 / max(1, args.count)
        report = {
            "schema_version": "production_generation_report_v1",
            "passed": bool(qa["passed"]) and failures == 0,
            "paths": {
                "path_config": str(args.path_config.resolve()),
                "dataset": str(dataset_dir),
                "shards": str(shard_dir),
                "cache": str(cache_dir),
                "logs": str(log_dir),
            },
            "plan": plan_config,
            "master_split_sha256": master.metadata["split_sha256"],
            "successful_schedule_count": successful,
            "failed_schedule_count": failures,
            "generation_seconds": generation_seconds,
            "total_pipeline_seconds": total_seconds,
            "average_generation_seconds_per_schedule": generation_seconds / max(1, successful),
            "generation_throughput_schedules_per_minute": successful * 60 / max(generation_seconds, 1e-9),
            "node_artifact_tool_peak_memory_mb": node_peak_mb,
            "python_peak_ram_mb": memory_monitor.peak_mb,
            "shard_generation_seconds": shard_seconds,
            "shards_resumed": shard_resumed,
            "sizes": {
                "dataset_total_mb": dataset_bytes / 1_000_000,
                "annotation_total_mb": annotation_bytes / 1_000_000,
                "parquet_shards_total_mb": shard_bytes / 1_000_000,
                "excel_total_mb": workbook_bytes / 1_000_000,
                "cache_total_mb": cache_bytes / 1_000_000,
                "logs_total_mb": log_bytes / 1_000_000,
                "all_storage_mb": (dataset_bytes + shard_bytes + cache_bytes + log_bytes) / 1_000_000,
            },
            "estimated_10000": {
                "generation_seconds": generation_seconds * scale,
                "generation_hours": generation_seconds * scale / 3600,
                "total_pipeline_hours": total_seconds * scale / 3600,
                "dataset_and_shards_mb": (dataset_bytes + shard_bytes) * scale / 1_000_000,
                "all_storage_including_resume_cache_mb": (
                    dataset_bytes + shard_bytes + cache_bytes + log_bytes
                ) * scale / 1_000_000,
            },
            "chunks": [{
                "family": state["family"], "first": state["first"], "last": state["last"],
                "duration_seconds": state["chunk_duration_seconds"],
                "node_peak_rss_mb": state["renderer"].get("node_peak_rss_mb"),
                "retry_number": state["retry_number"],
                "resumed_from_cache": state.get("resumed_from_cache", False),
            } for state in chunk_states],
            "shard_manifest": shard_result,
            "qa": qa,
        }
        atomic_json(report_path, report)
        atomic_json(dataset_dir / "qa" / "production_generation_report.json", report)
        if report["passed"]:
            atomic_json(completion_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    report = run(build_parser().parse_args())
    if not report.get("passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
