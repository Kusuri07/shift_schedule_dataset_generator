#!/usr/bin/env python3
"""Generate 10k in-distribution + held-out OOD schedules under one master split.

The split is planned, hashed and written before the first workbook/PNG or
annotation is produced.  Rendering is chunked into retained Excel workbooks so
10,200 sheets do not need to coexist in one artifact-tool process.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Mapping

import generate_dataset as generator
from shift_ocr.charset import coverage_report, load_charset, write_coverage_report
from shift_ocr.master_split import MasterSplit, create_master_split, write_master_split
from shift_ocr.shards import build_parquet_shards, core_checksum, iter_jsonl


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
    "visibility", "ignore", "name_bbox_px", "name_cell_polygon", "image_path", "image_width", "image_height",
]
OBJECT_FIELDS = [
    "schedule_id", "split", "cv_fold", "master_split_sha256", "template_id", "layout_family",
    "object_type", "display_text", "canonical_code", "row_id", "row_index", "day", "bbox_px",
    "cell_polygon", "text_polygon", "text_polygon_source", "visibility", "ignore", "image_path",
    "image_width", "image_height",
]
JSON_FIELDS = {"codes_canonical", "codes_display", "bbox_px", "cell_polygon", "text_polygon", "name_bbox_px", "name_cell_polygon"}


class StreamingAnnotations:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.files = {}
        self.csv_writers = {}
        self.counts = {"rows": 0, "cells": 0, "objects": 0}
        for name, fields in (("rows", ROW_FIELDS), ("cells", CELL_FIELDS), ("objects", OBJECT_FIELDS)):
            jsonl = self.stack.enter_context((directory / f"{name}.jsonl").open("w", encoding="utf-8"))
            csv_file = self.stack.enter_context((directory / f"{name}.csv").open("w", newline="", encoding="utf-8-sig"))
            writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            self.files[name] = jsonl
            self.csv_writers[name] = writer

    def write(self, name: str, item: Mapping[str, Any]) -> None:
        record = {field: item.get(field) for field in self.csv_writers[name].fieldnames}
        self.files[name].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        csv_record = dict(record)
        for field in JSON_FIELDS & csv_record.keys():
            csv_record[field] = json.dumps(csv_record[field], ensure_ascii=False, separators=(",", ":"))
        self.csv_writers[name].writerow(csv_record)
        self.counts[name] += 1

    def close(self) -> None:
        self.stack.close()


def schedule_seed(root_seed: int, schedule_id: str) -> int:
    return int(hashlib.sha256(f"{root_seed}:{schedule_id}".encode()).hexdigest()[:15], 16)


def plan_family(count: int, root_seed: int, templates: list[str], prefix: str, *, ood: bool) -> list[dict[str, Any]]:
    plans = []
    for index in range(1, count + 1):
        schedule_id = f"{prefix}_{index:04d}"
        seed = schedule_seed(root_seed, schedule_id)
        rng = random.Random(seed)
        template = templates[(index - 1) % len(templates)]
        month = rng.randint(1, 12)
        people_count = rng.randint(18, 40 if ood else 32)
        plans.append({
            "schedule_id": schedule_id,
            "template_id": template,
            "layout_family": template,
            "month": month,
            "people_count": people_count,
            "seed": seed,
            "capture_target": 0,
            "schedule_index": index,
        })
    return plans


def assign_real_capture_targets(plans: list[dict[str, Any]], schedule_count: int, image_target: int, seed: int) -> None:
    if schedule_count > len(plans) or image_target < schedule_count:
        raise ValueError("capture targets require image_target >= schedule_count <= planned schedules")
    selected = sorted(plans, key=lambda item: hashlib.sha256(f"capture:{seed}:{item['schedule_id']}".encode()).digest())[:schedule_count]
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
            "surname_source_method": row.surname_source_method, "surname_source_url": row.surname_source_url,
            "given_name": row.given_name, "birth_year": row.birth_year, "gender": row.gender,
            "day_count": schedule.day_count, "codes_canonical": row.codes_canonical,
            "codes_display": row.codes_display, "name_cell": row.name_cell,
        })
    for annotation in schedule.cell_annotations:
        row = rows[annotation["row_id"]]
        excel_cell = f"{generator.excel_column_name(generator.schedule_day_start_col_1based(schedule) + int(annotation['day']) - 1)}{row.excel_row}"
        writers.write("cells", {**annotation, **common, "excel_cell": excel_cell})
    for annotation in schedule.training_objects:
        writers.write("objects", {**annotation, **common})


def render_plans(
    plans: list[dict[str, Any]], config: generator.GeneratorConfig, names, surname_entries, surname_pool,
    output_dir: Path, split: MasterSplit, writers: StreamingAnnotations, chunk_size: int,
) -> list[dict[str, Any]]:
    manifest_rows = []
    workbooks_dir = output_dir / "workbooks"
    workbooks_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(plans), chunk_size):
        selected = plans[start:start + chunk_size]
        schedules = [make_schedule(plan, config, names, surname_pool) for plan in selected]
        for schedule, plan in zip(schedules, selected):
            if schedule.schedule_id != plan["schedule_id"]:
                raise ValueError("schedule regeneration diverged from master plan")
            schedule.page_number = start + schedules.index(schedule) + 1
            schedule.page_count = len(plans)
            schedule.show_page_number = config.show_page_numbers
        if start == 0 and config.ensure_all_codes:
            generator.ensure_code_coverage(schedules, random.Random(config.seed), config.case_mutation_probability)
        first, last = start + 1, start + len(schedules)
        workbook_name = f"workbooks/synthetic_shift_dataset_{config.schedule_id_prefix}_{first:04d}-{last:04d}.xlsx"
        generator.render_dataset_workbook(
            schedules, names, surname_entries, surname_pool, output_dir,
            export_workbook=True, workbook_name=workbook_name,
        )
        for schedule in schedules:
            write_schedule_annotations(schedule, split, writers)
            manifest_rows.append({
                "schedule_id": schedule.schedule_id, "template_id": schedule.template_id,
                "layout_family": split.require(schedule.schedule_id).layout_family,
                "split": split.require(schedule.schedule_id).split,
                "cv_fold": split.require(schedule.schedule_id).cv_fold,
                "year": schedule.year, "month": schedule.month, "day_count": schedule.day_count,
                "people_count": len(schedule.rows), "image_path": schedule.clean_image_path,
                "image_size": [schedule.image_width, schedule.image_height], "workbook_path": workbook_name,
            })
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--ood-count", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("training_dataset"))
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--render-chunk-size", type=int, default=25)
    parser.add_argument("--real-schedule-target", type=int)
    parser.add_argument("--real-photo-target", type=int)
    parser.add_argument("--skip-parquet", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_plans = plan_family(args.count, args.seed, generator.TEMPLATE_IDS, "schedule", ood=False)
    ood_plans = plan_family(args.ood_count, args.seed + 1, generator.OOD_TEMPLATE_IDS, "ood_schedule", ood=True)
    real_schedule_target = args.real_schedule_target if args.real_schedule_target is not None else min(300, args.count)
    real_photo_target = args.real_photo_target if args.real_photo_target is not None else (
        1000 if real_schedule_target >= 300 else max(real_schedule_target, round(real_schedule_target * 10 / 3))
    )
    assign_real_capture_targets(base_plans, real_schedule_target, real_photo_target, args.seed)
    for item in sorted(ood_plans, key=lambda value: value["schedule_id"])[: min(100, len(ood_plans))]:
        item["capture_target"] = 1
    config_payload = vars(args).copy()
    config_payload["output_dir"] = str(args.output_dir.resolve())
    split_records, split_metadata = create_master_split(
        base_plans, seed=args.seed, config=config_payload, ood_schedules=ood_plans
    )
    split_dir = args.output_dir / "splits"
    write_master_split(split_records, split_metadata, split_dir)
    master = MasterSplit.load(split_dir / "master_split.jsonl")

    names = generator.build_name_dictionary(generator.MIN_BIRTH_YEAR, generator.MAX_BIRTH_YEAR)
    surname_entries = generator.build_surname_dictionary(scrape=False)
    surname_pool = generator.aggregate_surnames(surname_entries)
    generator.export_dictionaries(names, surname_entries, args.output_dir)
    writers = StreamingAnnotations(args.output_dir / "annotations")
    try:
        base_config = generator.GeneratorConfig(
            count=args.count, seed=args.seed, output_dir=str(args.output_dir), template_ids=generator.TEMPLATE_IDS.copy(),
            schedule_id_prefix="schedule", scrape_surnames=False,
        )
        ood_config = generator.GeneratorConfig(
            count=args.ood_count, seed=args.seed + 1, output_dir=str(args.output_dir), template_ids=generator.OOD_TEMPLATE_IDS.copy(),
            schedule_id_prefix="ood_schedule", scrape_surnames=False, min_people=18, max_people=40,
        )
        manifest_rows = render_plans(base_plans, base_config, names, surname_entries, surname_pool, args.output_dir, master, writers, args.render_chunk_size)
        manifest_rows += render_plans(ood_plans, ood_config, names, surname_entries, surname_pool, args.output_dir, master, writers, args.render_chunk_size)
    finally:
        writers.close()

    manifest = {
        "dataset_version": "2.0-training",
        "master_split_sha256": master.metadata["split_sha256"],
        "schedule_count": len(base_plans), "ood_layout_schedule_count": len(ood_plans),
        "annotation_counts": writers.counts, "schedules": manifest_rows,
    }
    (args.output_dir / "annotations" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    charset = load_charset(Path(__file__).resolve().parent / "data" / "korean_charset_v1.txt")
    report = coverage_report(iter_jsonl(args.output_dir / "annotations" / "objects.jsonl"), charset)
    write_coverage_report(report, args.output_dir / "annotations" / "charset_coverage.json")
    if not args.skip_parquet:
        build_parquet_shards(args.output_dir / "annotations" / "objects.jsonl", master, args.output_dir / "shards")
    print(json.dumps({**manifest, "schedules": f"{len(manifest_rows)} records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
