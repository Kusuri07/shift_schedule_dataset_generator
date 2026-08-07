#!/usr/bin/env python3
"""Generate printable synthetic shift schedules with training ground truth.

Each schedule is exported as a numbered PNG and included in an A4-landscape PDF
volume. Row-level and cell-level answer data are streamed to CSV/JSONL so large
batches can be generated without keeping every annotation in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, IO, Sequence

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import generate_dataset as gen


MANIFEST_FIELDS = [
    'page_number',
    'page_count',
    'page_label',
    'schedule_id',
    'template_id',
    'year',
    'month',
    'people_count',
    'image_file',
    'pdf_file',
    'pdf_page',
]

ROW_TRUTH_FIELDS = [
    'schedule_id',
    'template_id',
    'page_number',
    'page_count',
    'page_label',
    'sheet_name',
    'row_id',
    'row_index',
    'excel_row',
    'group',
    'name',
    'surname',
    'surname_rank',
    'surname_population',
    'surname_hanja_variants',
    'surname_source_method',
    'surname_source_url',
    'given_name',
    'birth_year',
    'gender',
    'day_count',
    'codes_canonical',
    'codes_display',
    'codes_canonical_joined',
    'codes_display_joined',
    'name_cell',
    'image_path',
]

CELL_TRUTH_FIELDS = [
    'schedule_id',
    'template_id',
    'page_number',
    'page_count',
    'page_label',
    'row_id',
    'row_index',
    'name',
    'surname',
    'surname_rank',
    'surname_population',
    'birth_year',
    'gender',
    'group',
    'day',
    'date',
    'canonical_code',
    'display_code',
    'excel_cell',
    'bbox_px',
    'name_bbox_px',
    'image_path',
]


class AnnotationWriters:
    """Stream row/cell annotations to UTF-8 CSV and JSONL files."""

    def __init__(self, annotations_dir: Path) -> None:
        annotations_dir.mkdir(parents=True, exist_ok=True)
        self.rows_csv_file: IO[str] = (annotations_dir / 'rows.csv').open(
            'w', newline='', encoding='utf-8-sig'
        )
        self.cells_csv_file: IO[str] = (annotations_dir / 'cells.csv').open(
            'w', newline='', encoding='utf-8-sig'
        )
        self.rows_jsonl_file: IO[str] = (annotations_dir / 'rows.jsonl').open(
            'w', encoding='utf-8'
        )
        self.cells_jsonl_file: IO[str] = (annotations_dir / 'cells.jsonl').open(
            'w', encoding='utf-8'
        )
        self.rows_csv = csv.DictWriter(self.rows_csv_file, fieldnames=ROW_TRUTH_FIELDS)
        self.cells_csv = csv.DictWriter(self.cells_csv_file, fieldnames=CELL_TRUTH_FIELDS)
        self.rows_csv.writeheader()
        self.cells_csv.writeheader()
        self.row_count = 0
        self.cell_count = 0

    def write_row(self, row_record: dict[str, Any]) -> None:
        self.rows_jsonl_file.write(
            json.dumps(row_record, ensure_ascii=False, separators=(',', ':')) + '\n'
        )
        csv_record = dict(row_record)
        csv_record['codes_canonical'] = json.dumps(
            row_record['codes_canonical'], ensure_ascii=False, separators=(',', ':')
        )
        csv_record['codes_display'] = json.dumps(
            row_record['codes_display'], ensure_ascii=False, separators=(',', ':')
        )
        self.rows_csv.writerow(csv_record)
        self.row_count += 1

    def write_cell(self, cell_record: dict[str, Any]) -> None:
        self.cells_jsonl_file.write(
            json.dumps(cell_record, ensure_ascii=False, separators=(',', ':')) + '\n'
        )
        csv_record = dict(cell_record)
        csv_record['bbox_px'] = json.dumps(cell_record['bbox_px'], separators=(',', ':'))
        csv_record['name_bbox_px'] = json.dumps(
            cell_record['name_bbox_px'], separators=(',', ':')
        )
        self.cells_csv.writerow(csv_record)
        self.cell_count += 1

    def close(self) -> None:
        for file in (
            self.rows_csv_file,
            self.cells_csv_file,
            self.rows_jsonl_file,
            self.cells_jsonl_file,
        ):
            file.close()


def pdf_volume_name(first_page: int, last_page: int) -> str:
    return f'shift_schedules_{first_page:04d}-{last_page:04d}.pdf'


def build_records(
    *,
    count: int,
    seed: int,
    min_people: int,
    max_people: int,
    pages_per_pdf: int,
    render_chunk_size: int,
    images_dir: Path,
) -> tuple[list[dict[str, Any]], int, int]:
    config = gen.GeneratorConfig(
        count=count,
        seed=seed,
        output_dir=str(images_dir.parent),
        min_people=min_people,
        max_people=max_people,
        ensure_all_codes=False,
        scrape_recent_names=False,
        scrape_surnames=False,
        show_page_numbers=True,
    )
    names = gen.build_name_dictionary(
        config.min_birth_year,
        config.max_birth_year,
        scrape_recent=False,
    )
    surname_entries = gen.build_surname_dictionary(
        scrape=False,
        max_rank=config.surname_max_rank,
        min_population=config.surname_min_population,
    )
    surname_pool = gen.aggregate_surnames(surname_entries)

    records: list[dict[str, Any]] = []
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = images_dir.parent / 'annotations'
    gen.export_dictionaries(names, surname_entries, annotations_dir / 'dictionaries')
    writers = AnnotationWriters(annotations_dir)

    try:
        for chunk_start in range(1, count + 1, render_chunk_size):
            chunk_end = min(chunk_start + render_chunk_size - 1, count)
            schedules: list[gen.ScheduleRecord] = []
            for page_number in range(chunk_start, chunk_end + 1):
                template_id = gen.TEMPLATE_IDS[(page_number - 1) % len(gen.TEMPLATE_IDS)]
                schedule_rng = random.Random(seed + page_number * 1009)
                schedules.append(gen.generate_schedule(
                    page_number,
                    config,
                    schedule_rng,
                    names,
                    surname_pool,
                    forced_template=template_id,
                ))

            gen.render_dataset_workbook(
                schedules,
                names,
                surname_entries,
                surname_pool,
                images_dir.parent,
                export_workbook=False,
            )

            for schedule in schedules:
                relative_image_path = schedule.clean_image_path
                for row_index, row in enumerate(schedule.rows, start=1):
                    writers.write_row({
                        'schedule_id': schedule.schedule_id,
                        'template_id': schedule.template_id,
                        'page_number': schedule.page_number,
                        'page_count': schedule.page_count,
                        'page_label': schedule.page_label,
                        'sheet_name': schedule.sheet_name,
                        'row_id': row.row_id,
                        'row_index': row_index,
                        'excel_row': row.excel_row,
                        'group': row.group,
                        'name': row.name,
                        'surname': row.surname,
                        'surname_rank': row.surname_rank,
                        'surname_population': row.surname_population,
                        'surname_hanja_variants': row.surname_hanja_variants,
                        'surname_source_method': row.surname_source_method,
                        'surname_source_url': row.surname_source_url,
                        'given_name': row.given_name,
                        'birth_year': row.birth_year,
                        'gender': row.gender,
                        'day_count': schedule.day_count,
                        'codes_canonical': row.codes_canonical,
                        'codes_display': row.codes_display,
                        'codes_canonical_joined': '|'.join(row.codes_canonical),
                        'codes_display_joined': '|'.join(row.codes_display),
                        'name_cell': row.name_cell,
                        'image_path': relative_image_path,
                    })

                rows_by_id = {row.row_id: row for row in schedule.rows}
                for annotation in schedule.cell_annotations:
                    row = rows_by_id[annotation['row_id']]
                    day = int(annotation['day'])
                    excel_cell = (
                        f'{gen.excel_column_name(gen.schedule_day_start_col_1based(schedule) + day - 1)}'
                        f'{row.excel_row}'
                    )
                    writers.write_cell({
                        'schedule_id': schedule.schedule_id,
                        'template_id': schedule.template_id,
                        'page_number': schedule.page_number,
                        'page_count': schedule.page_count,
                        'page_label': schedule.page_label,
                        'row_id': annotation['row_id'],
                        'row_index': annotation['row_index'],
                        'name': annotation['name'],
                        'surname': annotation['surname'],
                        'surname_rank': annotation['surname_rank'],
                        'surname_population': annotation['surname_population'],
                        'birth_year': annotation['birth_year'],
                        'gender': annotation['gender'],
                        'group': annotation['group'],
                        'day': day,
                        'date': annotation['date'],
                        'canonical_code': annotation['canonical_code'],
                        'display_code': annotation['display_code'],
                        'excel_cell': excel_cell,
                        'bbox_px': annotation['bbox_px'],
                        'name_bbox_px': annotation['name_bbox_px'],
                        'image_path': relative_image_path,
                    })

                page_number = schedule.page_number
                volume_index = (page_number - 1) // pages_per_pdf
                first_page = volume_index * pages_per_pdf + 1
                last_page = min(first_page + pages_per_pdf - 1, count)
                records.append({
                    'page_number': page_number,
                    'page_count': count,
                    'page_label': schedule.page_label,
                    'schedule_id': schedule.schedule_id,
                    'template_id': schedule.template_id,
                    'year': schedule.year,
                    'month': schedule.month,
                    'people_count': len(schedule.rows),
                    'image_file': relative_image_path,
                    'pdf_file': (
                        Path('pdf') / pdf_volume_name(first_page, last_page)
                    ).as_posix(),
                    'pdf_page': page_number - first_page + 1,
                })
                schedule.cell_annotations.clear()

            print(
                f'Generated Excel-rendered PNGs and answers {chunk_end}/{count}',
                flush=True,
            )
    finally:
        writers.close()

    return records, writers.row_count, writers.cell_count


def write_manifests(records: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / 'print_manifest.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    json_path = output_dir / 'print_manifest.json'
    json_path.write_text(
        json.dumps({
            'dataset_version': 'print-with-ground-truth-2.0',
            'schedule_count': len(records),
            'first_page': records[0]['page_label'] if records else '',
            'last_page': records[-1]['page_label'] if records else '',
            'records': records,
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def write_annotation_manifest(
    records: list[dict[str, Any]],
    output_dir: Path,
    row_count: int,
    cell_count: int,
    seed: int,
) -> None:
    annotations_dir = output_dir / 'annotations'
    manifest = {
        'dataset_version': 'print-with-ground-truth-2.0',
        'seed': seed,
        'schedule_count': len(records),
        'row_answer_count': row_count,
        'cell_answer_count': cell_count,
        'join_keys': {
            'schedule_to_row': 'schedule_id',
            'row_to_cell': 'row_id',
            'schedule_page': 'page_number',
        },
        'files': {
            'row_answers_csv': 'rows.csv',
            'row_answers_jsonl': 'rows.jsonl',
            'cell_answers_csv': 'cells.csv',
            'cell_answers_jsonl': 'cells.jsonl',
            'dictionaries': 'dictionaries/',
        },
        'bbox_format': '[left, top, right, bottom] in PNG pixels',
        'schedule_pages': [{
            'schedule_id': record['schedule_id'],
            'template_id': record['template_id'],
            'page_number': record['page_number'],
            'page_count': record['page_count'],
            'page_label': record['page_label'],
            'people_count': record['people_count'],
            'image_file': record['image_file'],
        } for record in records],
    }
    (annotations_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    guide = f"""학습용 정답 데이터 안내

- 생성된 근무표: {len(records):,}장
- 행 단위 정답: {row_count:,}건
- 셀 단위 정답: {cell_count:,}건
- rows.csv / rows.jsonl: 직원 행별 이름, 인구통계, 근무 코드 배열
- cells.csv / cells.jsonl: 날짜별 정답 코드, 화면 표시 코드, PNG 픽셀 좌표
- dictionaries/: 이름·성씨·근무 코드 생성 사전
- 연결 키: schedule_id로 근무표와 행을, row_id로 행과 셀을 연결
- bbox_px 형식: [왼쪽, 위, 오른쪽, 아래] (PNG 픽셀 좌표)
- canonical_code는 학습 정답, display_code는 이미지에 실제로 표시된 대소문자 변형
"""
    (annotations_dir / 'ANSWER_GUIDE_KO.txt').write_text(guide, encoding='utf-8')


def create_pdf_volumes(
    records: list[dict[str, Any]],
    output_dir: Path,
    pages_per_pdf: int,
) -> list[Path]:
    pdf_dir = output_dir / 'pdf'
    pdf_dir.mkdir(parents=True, exist_ok=True)
    page_width, page_height = landscape(A4)
    margin = 14
    pdf_paths: list[Path] = []

    for volume_start in range(0, len(records), pages_per_pdf):
        volume_records = records[volume_start:volume_start + pages_per_pdf]
        first_page = int(volume_records[0]['page_number'])
        last_page = int(volume_records[-1]['page_number'])
        pdf_path = pdf_dir / pdf_volume_name(first_page, last_page)
        document = canvas.Canvas(
            str(pdf_path),
            pagesize=(page_width, page_height),
            pageCompression=1,
        )
        document.setTitle(f'Synthetic shift schedules {first_page}-{last_page}')
        document.setAuthor('shift_schedule_dataset_generator')

        for record in volume_records:
            image_path = output_dir / str(record['image_file'])
            image_reader = ImageReader(str(image_path))
            image_width, image_height = image_reader.getSize()
            scale = min(
                (page_width - margin * 2) / image_width,
                (page_height - margin * 2) / image_height,
            )
            draw_width = image_width * scale
            draw_height = image_height * scale
            x = (page_width - draw_width) / 2
            y = (page_height - draw_height) / 2
            document.drawImage(
                image_reader,
                x,
                y,
                width=draw_width,
                height=draw_height,
                preserveAspectRatio=True,
                mask='auto',
            )
            document.showPage()

        document.save()
        pdf_paths.append(pdf_path)
        print(
            f'Created PDF {pdf_path.name} ({len(volume_records)} pages)',
            flush=True,
        )

    return pdf_paths


def create_answer_index(output_dir: Path, count: int) -> Path:
    """Build a compact Excel index containing print and row-level answers."""
    node_executable, node_modules = gen.resolve_artifact_tool_runtime()
    builder_source = Path(__file__).resolve().with_name('build_print_index.mjs')
    if not builder_source.exists():
        raise RuntimeError(f'Answer-index builder not found: {builder_source}')
    output_path = output_dir / f'training_answer_index_{count}.xlsx'

    with tempfile.TemporaryDirectory(prefix='shift-answer-index-') as temp_name:
        temp_dir = Path(temp_name)
        builder_path = temp_dir / builder_source.name
        node_modules_link = temp_dir / 'node_modules'
        shutil.copy2(builder_source, builder_path)
        gen.create_node_modules_link(node_modules_link, node_modules)
        try:
            completed = subprocess.run(
                [node_executable, str(builder_path), str(output_dir), str(output_path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(f'Answer-index builder failed: {detail}')
            if completed.stdout.strip():
                print(completed.stdout.strip(), flush=True)
        finally:
            if node_modules_link.exists():
                os.rmdir(node_modules_link)
    return output_path


def write_print_guide(output_dir: Path, count: int, pages_per_pdf: int) -> None:
    volume_count = math.ceil(count / pages_per_pdf)
    guide = f"""합성 병동 근무표 {count:,}장 인쇄 가이드

1. pdf 폴더의 파일을 이름 순서대로 엽니다.
2. 먼저 첫 PDF의 1페이지만 시험 인쇄합니다.
3. 용지 방향은 가로, 크기는 A4, 배율은 '인쇄 가능 영역에 맞춤'을 선택합니다.
4. 순서 보존을 위해 한 부씩 인쇄하고, 가능하면 단면 인쇄를 사용합니다.
5. PDF는 {pages_per_pdf}페이지씩 총 {volume_count}권입니다.
6. 각 표 우측의 '페이지 n/{count}'와 print_manifest.csv를 대조합니다.
7. A3 용지를 쓰면 같은 PDF를 A3 가로와 '맞춤'으로 확대 인쇄할 수 있습니다.
8. 학습용 정답은 annotations 폴더에 있으며 인쇄할 필요는 없습니다.
"""
    (output_dir / 'PRINT_GUIDE_KO.txt').write_text(guide, encoding='utf-8')


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate numbered PNG schedules and print-ready PDF volumes.'
    )
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=20260724)
    parser.add_argument('--min-people', type=int, default=18)
    parser.add_argument('--max-people', type=int, default=32)
    parser.add_argument('--pages-per-pdf', type=int, default=100)
    parser.add_argument('--render-chunk-size', type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count < 1:
        raise ValueError('count must be at least 1')
    if args.min_people < 1 or args.max_people < args.min_people:
        raise ValueError('invalid people range')
    if args.pages_per_pdf < 1:
        raise ValueError('pages-per-pdf must be at least 1')
    if args.render_chunk_size < 1:
        raise ValueError('render-chunk-size must be at least 1')

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, row_answer_count, cell_answer_count = build_records(
        count=args.count,
        seed=args.seed,
        min_people=args.min_people,
        max_people=args.max_people,
        pages_per_pdf=args.pages_per_pdf,
        render_chunk_size=args.render_chunk_size,
        images_dir=output_dir / 'images',
    )
    write_manifests(records, output_dir)
    write_annotation_manifest(
        records,
        output_dir,
        row_answer_count,
        cell_answer_count,
        args.seed,
    )
    pdf_paths = create_pdf_volumes(records, output_dir, args.pages_per_pdf)
    write_print_guide(output_dir, args.count, args.pages_per_pdf)
    answer_index_path = create_answer_index(output_dir, args.count)
    summary = {
        'schedule_count': len(records),
        'image_count': len(records),
        'pdf_count': len(pdf_paths),
        'row_answer_count': row_answer_count,
        'cell_answer_count': cell_answer_count,
        'answer_index_file': answer_index_path.name,
        'render_chunk_size': args.render_chunk_size,
        'pages_per_pdf': args.pages_per_pdf,
        'first_page': records[0]['page_label'],
        'last_page': records[-1]['page_label'],
        'output_dir': str(output_dir),
    }
    (output_dir / 'generation_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
