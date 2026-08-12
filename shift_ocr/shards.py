"""ZSTD Parquet object shards with an image-level random-access index."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .master_split import MasterSplit


def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional heavy dependency
        raise RuntimeError("Parquet sharding requires pyarrow from requirements-training.txt") from exc
    return pa, pq


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def core_checksum(records: Iterable[Mapping[str, Any]]) -> str:
    xor_value = 0
    sum_value = 0
    count = 0
    modulus = 1 << 256
    for item in records:
        core = {key: item.get(key) for key in (
            "schedule_id", "image_path", "object_type", "row_id", "row_index", "day",
            "display_text", "canonical_code", "cell_polygon", "text_polygon",
        )}
        for key in ("row_index", "day"):
            core[key] = int(core[key]) if core[key] not in (None, "") else None
        for key in ("row_id", "canonical_code"):
            core[key] = None if core[key] in (None, "") else str(core[key])
        for key in ("schedule_id", "image_path", "object_type", "display_text"):
            core[key] = str(core[key] or "")
        digest = int.from_bytes(hashlib.sha256(
            json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).digest(), "big")
        xor_value ^= digest
        sum_value = (sum_value + digest) % modulus
        count += 1
    return hashlib.sha256(f"{count}:{xor_value:064x}:{sum_value:064x}".encode()).hexdigest()


def _decode_json_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(record)
    for field in ("bbox_px", "cell_polygon", "text_polygon", "name_bbox_px", "name_cell_polygon"):
        value = output.get(field)
        if isinstance(value, str) and value:
            output[field] = json.loads(value)
    return output


def verify_compatible_annotations(objects_jsonl: Path, objects_csv: Path, shard_dir: Path | None = None) -> dict[str, Any]:
    import csv

    json_records = list(iter_jsonl(objects_jsonl))
    with objects_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        csv_records = [_decode_json_fields(item) for item in csv.DictReader(stream)]
    json_checksum = core_checksum(json_records)
    csv_checksum = core_checksum(csv_records)
    if len(json_records) != len(csv_records) or json_checksum != csv_checksum:
        raise ValueError("CSV and JSONL object annotations disagree")
    result = {
        "jsonl_count": len(json_records), "csv_count": len(csv_records),
        "jsonl_checksum": json_checksum, "csv_checksum": csv_checksum,
    }
    if shard_dir is not None:
        _pa, pq = _arrow()
        parquet_records = []
        for path in sorted(shard_dir.glob("*.parquet")):
            if path.name == "image_index.parquet":
                continue
            parquet_records.extend(_decode_json_fields(item) for item in pq.read_table(path).to_pylist())
        parquet_checksum = core_checksum(parquet_records)
        if len(parquet_records) != len(json_records) or parquet_checksum != json_checksum:
            raise ValueError("Parquet shards disagree with compatibility annotations")
        result.update({"parquet_count": len(parquet_records), "parquet_checksum": parquet_checksum})
    return result


def _table_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        for field in ("bbox_px", "cell_polygon", "text_polygon", "name_bbox_px", "name_cell_polygon"):
            if field in row and not isinstance(row[field], str):
                row[field] = json.dumps(row[field], ensure_ascii=False, separators=(",", ":"))
        rows.append(row)
    return rows


def build_parquet_shards(
    objects_jsonl: Path,
    master_split: MasterSplit,
    output_dir: Path,
    *,
    max_schedules: int = 500,
    max_objects: int = 250_000,
) -> dict[str, Any]:
    pa, pq = _arrow()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_records = list(iter_jsonl(objects_jsonl))
    for item in source_records:
        split = master_split.require(str(item["schedule_id"]), item.get("split")).split
        item = dict(item)
        item["split"] = split
        item["master_split_sha256"] = master_split.metadata["split_sha256"]
        grouped[str(item["schedule_id"])].append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    shard_manifest: list[dict[str, Any]] = []
    for split in ("train", "validation", "test", "ood_layout"):
        schedule_ids = sorted(
            schedule_id for schedule_id in grouped
            if master_split.require(schedule_id).split == split
        )
        shard_number = 0
        cursor = 0
        while cursor < len(schedule_ids):
            selected: list[str] = []
            object_count = 0
            while cursor < len(schedule_ids) and len(selected) < max_schedules:
                schedule_id = schedule_ids[cursor]
                count = len(grouped[schedule_id])
                if selected and object_count + count > max_objects:
                    break
                selected.append(schedule_id)
                object_count += count
                cursor += 1
            shard_number += 1
            shard_name = f"{split}-{shard_number:04d}.parquet"
            shard_path = output_dir / shard_name
            writer = None
            row_offset = 0
            row_group = 0
            try:
                for schedule_id in selected:
                    records = _table_rows(grouped[schedule_id])
                    table = pa.Table.from_pylist(records)
                    if writer is None:
                        writer = pq.ParquetWriter(shard_path, table.schema, compression="zstd")
                    writer.write_table(table, row_group_size=len(records))
                    image_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                    for record in records:
                        image_groups[str(record["image_path"])].append(record)
                    for image_path, image_records in image_groups.items():
                        index_rows.append({
                            "image_path": image_path,
                            "schedule_id": schedule_id,
                            "split": split,
                            "shard": shard_name,
                            "row_group": row_group,
                            "offset": row_offset,
                            "object_count": len(image_records),
                            "master_split_sha256": master_split.metadata["split_sha256"],
                        })
                    row_offset += len(records)
                    row_group += 1
            finally:
                if writer is not None:
                    writer.close()
            shard_manifest.append({
                "split": split,
                "shard": shard_name,
                "schedule_count": len(selected),
                "object_count": object_count,
            })

    index_path = output_dir / "image_index.parquet"
    pq.write_table(pa.Table.from_pylist(index_rows), index_path, compression="zstd")
    result = {
        "schema_version": "training_shards_v1",
        "master_split_sha256": master_split.metadata["split_sha256"],
        "source_record_count": len(source_records),
        "source_core_checksum": core_checksum(source_records),
        "index_record_count": len(index_rows),
        "shards": shard_manifest,
    }
    (output_dir / "shards.manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


class ParquetImageStore:
    """Read only the row group for an image and keep a bounded shard LRU."""

    def __init__(self, shard_dir: Path, master_split: MasterSplit, cache_size: int = 4) -> None:
        _pa, pq = _arrow()
        self._pq = pq
        self.shard_dir = shard_dir
        self.master_split = master_split
        index = pq.read_table(shard_dir / "image_index.parquet").to_pylist()
        self.index = {str(row["image_path"]): row for row in index}
        self.cache_size = max(1, cache_size)
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def _file(self, shard: str):
        if shard not in self.cache:
            self.cache[shard] = self._pq.ParquetFile(self.shard_dir / shard)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        return self.cache[shard]

    def load_image(self, image_path: str, *, purpose: str) -> list[dict[str, Any]]:
        if image_path not in self.index:
            raise KeyError(image_path)
        index = self.index[image_path]
        self.master_split.authorize(str(index["schedule_id"]), purpose, str(index["split"]))
        rows = self._file(str(index["shard"])).read_row_group(int(index["row_group"])).to_pylist()
        return [row for row in rows if str(row["image_path"]) == image_path]
