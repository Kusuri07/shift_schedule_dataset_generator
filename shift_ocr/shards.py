"""ZSTD Parquet object shards with an image-level random-access index."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .master_split import MasterSplit


SHARD_SCHEMA_VERSION = "training_shards_v2"
CHECKSUM_SCHEMA_VERSION = "object_annotations_v2"
LEGACY_CHECKSUM_SCHEMA_VERSION = "object_annotations_v1"

# These are the fields that affect a model target or determine whether an
# annotation is eligible for training/evaluation.  Keeping the list explicit
# makes the compatibility contract stable while preventing CSV/JSONL/Parquet
# parity checks from silently ignoring geometry or visibility changes.
COMPATIBILITY_FIELDS_V2 = (
    "schedule_id", "split", "cv_fold", "master_split_sha256", "template_id",
    "layout_family", "object_type", "display_text", "canonical_code", "row_id",
    "row_index", "day", "bbox_px", "cell_polygon", "text_polygon",
    "text_polygon_source", "text_polygon_validation_max_error_px", "visibility",
    "ignore", "source_domain", "registration_profile",
    "registration_high_confidence", "text_polygon_margin_px", "image_path",
    "image_width", "image_height",
)
COMPATIBILITY_FIELDS_V1 = (
    "schedule_id", "image_path", "object_type", "row_id", "row_index", "day",
    "display_text", "canonical_code", "cell_polygon", "text_polygon",
)
LEGACY_COMPATIBILITY_FIELDS_V2 = tuple(
    field for field in COMPATIBILITY_FIELDS_V2
    if field not in {
        "source_domain", "registration_profile", "registration_high_confidence",
        "text_polygon_margin_px",
    }
)

_JSON_FIELDS = {"bbox_px", "cell_polygon", "text_polygon", "name_bbox_px", "name_cell_polygon"}
_INTEGER_FIELDS = {"cv_fold", "row_index", "day", "image_width", "image_height"}
_FLOAT_FIELDS = {"text_polygon_validation_max_error_px", "text_polygon_margin_px", "visibility"}
_NULLABLE_STRING_FIELDS = {"canonical_code", "registration_profile", "row_id"}
_NULLABLE_BOOLEAN_FIELDS = {"registration_high_confidence"}
_CANONICAL_MASTER_FIELDS = ("split", "cv_fold", "master_split_sha256")
_INDEX_CHECKSUM_SCHEMA = "index_rows_v1"


@dataclass
class _ChecksumAccumulator:
    schema_version: str = CHECKSUM_SCHEMA_VERSION
    compatibility_fields: tuple[str, ...] | None = None
    xor_value: int = 0
    sum_value: int = 0
    count: int = 0

    def update(self, item: Mapping[str, Any]) -> None:
        core = _compatibility_record(item, self.schema_version, self.compatibility_fields)
        digest = int.from_bytes(hashlib.sha256(
            json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).digest(), "big")
        self.xor_value ^= digest
        self.sum_value = (self.sum_value + digest) % (1 << 256)
        self.count += 1

    def hexdigest(self) -> str:
        return hashlib.sha256(
            f"{self.count}:{self.xor_value:064x}:{self.sum_value:064x}".encode()
        ).hexdigest()


@dataclass
class _GenericChecksumAccumulator:
    xor_value: int = 0
    sum_value: int = 0
    count: int = 0

    def update(self, item: Mapping[str, Any]) -> None:
        digest = int.from_bytes(hashlib.sha256(json.dumps(
            dict(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode()).digest(), "big")
        self.xor_value ^= digest
        self.sum_value = (self.sum_value + digest) % (1 << 256)
        self.count += 1

    def hexdigest(self) -> str:
        return hashlib.sha256(
            f"{self.count}:{self.xor_value:064x}:{self.sum_value:064x}".encode()
        ).hexdigest()


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


def _nullable_number(value: Any, converter):
    if value in (None, ""):
        return None
    converted = converter(value)
    if isinstance(converted, float) and not math.isfinite(converted):
        raise ValueError("annotation compatibility fields cannot contain NaN or infinity")
    return converted


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid annotation boolean: {value!r}")


def _nullable_boolean(value: Any) -> bool | None:
    # Missing registration confidence is semantically different from false:
    # DenseScheduleDataset excludes DBNet targets only when the key exists and
    # is false, while legacy/synthetic annotations without the key are valid.
    return None if value in (None, "") else _boolean(value)


def _json_value(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("annotation geometry cannot contain NaN or infinity")
        return number
    return value


def _compatibility_record(
    item: Mapping[str, Any], schema_version: str,
    compatibility_fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if schema_version in {LEGACY_CHECKSUM_SCHEMA_VERSION, "training_shards_v1"}:
        core = {key: item.get(key) for key in COMPATIBILITY_FIELDS_V1}
        for key in ("row_index", "day"):
            core[key] = int(core[key]) if core[key] not in (None, "") else None
        for key in ("row_id", "canonical_code"):
            core[key] = None if core[key] in (None, "") else str(core[key])
        for key in ("schedule_id", "image_path", "object_type", "display_text"):
            core[key] = str(core[key] or "")
        return core
    if schema_version not in {CHECKSUM_SCHEMA_VERSION, SHARD_SCHEMA_VERSION}:
        raise ValueError(f"unsupported annotation checksum schema: {schema_version}")

    fields = compatibility_fields or COMPATIBILITY_FIELDS_V2
    if fields not in {COMPATIBILITY_FIELDS_V2, LEGACY_COMPATIBILITY_FIELDS_V2}:
        raise ValueError("unsupported v2 annotation checksum field contract")
    core: dict[str, Any] = {}
    for key in fields:
        value = item.get(key)
        if key in _JSON_FIELDS:
            core[key] = _json_value(value)
        elif key in _INTEGER_FIELDS:
            core[key] = _nullable_number(value, int)
        elif key in _FLOAT_FIELDS:
            core[key] = _nullable_number(value, float)
        elif key == "ignore":
            core[key] = _boolean(value)
        elif key in _NULLABLE_BOOLEAN_FIELDS:
            core[key] = _nullable_boolean(value)
        elif key == "source_domain":
            # Training treats a missing source_domain as synthetic.
            core[key] = str(value or "synthetic")
        elif key in _NULLABLE_STRING_FIELDS:
            core[key] = None if value in (None, "") else str(value)
        else:
            core[key] = str(value or "")
    return core


def core_checksum(
    records: Iterable[Mapping[str, Any]], *, schema_version: str = CHECKSUM_SCHEMA_VERSION,
    compatibility_fields: tuple[str, ...] | None = None,
) -> str:
    accumulator = _ChecksumAccumulator(schema_version, compatibility_fields)
    for item in records:
        accumulator.update(item)
    return accumulator.hexdigest()


def _index_checksum(records: Iterable[Mapping[str, Any]]) -> str:
    accumulator = _GenericChecksumAccumulator()
    for item in records:
        accumulator.update(item)
    return accumulator.hexdigest()


def verify_parquet_index(shard_dir: Path, kind: str) -> dict[str, Any]:
    """Eagerly authenticate a compact index before any indexed batch is used."""

    if kind not in {"image", "recognition"}:
        raise ValueError("index kind must be 'image' or 'recognition'")
    _pa, pq = _arrow()
    manifest, _schema, _fields = _manifest_checksum_contract(shard_dir)
    filename = f"{kind}_index.parquet"
    count_key = f"{kind}_index_record_count"
    checksum_key = f"{kind}_index_checksum"
    path = shard_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"{kind} index is missing: {path}")
    accumulator = _GenericChecksumAccumulator()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16_384):
        for item in batch.to_pylist():
            accumulator.update(item)
    declared_count = manifest.get(count_key)
    declared_checksum = manifest.get(checksum_key)
    if declared_count is None or declared_checksum is None:
        raise ValueError(f"{kind} index manifest contract is missing")
    if accumulator.count != int(declared_count):
        raise ValueError(f"{kind} index record count disagrees with shard manifest")
    if accumulator.hexdigest() != str(declared_checksum):
        raise ValueError(f"{kind} index checksum disagrees with shard manifest")
    return {
        "kind": kind, "path": str(path), "record_count": accumulator.count,
        "checksum": accumulator.hexdigest(),
    }


def iter_verified_index_rows(shard_dir: Path, kind: str) -> Iterator[dict[str, Any]]:
    """Authenticate an index, then stream its rows from a fresh Parquet scan.

    Authentication intentionally completes before the first yield so a
    corrupted index can never feed even one training batch.
    """

    verify_parquet_index(shard_dir, kind)
    _pa, pq = _arrow()
    path = shard_dir / f"{kind}_index.parquet"
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16_384):
        yield from batch.to_pylist()


def _decode_json_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(record)
    for field in _JSON_FIELDS:
        value = output.get(field)
        if isinstance(value, str) and value:
            output[field] = json.loads(value)
    return output


def _checksum_fields(
    schema_version: str, declared_fields: Iterable[str] | None = None,
) -> tuple[str, ...]:
    if schema_version in {LEGACY_CHECKSUM_SCHEMA_VERSION, "training_shards_v1"}:
        if declared_fields is not None and tuple(declared_fields) != COMPATIBILITY_FIELDS_V1:
            raise ValueError("unsupported v1 annotation checksum field contract")
        return COMPATIBILITY_FIELDS_V1
    if schema_version in {CHECKSUM_SCHEMA_VERSION, SHARD_SCHEMA_VERSION}:
        fields = tuple(declared_fields) if declared_fields is not None else COMPATIBILITY_FIELDS_V2
        if fields not in {COMPATIBILITY_FIELDS_V2, LEGACY_COMPATIBILITY_FIELDS_V2}:
            raise ValueError("unsupported v2 annotation checksum field contract")
        return fields
    raise ValueError(f"unsupported annotation checksum schema: {schema_version}")


def _manifest_checksum_contract(
    shard_dir: Path,
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    manifest_path = shard_dir / "shards.manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"shard manifest is required for compatibility verification: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_schema = str(manifest.get("schema_version") or "")
    if shard_schema not in {"training_shards_v1", SHARD_SCHEMA_VERSION}:
        raise ValueError(f"unsupported shard schema: {shard_schema!r}")
    checksum_schema = manifest.get("checksum_schema_version")
    if checksum_schema is None:
        if shard_schema == "training_shards_v1":
            checksum_schema = LEGACY_CHECKSUM_SCHEMA_VERSION
        elif shard_schema == SHARD_SCHEMA_VERSION:
            checksum_schema = CHECKSUM_SCHEMA_VERSION
        else:
            raise ValueError(f"unsupported shard schema: {shard_schema!r}")
    checksum_schema = str(checksum_schema)
    expected_checksum_schema = (
        LEGACY_CHECKSUM_SCHEMA_VERSION
        if shard_schema == "training_shards_v1" else CHECKSUM_SCHEMA_VERSION
    )
    if checksum_schema != expected_checksum_schema:
        raise ValueError(
            f"shard/checksum schema mismatch: {shard_schema} requires {expected_checksum_schema}"
        )
    checksum_fields = _checksum_fields(checksum_schema, manifest.get("checksum_fields"))
    return manifest, checksum_schema, checksum_fields


def _declared_shard_paths(shard_dir: Path, manifest: Mapping[str, Any]) -> list[Path]:
    declarations = manifest.get("shards")
    if declarations is None:
        # Legacy manifests did not enumerate shards; retain their read path.
        return [
            path for path in sorted(shard_dir.glob("*.parquet"))
            if path.name not in {"image_index.parquet", "recognition_index.parquet"}
        ]
    names = [str(item.get("shard") or "") for item in declarations]
    if any(not name or Path(name).name != name for name in names) or len(names) != len(set(names)):
        raise ValueError("shard manifest contains invalid or duplicate shard names")
    actual = {
        path.name for path in shard_dir.glob("*.parquet")
        if path.name not in {"image_index.parquet", "recognition_index.parquet"}
    }
    expected = set(names)
    if actual != expected:
        raise ValueError(
            f"Parquet shard set disagrees with manifest: missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}"
        )
    return [shard_dir / name for name in names]


def _normalized_master_metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": str(item.get("split") or ""),
        "cv_fold": _nullable_number(item.get("cv_fold"), int),
        "master_split_sha256": str(item.get("master_split_sha256") or ""),
    }


def _apply_master_metadata(
    item: Mapping[str, Any], expected: Mapping[str, Any], *, schedule_id: str,
) -> dict[str, Any]:
    output = dict(item)
    actual = _normalized_master_metadata(item)
    normalized_expected = _normalized_master_metadata(expected)
    for field in _CANONICAL_MASTER_FIELDS:
        if actual[field] not in (None, "") and actual[field] != normalized_expected[field]:
            raise ValueError(
                f"master metadata mismatch for {schedule_id}: "
                f"{field}={actual[field]!r}, expected={normalized_expected[field]!r}"
            )
        output[field] = normalized_expected[field]
    return output


def _iter_csv_records(path: Path) -> Iterator[dict[str, Any]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            yield _decode_json_fields(item)


def _iter_parquet_records(
    shard_dir: Path, pq, paths: Iterable[Path] | None = None,
) -> Iterator[dict[str, Any]]:
    for path in paths or (
        path for path in sorted(shard_dir.glob("*.parquet"))
        if path.name not in {"image_index.parquet", "recognition_index.parquet"}
    ):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=16_384):
            for item in batch.to_pylist():
                yield _decode_json_fields(item)


def _stream_parquet_checksum(
    shard_dir: Path, pq, *, schema_version: str, compatibility_fields: tuple[str, ...],
    database: sqlite3.Connection | None = None, paths: Iterable[Path] | None = None,
) -> tuple[int, str]:
    checksum = _ChecksumAccumulator(schema_version, compatibility_fields)
    for item in _iter_parquet_records(shard_dir, pq, paths):
        if database is not None:
            schedule_id = str(item.get("schedule_id") or "")
            metadata = _normalized_master_metadata(item)
            if not schedule_id:
                raise ValueError("Parquet shard record is missing schedule_id")
            if not metadata["split"] or metadata["cv_fold"] is None or not metadata["master_split_sha256"]:
                raise ValueError(f"v2 Parquet record is missing canonical master metadata: {schedule_id}")
            existing = database.execute(
                "SELECT split, cv_fold, master_hash FROM metadata WHERE schedule_id=?", (schedule_id,),
            ).fetchone()
            candidate = (metadata["split"], metadata["cv_fold"], metadata["master_split_sha256"])
            if existing is None:
                database.execute("INSERT INTO metadata VALUES(?,?,?,?)", (schedule_id, *candidate))
            elif tuple(existing) != candidate:
                raise ValueError(f"inconsistent canonical master metadata in Parquet shards: {schedule_id}")
        checksum.update(item)
    if database is not None:
        database.commit()
    return checksum.count, checksum.hexdigest()


def _canonical_source_checksum(
    records: Iterable[Mapping[str, Any]], database: sqlite3.Connection,
    *, schema_version: str, compatibility_fields: tuple[str, ...],
) -> tuple[int, str]:
    checksum = _ChecksumAccumulator(schema_version, compatibility_fields)
    count = 0
    for item in records:
        schedule_id = str(item.get("schedule_id") or "")
        row = database.execute(
            "SELECT split, cv_fold, master_hash FROM metadata WHERE schedule_id=?", (schedule_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"source schedule is absent from Parquet shards: {schedule_id}")
        canonical = _apply_master_metadata(item, {
            "split": row[0], "cv_fold": row[1], "master_split_sha256": row[2],
        }, schedule_id=schedule_id)
        checksum.update(canonical)
        count += 1
    return count, checksum.hexdigest()


def verify_compatible_annotations(objects_jsonl: Path, objects_csv: Path, shard_dir: Path | None = None) -> dict[str, Any]:
    manifest: dict[str, Any] | None = None
    checksum_schema = CHECKSUM_SCHEMA_VERSION
    checksum_fields = COMPATIBILITY_FIELDS_V2
    if shard_dir is not None:
        manifest, checksum_schema, checksum_fields = _manifest_checksum_contract(shard_dir)
    json_accumulator = _ChecksumAccumulator(checksum_schema, checksum_fields)
    for item in iter_jsonl(objects_jsonl):
        json_accumulator.update(item)
    csv_accumulator = _ChecksumAccumulator(checksum_schema, checksum_fields)
    for item in _iter_csv_records(objects_csv):
        csv_accumulator.update(item)
    json_checksum = json_accumulator.hexdigest()
    csv_checksum = csv_accumulator.hexdigest()
    if json_accumulator.count != csv_accumulator.count or json_checksum != csv_checksum:
        raise ValueError("CSV and JSONL object annotations disagree")
    if manifest is not None:
        manifest_source_count = manifest.get("source_record_count")
        if manifest_source_count is not None and int(manifest_source_count) != json_accumulator.count:
            raise ValueError("shard manifest source record count disagrees with annotations")
        manifest_source_checksum = manifest.get("source_core_checksum")
        if manifest_source_checksum is not None and str(manifest_source_checksum) != json_checksum:
            raise ValueError("shard manifest source checksum disagrees with annotations")
    result = {
        "jsonl_count": json_accumulator.count, "csv_count": csv_accumulator.count,
        "jsonl_checksum": json_checksum, "csv_checksum": csv_checksum,
        "checksum_schema_version": checksum_schema,
        "checksum_fields": list(checksum_fields),
        "source_checksum_scope": "source_annotations_before_master_metadata_enrichment",
    }
    if shard_dir is not None:
        _pa, pq = _arrow()
        declared_paths = _declared_shard_paths(shard_dir, manifest or {})
        if checksum_schema == CHECKSUM_SCHEMA_VERSION:
            with tempfile.TemporaryDirectory(prefix="shift-shard-verify-") as temporary_directory:
                database = sqlite3.connect(Path(temporary_directory) / "metadata.sqlite3")
                try:
                    database.execute(
                        "CREATE TABLE metadata(schedule_id TEXT PRIMARY KEY, split TEXT, cv_fold INTEGER, master_hash TEXT)"
                    )
                    parquet_count, parquet_checksum = _stream_parquet_checksum(
                        shard_dir, pq, schema_version=checksum_schema,
                        compatibility_fields=checksum_fields, database=database,
                        paths=declared_paths,
                    )
                    json_count, canonical_json_checksum = _canonical_source_checksum(
                        iter_jsonl(objects_jsonl), database, schema_version=checksum_schema,
                        compatibility_fields=checksum_fields,
                    )
                    csv_count, canonical_csv_checksum = _canonical_source_checksum(
                        _iter_csv_records(objects_csv), database, schema_version=checksum_schema,
                        compatibility_fields=checksum_fields,
                    )
                finally:
                    database.close()
            if canonical_json_checksum != canonical_csv_checksum or json_count != csv_count:
                raise ValueError("canonicalized CSV and JSONL object annotations disagree")
            comparison_checksum = canonical_json_checksum
            manifest_canonical_checksum = manifest.get("canonical_core_checksum") if manifest else None
            if manifest_canonical_checksum is not None and str(manifest_canonical_checksum) != parquet_checksum:
                raise ValueError("shard manifest canonical checksum disagrees with Parquet shards")
            result.update({
                "canonical_jsonl_checksum": canonical_json_checksum,
                "canonical_csv_checksum": canonical_csv_checksum,
                "canonical_metadata_fields": list(_CANONICAL_MASTER_FIELDS),
            })
        else:
            parquet_accumulator = _ChecksumAccumulator(checksum_schema, checksum_fields)
            for item in _iter_parquet_records(shard_dir, pq, declared_paths):
                parquet_accumulator.update(item)
            parquet_count = parquet_accumulator.count
            parquet_checksum = parquet_accumulator.hexdigest()
            comparison_checksum = json_checksum
        if parquet_count != json_accumulator.count or parquet_checksum != comparison_checksum:
            raise ValueError("Parquet shards disagree with compatibility annotations")
        result.update({
            "parquet_count": parquet_count, "parquet_checksum": parquet_checksum,
            "parquet_checksum_scope": (
                "master_enriched_annotations"
                if checksum_schema == CHECKSUM_SCHEMA_VERSION else "legacy_v1_annotations"
            ),
        })
        if manifest and manifest.get("image_index_record_count") is not None:
            image_index = shard_dir / "image_index.parquet"
            if not image_index.exists():
                raise FileNotFoundError(f"image index is missing: {image_index}")
            index_accumulator = _GenericChecksumAccumulator()
            for batch in pq.ParquetFile(image_index).iter_batches(batch_size=16_384):
                for item in batch.to_pylist():
                    index_accumulator.update(item)
            if index_accumulator.count != int(manifest["image_index_record_count"]):
                raise ValueError("image index record count disagrees with shard manifest")
            if index_accumulator.hexdigest() != str(manifest.get("image_index_checksum")):
                raise ValueError("image index checksum disagrees with shard manifest")
        if manifest and manifest.get("recognition_index_record_count") is not None:
            recognition_index = shard_dir / "recognition_index.parquet"
            if not recognition_index.exists():
                raise FileNotFoundError(f"recognition index is missing: {recognition_index}")
            recognition_accumulator = _GenericChecksumAccumulator()
            for batch in pq.ParquetFile(recognition_index).iter_batches(batch_size=16_384):
                for item in batch.to_pylist():
                    recognition_accumulator.update(item)
            if recognition_accumulator.count != int(manifest["recognition_index_record_count"]):
                raise ValueError("recognition index record count disagrees with shard manifest")
            if recognition_accumulator.hexdigest() != str(manifest.get("recognition_index_checksum")):
                raise ValueError("recognition index checksum disagrees with shard manifest")
    return result


def _annotation_arrow_schema(pa):
    """Stable superset schema for mixed synthetic and registered-real rows."""

    field_names = list(dict.fromkeys((
        *COMPATIBILITY_FIELDS_V2,
        "excel_cell", "name", "surname", "birth_year", "gender", "group", "date",
        "display_code", "name_bbox_px", "name_cell_polygon",
    )))
    fields = []
    for name in field_names:
        if name in _INTEGER_FIELDS or name in {"birth_year"}:
            data_type = pa.int64()
        elif name in _FLOAT_FIELDS:
            data_type = pa.float64()
        elif name in _NULLABLE_BOOLEAN_FIELDS or name == "ignore":
            data_type = pa.bool_()
        else:
            # JSON geometry is serialized before Arrow conversion.  Keeping
            # optional text fields nullable avoids schema drift when the first
            # schedule in a shard is synthetic and later rows are real.
            data_type = pa.string()
        fields.append(pa.field(name, data_type, nullable=True))
    return pa.schema(fields)


def _arrow_row(record: Mapping[str, Any], schema) -> dict[str, Any]:
    output = {}
    for field in schema.names:
        value = record.get(field)
        if field in _JSON_FIELDS and value not in (None, "") and not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if field == "source_domain":
            value = value or "synthetic"
        output[field] = value
    return output


def recognition_width_bucket(record: Mapping[str, Any]) -> int:
    """Predict the same width bucket as ``augmentation.rectify_cell``.

    Use Euclidean lengths of both opposing edges so rotation, shear and
    registered-photo homographies cannot change the index/runtime contract.
    """

    polygon = record.get("cell_polygon")
    if isinstance(polygon, str):
        polygon = json.loads(polygon)
    if not polygon or len(polygon) < 4:
        return 640
    try:
        points = [(float(point[0]), float(point[1])) for point in polygon[:4]]
    except (TypeError, ValueError, IndexError):
        return 640
    if not all(math.isfinite(coordinate) for point in points for coordinate in point):
        return 640

    def edge(first: int, second: int) -> float:
        return math.hypot(
            points[second][0] - points[first][0],
            points[second][1] - points[first][1],
        )

    width = max(edge(0, 1), edge(3, 2))
    height = max(1.0, edge(0, 3), edge(1, 2))
    content_width = min(640, max(8, int(round(48.0 * width / height))))
    return 160 if content_width <= 160 else 320 if content_width <= 320 else 640


def _write_index_parquet(pa, pq, path: Path, rows: Iterable[Mapping[str, Any]], schema):
    writer = None
    accumulator = _GenericChecksumAccumulator()
    buffer = []
    try:
        for item in rows:
            normalized = {field: item.get(field) for field in schema.names}
            accumulator.update(normalized)
            buffer.append(normalized)
            if len(buffer) >= 16_384:
                table = pa.Table.from_pylist(buffer, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(path, schema, compression="zstd")
                writer.write_table(table)
                buffer.clear()
        if buffer:
            table = pa.Table.from_pylist(buffer, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(path, schema, compression="zstd")
            writer.write_table(table)
        if writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=schema), path, compression="zstd")
    finally:
        if writer is not None:
            writer.close()
    return accumulator.count, accumulator.hexdigest()


def build_parquet_shards(
    objects_jsonl: Path,
    master_split: MasterSplit,
    output_dir: Path,
    *,
    max_schedules: int = 500,
    max_objects: int = 250_000,
) -> dict[str, Any]:
    pa, pq = _arrow()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"shard output directory must be empty to prevent stale Parquet reuse: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_schema = _annotation_arrow_schema(pa)
    split_hash = str(master_split.metadata["split_sha256"])
    source_checksum = _ChecksumAccumulator()
    canonical_checksum = _ChecksumAccumulator()
    shard_manifest: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="shift-shard-build-") as temporary_directory, ExitStack() as resources:
        database = sqlite3.connect(Path(temporary_directory) / "spool.sqlite3")
        resources.callback(database.close)
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute(
            "CREATE TABLE objects(split TEXT, schedule_id TEXT, ordinal INTEGER, payload TEXT)"
        )
        database.execute("CREATE INDEX objects_order ON objects(split, schedule_id, ordinal)")
        for ordinal, source in enumerate(iter_jsonl(objects_jsonl)):
            schedule_id = str(source["schedule_id"])
            split_record = master_split.require(schedule_id, source.get("split"))
            item = _apply_master_metadata(source, {
                "split": split_record.split,
                "cv_fold": split_record.cv_fold,
                "master_split_sha256": split_hash,
            }, schedule_id=schedule_id)
            source_checksum.update(source)
            canonical_checksum.update(item)
            database.execute(
                "INSERT INTO objects VALUES(?,?,?,?)",
                (split_record.split, schedule_id, ordinal, json.dumps(item, ensure_ascii=False)),
            )
            if ordinal and ordinal % 25_000 == 0:
                database.commit()
        database.commit()

        image_index_schema = pa.schema([
            pa.field("image_path", pa.string(), nullable=False),
            pa.field("schedule_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("cv_fold", pa.int64(), nullable=False),
            pa.field("source_domain", pa.string(), nullable=False),
            pa.field("shard", pa.string(), nullable=False),
            pa.field("row_group", pa.int64(), nullable=False),
            pa.field("offset", pa.int64(), nullable=False),
            pa.field("object_count", pa.int64(), nullable=False),
            pa.field("master_split_sha256", pa.string(), nullable=False),
            pa.field("image_width", pa.int64(), nullable=True),
            pa.field("image_height", pa.int64(), nullable=True),
        ])
        recognition_index_schema = pa.schema([
            pa.field("image_path", pa.string(), nullable=False),
            pa.field("schedule_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("cv_fold", pa.int64(), nullable=False),
            pa.field("source_domain", pa.string(), nullable=False),
            pa.field("shard", pa.string(), nullable=False),
            pa.field("row_group", pa.int64(), nullable=False),
            pa.field("row_group_object_index", pa.int64(), nullable=False),
            pa.field("image_object_index", pa.int64(), nullable=False),
            pa.field("object_type", pa.string(), nullable=False),
            pa.field("canonical_code", pa.string(), nullable=True),
            pa.field("width_bucket", pa.int64(), nullable=False),
            pa.field("master_split_sha256", pa.string(), nullable=False),
        ])
        image_index_rows_path = Path(temporary_directory) / "image-index.jsonl"
        recognition_index_rows_path = Path(temporary_directory) / "recognition-index.jsonl"
        with image_index_rows_path.open("w", encoding="utf-8") as image_stream, recognition_index_rows_path.open(
            "w", encoding="utf-8",
        ) as recognition_stream:
            for split in ("train", "validation", "test", "ood_layout"):
                schedule_cursor = database.execute(
                    "SELECT schedule_id, COUNT(*) FROM objects WHERE split=? "
                    "GROUP BY schedule_id ORDER BY schedule_id", (split,),
                )
                shard_number = 0
                writer = None
                shard_schedule_count = shard_object_count = row_group = row_offset = 0

                def close_shard():
                    nonlocal writer, shard_schedule_count, shard_object_count
                    if writer is None:
                        return
                    writer.close()
                    shard_manifest.append({
                        "split": split, "shard": f"{split}-{shard_number:04d}.parquet",
                        "schedule_count": shard_schedule_count,
                        "object_count": shard_object_count,
                    })
                    writer = None

                for schedule_id, schedule_object_count in schedule_cursor:
                    if writer is None or shard_schedule_count >= max_schedules or (
                        shard_schedule_count and shard_object_count + schedule_object_count > max_objects
                    ):
                        close_shard()
                        shard_number += 1
                        shard_schedule_count = shard_object_count = row_group = row_offset = 0
                        shard_name = f"{split}-{shard_number:04d}.parquet"
                        writer = pq.ParquetWriter(output_dir / shard_name, annotation_schema, compression="zstd")
                    raw_records = [json.loads(row[0]) for row in database.execute(
                        "SELECT payload FROM objects WHERE split=? AND schedule_id=? ORDER BY ordinal",
                        (split, schedule_id),
                    )]
                    records = [_arrow_row(item, annotation_schema) for item in raw_records]
                    writer.write_table(pa.Table.from_pylist(records, schema=annotation_schema), row_group_size=len(records))
                    image_groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
                    for object_index, record in enumerate(raw_records):
                        image_groups[str(record["image_path"])].append((object_index, record))
                    for image_path, indexed_records in sorted(image_groups.items()):
                        domains = {str(record.get("source_domain") or "synthetic") for _, record in indexed_records}
                        widths = {record.get("image_width") for _, record in indexed_records if record.get("image_width") is not None}
                        heights = {record.get("image_height") for _, record in indexed_records if record.get("image_height") is not None}
                        if len(domains) != 1 or len(widths) > 1 or len(heights) > 1:
                            raise ValueError(f"inconsistent image metadata: {image_path}")
                        first = indexed_records[0][1]
                        image_index = {
                            "image_path": image_path, "schedule_id": schedule_id, "split": split,
                            "cv_fold": int(first["cv_fold"]), "source_domain": next(iter(domains)),
                            "shard": shard_name, "row_group": row_group, "offset": row_offset,
                            "object_count": len(indexed_records), "master_split_sha256": split_hash,
                            "image_width": next(iter(widths)) if widths else None,
                            "image_height": next(iter(heights)) if heights else None,
                        }
                        image_stream.write(json.dumps(image_index, ensure_ascii=False) + "\n")
                        for image_object_index, (object_index, record) in enumerate(indexed_records):
                            if record.get("object_type") not in {"shift_code", "name"}:
                                continue
                            recognition_stream.write(json.dumps({
                                "image_path": image_path, "schedule_id": schedule_id, "split": split,
                                "cv_fold": int(record["cv_fold"]),
                                "source_domain": str(record.get("source_domain") or "synthetic"),
                                "shard": shard_name, "row_group": row_group,
                                "row_group_object_index": object_index,
                                "image_object_index": image_object_index,
                                "object_type": str(record["object_type"]),
                                "canonical_code": record.get("canonical_code"),
                                "width_bucket": recognition_width_bucket(record),
                                "master_split_sha256": split_hash,
                            }, ensure_ascii=False) + "\n")
                    shard_schedule_count += 1
                    shard_object_count += len(records)
                    row_offset += len(records)
                    row_group += 1
                close_shard()

        def jsonl_rows(path):
            return iter_jsonl(path)

        image_count, image_checksum = _write_index_parquet(
            pa, pq, output_dir / "image_index.parquet", jsonl_rows(image_index_rows_path),
            image_index_schema,
        )
        recognition_count, recognition_checksum = _write_index_parquet(
            pa, pq, output_dir / "recognition_index.parquet", jsonl_rows(recognition_index_rows_path),
            recognition_index_schema,
        )
    result = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "checksum_schema_version": CHECKSUM_SCHEMA_VERSION,
        "checksum_fields": list(COMPATIBILITY_FIELDS_V2),
        "source_checksum_scope": "source_annotations_before_master_metadata_enrichment",
        "canonical_checksum_scope": "shard_annotations_after_master_metadata_enrichment",
        "canonical_metadata_fields": list(_CANONICAL_MASTER_FIELDS),
        "master_split_sha256": master_split.metadata["split_sha256"],
        "source_record_count": source_checksum.count,
        "source_core_checksum": source_checksum.hexdigest(),
        "canonical_core_checksum": canonical_checksum.hexdigest(),
        "index_checksum_schema": _INDEX_CHECKSUM_SCHEMA,
        "index_record_count": image_count,
        "image_index_record_count": image_count,
        "image_index_checksum": image_checksum,
        "recognition_index_record_count": recognition_count,
        "recognition_index_checksum": recognition_checksum,
        "recognition_index_schema_version": "recognition_index_v1",
        "recognition_index_object_offsets": {
            "row_group_object_index": "zero_based_raw_row_group",
            "image_object_index": "zero_based_load_image_result",
        },
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
        manifest, checksum_schema, _checksum_fields = _manifest_checksum_contract(shard_dir)
        self.checksum_schema = checksum_schema
        self.strict_index_contract = manifest.get("image_index_checksum") is not None
        declared_master_hash = manifest.get("master_split_sha256")
        supplied_master_hash = master_split.metadata.get("split_sha256")
        if declared_master_hash not in (None, "", supplied_master_hash):
            raise ValueError("shard manifest master split hash mismatch")
        _declared_shard_paths(shard_dir, manifest)
        manifest_shards = {str(item["shard"]): item for item in manifest.get("shards", [])}
        if manifest.get("image_index_record_count") is not None:
            index_path = shard_dir / "image_index.parquet"
            if not index_path.exists():
                raise FileNotFoundError(f"image index is missing: {index_path}")
        index = pq.read_table(shard_dir / "image_index.parquet").to_pylist()
        if manifest.get("image_index_record_count") is not None:
            if len(index) != int(manifest["image_index_record_count"]):
                raise ValueError("image index record count disagrees with shard manifest")
            if _index_checksum(index) != str(manifest.get("image_index_checksum")):
                raise ValueError("image index checksum disagrees with shard manifest")
        for row in index:
            shard_name = str(row["shard"])
            if manifest_shards and shard_name not in manifest_shards:
                raise ValueError(f"image index references undeclared shard: {shard_name}")
            if not (shard_dir / shard_name).exists():
                raise FileNotFoundError(f"image index shard is missing: {shard_name}")
        self.index = {str(row["image_path"]): row for row in index}
        if len(self.index) != len(index):
            raise ValueError("image index contains duplicate image_path values")
        self.cache_size = max(1, cache_size)
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def _file(self, shard: str):
        if shard not in self.cache:
            self.cache[shard] = self._pq.ParquetFile(self.shard_dir / shard)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        return self.cache[shard]

    def close(self) -> None:
        """Release Parquet handles promptly (required for Windows cleanup)."""

        for parquet_file in self.cache.values():
            close = getattr(parquet_file, "close", None)
            if close is not None:
                close()
        self.cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

    def __del__(self):  # pragma: no cover - best-effort interpreter cleanup
        try:
            self.close()
        except Exception:
            pass

    def load_indexed_image(
        self, image_path: str, *, purpose: str,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Return verified image rows paired with their raw row-group index."""

        if image_path not in self.index:
            raise KeyError(image_path)
        index = self.index[image_path]
        schedule_id = str(index["schedule_id"])
        split = str(index["split"])
        split_record = self.master_split.authorize(schedule_id, purpose, split)
        expected_hash = str(self.master_split.metadata.get("split_sha256") or "")
        if index.get("master_split_sha256") not in (None, "", expected_hash):
            raise ValueError(f"image index master split hash mismatch: {image_path}")
        if index.get("cv_fold") is not None and int(index["cv_fold"]) != split_record.cv_fold:
            raise ValueError(f"image index CV fold mismatch: {image_path}")
        parquet_file = self._file(str(index["shard"]))
        row_group = int(index["row_group"])
        if row_group < 0 or row_group >= parquet_file.num_row_groups:
            raise ValueError(f"image index row group is out of bounds: {image_path}")
        metadata = parquet_file.metadata.row_group(row_group)
        rows = parquet_file.read_row_group(row_group).to_pylist()
        offset = int(index.get("offset") or 0)
        if offset < 0:
            raise ValueError(f"image index offset is invalid: {image_path}")
        expected_offset = sum(
            parquet_file.metadata.row_group(group_index).num_rows
            for group_index in range(row_group)
        )
        if offset != expected_offset:
            raise ValueError(f"image index offset disagrees with row group: {image_path}")
        if metadata.num_rows != len(rows):
            raise ValueError(f"Parquet row group metadata disagrees with rows: {image_path}")
        output = [
            (row_group_index, _decode_json_fields(row))
            for row_group_index, row in enumerate(rows)
            if str(row["image_path"]) == image_path
        ]
        if len(output) != int(index["object_count"]):
            raise ValueError(f"image index object count mismatch: {image_path}")
        expected_domain = str(index.get("source_domain") or "synthetic")
        for _row_group_index, row in output:
            if str(row.get("schedule_id")) != schedule_id or str(row.get("split")) != split:
                raise ValueError(f"image index row ownership mismatch: {image_path}")
            if self.strict_index_contract and self.checksum_schema == CHECKSUM_SCHEMA_VERSION and str(
                row.get("master_split_sha256") or ""
            ) != expected_hash:
                raise ValueError(f"image row master split hash mismatch: {image_path}")
            if self.strict_index_contract and self.checksum_schema == CHECKSUM_SCHEMA_VERSION and int(
                row.get("cv_fold")
            ) != split_record.cv_fold:
                raise ValueError(f"image row CV fold mismatch: {image_path}")
            if str(row.get("source_domain") or "synthetic") != expected_domain:
                raise ValueError(f"image row source domain mismatch: {image_path}")
            if index.get("image_width") is not None and int(row.get("image_width")) != int(index["image_width"]):
                raise ValueError(f"image row width mismatch: {image_path}")
            if index.get("image_height") is not None and int(row.get("image_height")) != int(index["image_height"]):
                raise ValueError(f"image row height mismatch: {image_path}")
        return output

    def load_image(self, image_path: str, *, purpose: str) -> list[dict[str, Any]]:
        return [row for _row_group_index, row in self.load_indexed_image(
            image_path, purpose=purpose,
        )]
