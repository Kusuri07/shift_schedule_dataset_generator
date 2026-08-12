import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from shift_ocr.models import decode_table_candidates
from shift_ocr.master_split import MasterSplit, SplitRecord
from shift_ocr.shards import (
    CHECKSUM_SCHEMA_VERSION,
    LEGACY_COMPATIBILITY_FIELDS_V2,
    ParquetImageStore,
    SHARD_SCHEMA_VERSION,
    build_parquet_shards,
    core_checksum,
    iter_verified_index_rows,
    recognition_width_bucket,
    verify_parquet_index,
    verify_compatible_annotations,
)
from shift_ocr.validation import validate_table


def _annotation_record():
    return {
        "schedule_id": "schedule_0001",
        "split": "train",
        "cv_fold": 1,
        "master_split_sha256": "abc123",
        "template_id": "template_1",
        "layout_family": "standard",
        "object_type": "shift_code",
        "display_text": "D",
        "canonical_code": "D",
        "row_id": "row_01",
        "row_index": 1,
        "day": 2,
        "bbox_px": [1, 2, 11, 12],
        "cell_polygon": [[1, 2], [11, 2], [11, 12], [1, 12]],
        "text_polygon": [[3, 4], [8, 4], [8, 10], [3, 10]],
        "text_polygon_source": "glyph_mask",
        "text_polygon_validation_max_error_px": 0.75,
        "visibility": 1.0,
        "ignore": False,
        "image_path": "images/schedule_0001.png",
        "image_width": 100,
        "image_height": 80,
    }


class ShardCompatibilityTests(unittest.TestCase):
    def test_recognition_width_bucket_matches_rectification_after_rotation(self):
        import math

        import numpy as np

        from shift_ocr.augmentation import rectify_cell

        image = np.full((800, 800, 3), 255, np.uint8)
        for width, height, angle_degrees, shear in (
            (318.0, 96.0, 17.0, 0.0),
            (640.0, 96.0, -11.0, 0.35),
            (700.0, 48.0, 31.0, -0.25),
        ):
            angle = math.radians(angle_degrees)
            cosine, sine = math.cos(angle), math.sin(angle)
            base = [(-width / 2, -height / 2), (width / 2, -height / 2),
                    (width / 2, height / 2), (-width / 2, height / 2)]
            polygon = [
                [400 + (x + shear * y) * cosine - y * sine,
                 400 + (x + shear * y) * sine + y * cosine]
                for x, y in base
            ]
            _crop, expected, _content_width, _matrix = rectify_cell(image, polygon)
            self.assertEqual(
                recognition_width_bucket({"cell_polygon": polygon}), expected,
            )

        # Unequal opposing edges exercise the same perspective max-edge rule
        # used by rectification; the legacy x/y-component shortcut disagreed.
        perspective_polygon = [[60, 60], [160, 160], [155, 205], [35, 105]]
        _crop, expected, _content_width, _matrix = rectify_cell(
            image, perspective_polygon,
        )
        self.assertEqual(
            recognition_width_bucket({"cell_polygon": perspective_polygon}), expected,
        )

        # Training samples are grouped by the source annotation's index
        # bucket.  A jittered/predicted quad may cross a boundary, but forcing
        # that indexed bucket must keep the returned tensor width homogeneous.
        boundary_source = [[100, 100], [419, 100], [419, 196], [100, 196]]
        indexed_bucket = recognition_width_bucket({"cell_polygon": boundary_source})
        widened_quad = [[100, 100], [500, 100], [500, 196], [100, 196]]
        crop, returned_bucket, content_width, _matrix = rectify_cell(
            image, widened_quad, target_width=indexed_bucket,
        )
        self.assertEqual(indexed_bucket, 160)
        self.assertEqual(returned_bucket, indexed_bucket)
        self.assertEqual(crop.shape[1], indexed_bucket)
        self.assertLessEqual(content_width, indexed_bucket)

    def test_v2_checksum_normalizes_csv_types_and_covers_validation_fields(self):
        record = _annotation_record()
        csv_record = dict(record)
        for field in ("bbox_px", "cell_polygon", "text_polygon"):
            csv_record[field] = json.dumps(csv_record[field], separators=(",", ":"))
        for field in (
            "cv_fold", "row_index", "day", "text_polygon_validation_max_error_px",
            "visibility", "ignore", "image_width", "image_height",
        ):
            csv_record[field] = str(csv_record[field])

        self.assertEqual(core_checksum([record]), core_checksum([csv_record]))
        changed = dict(record, text_polygon_validation_max_error_px=1.25)
        self.assertNotEqual(core_checksum([record]), core_checksum([changed]))
        changed = dict(record, visibility=0.5)
        self.assertNotEqual(core_checksum([record]), core_checksum([changed]))
        changed = dict(record, ignore=True)
        self.assertNotEqual(core_checksum([record]), core_checksum([changed]))
        changed = dict(record, source_domain="real")
        self.assertNotEqual(core_checksum([record]), core_checksum([changed]))
        changed = dict(record, registration_high_confidence=False)
        self.assertNotEqual(core_checksum([record]), core_checksum([changed]))
        self.assertEqual(
            core_checksum([record]), core_checksum([dict(record, source_domain="synthetic")]),
        )
        self.assertNotEqual(
            core_checksum([record]),
            core_checksum([dict(record, registration_high_confidence=True)]),
        )
        self.assertEqual(SHARD_SCHEMA_VERSION, "training_shards_v2")
        self.assertEqual(CHECKSUM_SCHEMA_VERSION, "object_annotations_v2")

        # Legacy checksums remain callable so manifests produced before v2 can
        # still be audited or read without requiring a destructive rebuild.
        self.assertIsInstance(core_checksum([record], schema_version="training_shards_v1"), str)

    def test_manifest_rejects_crossed_shard_and_checksum_schemas(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for shard_schema, checksum_schema in (
                ("training_shards_v1", "object_annotations_v2"),
                ("training_shards_v2", "object_annotations_v1"),
            ):
                (root / "shards.manifest.json").write_text(json.dumps({
                    "schema_version": shard_schema,
                    "checksum_schema_version": checksum_schema,
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "schema mismatch"):
                    from shift_ocr.shards import _manifest_checksum_contract
                    _manifest_checksum_contract(root)

    def test_csv_jsonl_field_mismatch_is_rejected(self):
        record = _annotation_record()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            jsonl_path = root / "objects.jsonl"
            csv_path = root / "objects.csv"
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            csv_record = dict(record, text_polygon_validation_max_error_px=9.0)
            for field in ("bbox_px", "cell_polygon", "text_polygon"):
                csv_record[field] = json.dumps(csv_record[field], separators=(",", ":"))
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_record))
                writer.writeheader()
                writer.writerow(csv_record)
            with self.assertRaisesRegex(ValueError, "CSV and JSONL"):
                verify_compatible_annotations(jsonl_path, csv_path)

    def test_metadata_less_source_builds_and_verifies_against_enriched_v2_shards(self):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            self.skipTest(str(exc))

        record = _annotation_record()
        for field in ("split", "cv_fold", "master_split_sha256"):
            record.pop(field)
        split_record = SplitRecord(
            schedule_id=record["schedule_id"], split="train", cv_fold=2,
            template_id=record["template_id"], layout_family=record["layout_family"],
            seed=1, capture_target=0,
        )
        master = MasterSplit([split_record], {"record_count": 1, "split_sha256": "canonical-hash"})
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            jsonl_path = root / "objects.jsonl"
            csv_path = root / "objects.csv"
            shard_dir = root / "shards"
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            csv_record = dict(record)
            for field in ("bbox_px", "cell_polygon", "text_polygon"):
                csv_record[field] = json.dumps(csv_record[field], separators=(",", ":"))
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_record))
                writer.writeheader()
                writer.writerow(csv_record)

            manifest = build_parquet_shards(jsonl_path, master, shard_dir)
            result = verify_compatible_annotations(jsonl_path, csv_path, shard_dir)
            parquet_record = pq.read_table(shard_dir / "train-0001.parquet").to_pylist()[0]

            self.assertEqual(parquet_record["split"], "train")
            self.assertEqual(parquet_record["cv_fold"], 2)
            self.assertEqual(parquet_record["master_split_sha256"], "canonical-hash")
            self.assertNotEqual(manifest["source_core_checksum"], manifest["canonical_core_checksum"])
            self.assertEqual(result["canonical_jsonl_checksum"], result["parquet_checksum"])
            self.assertEqual(result["checksum_schema_version"], CHECKSUM_SCHEMA_VERSION)

    def test_v1_manifest_selects_legacy_checksum_contract(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            self.skipTest(str(exc))

        record = _annotation_record()
        record.update({
            "source_domain": "real", "registration_high_confidence": False,
            "registration_profile": "partial", "text_polygon_margin_px": 3.0,
        })
        legacy_record = {
            field: record[field]
            for field in (
                "schedule_id", "image_path", "object_type", "row_id", "row_index",
                "day", "display_text", "canonical_code", "cell_polygon", "text_polygon",
            )
        }
        for field in ("cell_polygon", "text_polygon"):
            legacy_record[field] = json.dumps(legacy_record[field], separators=(",", ":"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            jsonl_path = root / "objects.jsonl"
            csv_path = root / "objects.csv"
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            csv_record = dict(record)
            for field in ("bbox_px", "cell_polygon", "text_polygon"):
                csv_record[field] = json.dumps(csv_record[field], separators=(",", ":"))
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_record))
                writer.writeheader()
                writer.writerow(csv_record)
            pq.write_table(pa.Table.from_pylist([legacy_record]), root / "train-0001.parquet")
            legacy_checksum = core_checksum([record], schema_version="training_shards_v1")
            (root / "shards.manifest.json").write_text(json.dumps({
                "schema_version": "training_shards_v1",
                "source_record_count": 1,
                "source_core_checksum": legacy_checksum,
            }), encoding="utf-8")

            result = verify_compatible_annotations(jsonl_path, csv_path, root)
            self.assertEqual(result["checksum_schema_version"], "object_annotations_v1")
            self.assertEqual(result["jsonl_checksum"], legacy_checksum)
            self.assertEqual(result["parquet_checksum"], legacy_checksum)

    def test_initial_v2_manifest_field_contract_remains_verifiable(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            self.skipTest(str(exc))

        record = _annotation_record()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            jsonl_path = root / "objects.jsonl"
            csv_path = root / "objects.csv"
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            csv_record = dict(record)
            parquet_record = dict(record)
            for field in ("bbox_px", "cell_polygon", "text_polygon"):
                csv_record[field] = json.dumps(csv_record[field], separators=(",", ":"))
                parquet_record[field] = csv_record[field]
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_record))
                writer.writeheader()
                writer.writerow(csv_record)
            pq.write_table(pa.Table.from_pylist([parquet_record]), root / "train-0001.parquet")
            legacy_v2_checksum = core_checksum(
                [record], compatibility_fields=LEGACY_COMPATIBILITY_FIELDS_V2,
            )
            (root / "shards.manifest.json").write_text(json.dumps({
                "schema_version": "training_shards_v2",
                "checksum_schema_version": "object_annotations_v2",
                "checksum_fields": list(LEGACY_COMPATIBILITY_FIELDS_V2),
                "source_record_count": 1,
                "source_core_checksum": legacy_v2_checksum,
            }), encoding="utf-8")

            result = verify_compatible_annotations(jsonl_path, csv_path, root)
            self.assertEqual(result["checksum_schema_version"], "object_annotations_v2")
            self.assertEqual(result["checksum_fields"], list(LEGACY_COMPATIBILITY_FIELDS_V2))
            self.assertEqual(result["parquet_checksum"], legacy_v2_checksum)

    def test_v1_and_v2_readers_restore_json_geometry_as_lists(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            self.skipTest(str(exc))

        record = _annotation_record()
        record.update({
            "name_bbox_px": [1, 2, 20, 12],
            "name_cell_polygon": [[1, 2], [20, 2], [20, 12], [1, 12]],
        })
        stored = dict(record)
        geometry_fields = (
            "bbox_px", "cell_polygon", "text_polygon", "name_bbox_px", "name_cell_polygon",
        )
        for field in geometry_fields:
            stored[field] = json.dumps(stored[field], separators=(",", ":"))

        split_record = SplitRecord(
            schedule_id=record["schedule_id"], split="train", cv_fold=0,
            template_id=record["template_id"], layout_family=record["layout_family"],
            seed=1, capture_target=0,
        )
        master = MasterSplit([split_record], {"record_count": 1, "split_sha256": "test"})
        index = [{
            "image_path": record["image_path"], "schedule_id": record["schedule_id"],
            "split": "train", "shard": "train-0001.parquet", "row_group": 0,
            "offset": 0, "object_count": 1, "master_split_sha256": "test",
        }]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for schema_version in ("training_shards_v1", "training_shards_v2"):
                shard_dir = root / schema_version
                shard_dir.mkdir()
                pq.write_table(pa.Table.from_pylist([stored]), shard_dir / "train-0001.parquet")
                pq.write_table(pa.Table.from_pylist(index), shard_dir / "image_index.parquet")
                (shard_dir / "shards.manifest.json").write_text(
                    json.dumps({"schema_version": schema_version}), encoding="utf-8",
                )

                loaded = ParquetImageStore(shard_dir, master).load_image(
                    record["image_path"], purpose="train",
                )
                self.assertEqual(len(loaded), 1)
                for field in geometry_fields:
                    self.assertIsInstance(loaded[0][field], list, (schema_version, field))
                    self.assertEqual(loaded[0][field], record[field])

    def test_new_indexes_mixed_optional_schema_and_tamper_guards(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            self.skipTest(str(exc))

        synthetic = _annotation_record()
        synthetic.update({"schedule_id": "schedule_s", "image_path": "synthetic.png"})
        real = dict(_annotation_record())
        real.update({
            "schedule_id": "schedule_r", "image_path": "real.png", "source_domain": "real",
            "registration_profile": "partial", "registration_high_confidence": True,
            "text_polygon_margin_px": 4.25,
        })
        for record in (synthetic, real):
            for field in ("split", "cv_fold", "master_split_sha256"):
                record.pop(field, None)
        splits = [
            SplitRecord("schedule_s", "train", 0, "template_1", "standard", 1, 0),
            SplitRecord("schedule_r", "test", -1, "template_1", "standard", 2, 0),
        ]
        master = MasterSplit(splits, {"record_count": 2, "split_sha256": "secure-hash"})
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            objects = root / "objects.jsonl"
            csv_path = root / "objects.csv"
            objects.write_text("".join(json.dumps(item) + "\n" for item in (synthetic, real)), encoding="utf-8")
            fieldnames = list(dict.fromkeys([*synthetic, *real]))
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for record in (synthetic, real):
                    row = dict(record)
                    for field in ("bbox_px", "cell_polygon", "text_polygon"):
                        row[field] = json.dumps(row[field], separators=(",", ":"))
                    writer.writerow(row)
            shards = root / "shards"
            manifest = build_parquet_shards(objects, master, shards, max_schedules=1)
            verified = verify_compatible_annotations(objects, csv_path, shards)
            self.assertEqual(verified["parquet_count"], 2)
            self.assertEqual(manifest["image_index_record_count"], 2)
            self.assertEqual(manifest["recognition_index_record_count"], 2)
            recognition = pq.read_table(shards / "recognition_index.parquet").to_pylist()
            self.assertEqual({row["source_domain"] for row in recognition}, {"synthetic", "real"})
            self.assertTrue(all(row["image_object_index"] == 0 for row in recognition))
            real_stored = pq.read_table(shards / "test-0001.parquet").to_pylist()[0]
            self.assertEqual(real_stored["registration_profile"], "partial")
            self.assertTrue(real_stored["registration_high_confidence"])
            synthetic_stored = pq.read_table(shards / "train-0001.parquet").to_pylist()[0]
            self.assertIsNone(synthetic_stored["registration_profile"])
            self.assertEqual(verify_parquet_index(shards, "recognition")["record_count"], 2)
            self.assertEqual(len(list(iter_verified_index_rows(shards, "recognition"))), 2)

            # Even a manifest-authenticated index cannot redirect an image to
            # rows owned by another schedule.
            test_shard = shards / "test-0001.parquet"
            original_table = pq.read_table(test_shard)
            altered_rows = original_table.to_pylist()
            altered_rows[0]["schedule_id"] = "schedule_s"
            pq.write_table(pa.Table.from_pylist(altered_rows, schema=original_table.schema), test_shard)
            with ParquetImageStore(shards, master) as store:
                with self.assertRaisesRegex(ValueError, "ownership mismatch"):
                    store.load_image("real.png", purpose="test")
            pq.write_table(original_table, test_shard)

            # Index split tampering cannot authorize Test rows as Train: even
            # if an attacker also rewrites the manifest checksum, actual rows
            # are checked against the authorized index ownership.
            index_path = shards / "image_index.parquet"
            index_rows = pq.read_table(index_path).to_pylist()
            test_row = next(row for row in index_rows if row["schedule_id"] == "schedule_r")
            test_row["split"] = "train"
            pq.write_table(pa.Table.from_pylist(index_rows), index_path)
            manifest_path = shards / "shards.manifest.json"
            tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            from shift_ocr.shards import _index_checksum
            tampered_manifest["image_index_checksum"] = _index_checksum(index_rows)
            manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                ParquetImageStore(shards, master).load_image("real.png", purpose="train")

            index_path.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_compatible_annotations(objects, csv_path, shards)
            recognition_path = shards / "recognition_index.parquet"
            recognition_path.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_parquet_index(shards, "recognition")

    def test_manifest_rejects_stale_or_undeclared_shard(self):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            self.skipTest(str(exc))
        record = _annotation_record()
        for field in ("split", "cv_fold", "master_split_sha256"):
            record.pop(field, None)
        split = SplitRecord(record["schedule_id"], "train", 1, "template_1", "standard", 1, 0)
        master = MasterSplit([split], {"record_count": 1, "split_sha256": "hash"})
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            objects = root / "objects.jsonl"
            objects.write_text(json.dumps(record) + "\n", encoding="utf-8")
            build_parquet_shards(objects, master, root / "shards")
            manifest_path = root / "shards" / "shards.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["shards"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shard set disagrees"):
                ParquetImageStore(root / "shards", master)

    def test_streaming_build_and_verify_bound_rss_for_300_schedules(self):
        try:
            import psutil  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            script = root / "scale_fixture.py"
            script.write_text(r'''
import csv, json, os, psutil, sys
from pathlib import Path
from shift_ocr.master_split import MasterSplit, SplitRecord
from shift_ocr.shards import build_parquet_shards, verify_compatible_annotations
root=Path(sys.argv[1]); records=[]; splits=[]
for schedule in range(300):
    sid=f"schedule_{schedule:04d}"; splits.append(SplitRecord(sid,"train",schedule%3,"t","l",schedule,0))
    for obj in range(80):
        records.append({"schedule_id":sid,"template_id":"t","layout_family":"l","object_type":"shift_code","display_text":"D","canonical_code":"D","row_id":f"r{obj}","row_index":obj,"day":1,"bbox_px":[1,1,10,10],"cell_polygon":[[1,1],[10,1],[10,10],[1,10]],"text_polygon":[[2,2],[8,2],[8,8],[2,8]],"text_polygon_source":"glyph_mask","text_polygon_validation_max_error_px":0.0,"visibility":1.0,"ignore":False,"image_path":f"{sid}.png","image_width":100,"image_height":80})
jsonl=root/"objects.jsonl"; csvp=root/"objects.csv"
with jsonl.open("w",encoding="utf-8") as s:
    for r in records: s.write(json.dumps(r,separators=(",",":"))+"\n")
fields=list(records[0])
with csvp.open("w",encoding="utf-8-sig",newline="") as s:
    w=csv.DictWriter(s,fieldnames=fields);w.writeheader()
    for r in records:
        x=dict(r)
        for f in ("bbox_px","cell_polygon","text_polygon"):x[f]=json.dumps(x[f],separators=(",",":"))
        w.writerow(x)
del records
p=psutil.Process(os.getpid()); baseline=p.memory_info().rss
master=MasterSplit(splits,{"record_count":300,"split_sha256":"scale-hash"})
manifest=build_parquet_shards(jsonl,master,root/"shards",max_schedules=25,max_objects=3000)
result=verify_compatible_annotations(jsonl,csvp,root/"shards")
peak=max(baseline,p.memory_info().rss)
print(json.dumps({"delta":peak-baseline,"count":result["parquet_count"],"index":manifest["image_index_record_count"]}))
''', encoding="utf-8")
            repository_root = str(Path(__file__).parents[1])
            environment = dict(
                os.environ, PYTHONNOUSERSITE="1",
                PYTHONPATH=repository_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
            )
            completed = subprocess.run(
                [sys.executable, str(script), str(root)], cwd=Path(__file__).parents[1],
                env=environment, capture_output=True, text=True, timeout=180,
            )
            if completed.returncode:
                self.fail(f"scaling subprocess failed:\n{completed.stdout}\n{completed.stderr}")
            report = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(report["count"], 24_000)
            self.assertEqual(report["index"], 300)
            # The source annotations are about 25 MB as Python dictionaries;
            # the streaming implementation stays well below a full second
            # materialization of JSONL+CSV+Parquet.
            self.assertLess(report["delta"], 160_000_000)


def _table_outputs(*, collapse_rows=False, collapse_columns=False, bad_corners=False):
    points = [(1, 1), (1, 5), (5, 1), (5, 5)]
    heatmap = torch.full((1, 1, 8, 8), -20.0)
    corners = torch.zeros((1, 8, 8, 8))
    rows = torch.zeros((1, 2, 8, 8))
    columns = torch.zeros((1, 2, 8, 8))
    for y, x in points:
        heatmap[0, 0, y, x] = 20.0
        if not bad_corners:
            corners[0, :, y, x] = torch.tensor([-1, -1, 1, -1, 1, 1, -1, 1])
        if not collapse_rows:
            rows[0, 0, y, x] = 0.0 if y == 1 else 2.0
        if not collapse_columns:
            columns[0, 1, y, x] = 0.0 if x == 1 else 2.0
    return {
        "cell_heatmap": heatmap,
        "corner_offsets": corners,
        "row_embedding": rows,
        "column_embedding": columns,
    }


def _groups_by_point(candidates, name):
    groups = getattr(candidates, name).tolist()
    return {tuple(point): group for point, group in zip(candidates.points.tolist(), groups)}


class FixedTableModel(torch.nn.Module):
    def __init__(self, *, collapse_rows=False, collapse_columns=False, bad_corners=False):
        super().__init__()
        self.outputs = _table_outputs(
            collapse_rows=collapse_rows, collapse_columns=collapse_columns,
            bad_corners=bad_corners,
        )
        self.top_k = 1500

    def forward(self, image):
        batch = image.shape[0]
        return {name: value.expand(batch, -1, -1, -1) for name, value in self.outputs.items()}


def _table_batch():
    valid = torch.zeros((1, 1, 8, 8))
    corners = torch.zeros((1, 8, 8, 8))
    relation = torch.zeros((1, 2, 8, 8))
    for y, x in ((1, 1), (1, 5), (5, 1), (5, 5)):
        valid[0, 0, y, x] = 1.0
        corners[0, :, y, x] = torch.tensor([-1, -1, 1, -1, 1, 1, -1, 1])
        relation[0, 0, y, x] = 1 if y == 1 else 2
        relation[0, 1, y, x] = 1 if x == 1 else 2
    return {
        "image": torch.zeros((1, 3, 32, 32)),
        "cell_heatmap_target": torch.zeros((1, 1, 8, 8)),
        "corner_target": corners,
        "corner_valid": valid,
        "relation_target": relation,
    }


class TableRelationDecodeTests(unittest.TestCase):
    def test_relation_clustering_rejects_non_finite_values_and_thresholds(self):
        outputs = _table_outputs()
        outputs["row_embedding"][0, 0, 1, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            decode_table_candidates(outputs)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            decode_table_candidates(_table_outputs(), relation_distance_threshold=0.0)

    def test_decode_groups_come_from_both_relation_heads(self):
        correct = decode_table_candidates(_table_outputs(), top_k=1500)[0]
        collapsed_rows = decode_table_candidates(
            _table_outputs(collapse_rows=True), top_k=1500,
        )[0]
        collapsed_columns = decode_table_candidates(
            _table_outputs(collapse_columns=True), top_k=1500,
        )[0]

        correct_rows = _groups_by_point(correct, "row_groups")
        correct_columns = _groups_by_point(correct, "column_groups")
        collapsed_row_groups = _groups_by_point(collapsed_rows, "row_groups")
        collapsed_column_groups = _groups_by_point(collapsed_columns, "column_groups")
        self.assertEqual(len(correct.scores), 4)
        self.assertEqual(correct_rows[(1, 1)], correct_rows[(1, 5)])
        self.assertNotEqual(correct_rows[(1, 1)], correct_rows[(5, 1)])
        self.assertEqual(correct_columns[(1, 1)], correct_columns[(5, 1)])
        self.assertNotEqual(correct_columns[(1, 1)], correct_columns[(1, 5)])
        self.assertEqual(len(set(collapsed_row_groups.values())), 1)
        self.assertEqual(len(set(collapsed_column_groups.values())), 1)

    def test_validation_relation_metrics_change_with_each_embedding_head(self):
        batch = _table_batch()
        correct = validate_table(FixedTableModel(), [batch], torch.device("cpu"))
        bad_rows = validate_table(
            FixedTableModel(collapse_rows=True), [batch], torch.device("cpu"),
        )
        bad_columns = validate_table(
            FixedTableModel(collapse_columns=True), [batch], torch.device("cpu"),
        )
        bad_corners = validate_table(
            FixedTableModel(bad_corners=True), [batch], torch.device("cpu"),
        )

        self.assertEqual(correct["cell_polygon_f1"], 1.0)
        self.assertEqual(correct["row_level_accuracy"], 1.0)
        self.assertEqual(correct["column_level_accuracy"], 1.0)
        self.assertEqual(correct["table_relation_accuracy"], 1.0)
        self.assertEqual(bad_rows["row_level_accuracy"], 0.0)
        self.assertEqual(bad_rows["column_level_accuracy"], 1.0)
        self.assertEqual(bad_rows["table_relation_accuracy"], 0.0)
        self.assertEqual(bad_columns["row_level_accuracy"], 1.0)
        self.assertEqual(bad_columns["column_level_accuracy"], 0.0)
        self.assertEqual(bad_columns["table_relation_accuracy"], 0.0)
        self.assertEqual(bad_corners["cell_polygon_f1"], 0.0)
        self.assertEqual(bad_corners["row_level_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
