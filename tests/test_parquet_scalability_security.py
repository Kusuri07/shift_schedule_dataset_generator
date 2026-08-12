"""Scale and integrity regressions for the lazy Parquet training path.

The fixtures deliberately carry a large, non-training payload.  A streaming
implementation only holds one schedule's payload at a time, whereas turning
the complete JSONL/CSV corpus into Python dictionaries makes the difference
visible to ``tracemalloc`` without needing a production-sized data set.
"""

from __future__ import annotations

import csv
import gc
import json
import pickle
import shutil
import tempfile
import tracemalloc
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from shift_ocr.datasets import (
    LazyParquetDenseScheduleDataset,
    LazyParquetRecognitionCropDataset,
    ParquetRecognitionBatchSampler,
    load_parquet_image_entries,
    recognition_collate,
)
from shift_ocr.master_split import MasterSplit, SplitRecord, sha256_value
from shift_ocr.shards import (
    ParquetImageStore,
    build_parquet_shards,
    verify_compatible_annotations,
)
import shift_ocr.shards as shards_module


_CSV_FIELDS = (
    "schedule_id", "split", "template_id", "layout_family", "object_type",
    "display_text", "canonical_code", "row_id", "row_index", "day",
    "bbox_px", "cell_polygon", "text_polygon", "text_polygon_source",
    "text_polygon_validation_max_error_px", "visibility", "ignore",
    "source_domain", "registration_profile", "registration_high_confidence",
    "text_polygon_margin_px", "image_path", "image_width", "image_height",
    "excel_cell", "unused_payload",
)
_JSON_FIELDS = {"bbox_px", "cell_polygon", "text_polygon"}


def _master_split(
    schedule_count: int, *, test_count: int = 0, prefix: str = "schedule",
) -> MasterSplit:
    records = []
    train_count = schedule_count - test_count
    for index in range(schedule_count):
        split = "train" if index < train_count else "test"
        records.append(SplitRecord(
            schedule_id=f"{prefix}_{index:04d}",
            split=split,
            cv_fold=index % 3 if split == "train" else -1,
            template_id="template_a",
            layout_family="grid_a",
            seed=index,
            capture_target=0,
            month=(index % 12) + 1,
            people_count=20 + index % 8,
        ))
    metadata = {
        "record_count": len(records),
        "split_sha256": sha256_value([asdict(item) for item in records]),
    }
    return MasterSplit(records, metadata)


def _annotation(
    schedule_index: int,
    object_index: int,
    *,
    prefix: str,
    split: str,
    payload_bytes: int,
) -> dict[str, Any]:
    schedule_id = f"{prefix}_{schedule_index:04d}"
    left = 4 + object_index * 20
    right = left + 16
    cell = [[left, 8], [right, 8], [right, 32], [left, 32]]
    text = [[left + 2, 12], [right - 2, 12], [right - 2, 28], [left + 2, 28]]
    item: dict[str, Any] = {
        "schedule_id": schedule_id,
        "split": split,
        "template_id": "template_a",
        "layout_family": "grid_a",
        "object_type": "shift_code",
        "display_text": "D",
        "canonical_code": "D",
        "row_id": f"row_{object_index}",
        "row_index": object_index,
        "day": object_index + 1,
        "bbox_px": [left, 8, right, 32],
        "cell_polygon": cell,
        "text_polygon": text,
        "text_polygon_source": "glyph_mask",
        "text_polygon_validation_max_error_px": 0.25,
        "visibility": 1.0,
        "ignore": False,
        "image_path": f"images/{schedule_id}.png",
        "image_width": 96,
        "image_height": 48,
        "excel_cell": f"{chr(66 + object_index)}4",
        # Not part of the training compatibility schema.  It makes accidental
        # whole-corpus Python materialization measurable and should not enter a
        # lazy dataset's worker payload.
        "unused_payload": "x" * payload_bytes + f":{schedule_index}:{object_index}",
    }
    if schedule_index % 2:
        item.update({
            "source_domain": "real",
            "registration_profile": "registered_phone_v1",
            "registration_high_confidence": True,
            "text_polygon_margin_px": 1.25,
        })
    # Even-numbered rows intentionally omit all real-photo optional fields.
    return item


def _write_fixture(
    root: Path,
    schedule_count: int,
    *,
    objects_per_schedule: int = 1,
    payload_bytes: int = 0,
    test_count: int = 0,
    prefix: str = "schedule",
    write_images: bool = False,
) -> tuple[Path, Path, MasterSplit]:
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "objects.jsonl"
    csv_path = root / "objects.csv"
    train_count = schedule_count - test_count
    template_image: Path | None = None
    if write_images:
        from PIL import Image

        image_dir = root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        template_image = root / "template.png"
        Image.new("RGB", (96, 48), "white").save(template_image)

    with jsonl_path.open("w", encoding="utf-8") as jsonl_stream, csv_path.open(
        "w", encoding="utf-8-sig", newline="",
    ) as csv_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for schedule_index in range(schedule_count):
            split = "train" if schedule_index < train_count else "test"
            if template_image is not None:
                target = root / "images" / f"{prefix}_{schedule_index:04d}.png"
                shutil.copyfile(template_image, target)
            for object_index in range(objects_per_schedule):
                item = _annotation(
                    schedule_index,
                    object_index,
                    prefix=prefix,
                    split=split,
                    payload_bytes=payload_bytes,
                )
                jsonl_stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                csv_item = dict(item)
                for field in _JSON_FIELDS:
                    csv_item[field] = json.dumps(csv_item[field], separators=(",", ":"))
                writer.writerow(csv_item)
    if template_image is not None:
        template_image.unlink()
    return jsonl_path, csv_path, _master_split(
        schedule_count, test_count=test_count, prefix=prefix,
    )


def _measure_peak(call: Callable[[], Any]) -> tuple[Any, int]:
    gc.collect()
    tracemalloc.start()
    try:
        result = call()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def _materialization_peak(path: Path) -> int:
    def materialize():
        with path.open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        # Keep the list alive until tracemalloc has sampled the peak.
        return records

    records, peak = _measure_peak(materialize)
    del records
    gc.collect()
    return peak


def _dataset_spawn_payload_size(dataset: Any) -> int:
    """Measure the ordinary worker payload, excluding the shared epoch cell.

    ``multiprocessing.RawValue`` is reduced only while a process is spawning,
    so a plain ``pickle.dumps(dataset)`` is intentionally illegal.  The rest
    of ``__getstate__`` is exactly the annotation/index payload sent to each
    worker; the workers=2 test below exercises the actual spawn reduction.
    """

    state = dataset.__getstate__() if hasattr(dataset, "__getstate__") else dict(dataset.__dict__)
    state = dict(state)
    state.pop("_shared_absolute_epoch", None)
    return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


def _rewrite_image_index(shard_dir: Path, mutate: Callable[[list[dict[str, Any]]], None]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    index_path = shard_dir / "image_index.parquet"
    table = pq.read_table(index_path)
    rows = table.to_pylist()
    mutate(rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), index_path, compression="zstd")
    manifest_path = shard_dir / "shards.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Refresh the cryptographic envelope so the tests below reach semantic
    # ownership/hash/count checks rather than stopping at a checksum mismatch.
    manifest["image_index_record_count"] = len(rows)
    manifest["index_record_count"] = len(rows)
    manifest["image_index_checksum"] = shards_module._index_checksum(rows)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _close_parquet_store(store: ParquetImageStore) -> None:
    """Release Arrow file handles eagerly so Windows can remove temp shards."""

    for parquet_file in store.cache.values():
        parquet_file.close(force=True)
    store.cache.clear()


class ParquetScalabilityAndSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pyarrow  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))

    def test_index_manifest_and_cross_split_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            jsonl_path, csv_path, master = _write_fixture(
                source, 8, test_count=2, prefix="secure", payload_bytes=32,
            )
            base = root / "base"
            build_parquet_shards(jsonl_path, master, base, max_schedules=2)
            verify_compatible_annotations(jsonl_path, csv_path, base)

            missing = root / "missing-index"
            shutil.copytree(base, missing)
            (missing / "image_index.parquet").unlink()
            with self.assertRaises((FileNotFoundError, ValueError)):
                ParquetImageStore(missing, master)

            bad_manifest_hash = root / "bad-manifest-hash"
            shutil.copytree(base, bad_manifest_hash)
            manifest_path = bad_manifest_hash / "shards.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["master_split_sha256"] = "forged-master-hash"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            with self.assertRaises(ValueError):
                ParquetImageStore(bad_manifest_hash, master)

            bad_index_hash = root / "bad-index-hash"
            shutil.copytree(base, bad_index_hash)
            _rewrite_image_index(
                bad_index_hash,
                lambda rows: rows[0].__setitem__("master_split_sha256", "forged-master-hash"),
            )
            store = ParquetImageStore(bad_index_hash, master)
            try:
                with self.assertRaises(ValueError):
                    store.load_image(str(next(iter(store.index))), purpose="train")
            finally:
                _close_parquet_store(store)

            bad_count = root / "bad-object-count"
            shutil.copytree(base, bad_count)
            _rewrite_image_index(
                bad_count,
                lambda rows: rows[0].__setitem__("object_count", int(rows[0]["object_count"]) + 1),
            )
            store = ParquetImageStore(bad_count, master)
            try:
                with self.assertRaises(ValueError):
                    store.load_image(str(next(iter(store.index))), purpose="train")
            finally:
                _close_parquet_store(store)

            leakage = root / "cross-split-leakage"
            shutil.copytree(base, leakage)

            def forge_test_entry_as_train(rows: list[dict[str, Any]]) -> None:
                train = next(item for item in rows if item["split"] == "train")
                test = next(item for item in rows if item["split"] == "test")
                # Retain the test image/shard/row-group, but claim that the
                # index entry belongs to a train schedule.  Its checksum is
                # refreshed by _rewrite_image_index to test the row guard.
                test["schedule_id"] = train["schedule_id"]
                test["split"] = "train"
                test["cv_fold"] = train["cv_fold"]

            _rewrite_image_index(leakage, forge_test_entry_as_train)
            store = ParquetImageStore(leakage, master)
            test_image = next(path for path in store.index if "secure_0006" in path)
            try:
                with self.assertRaises(ValueError):
                    store.load_image(test_image, purpose="train")
            finally:
                _close_parquet_store(store)

            bad_source_count = root / "bad-source-count"
            shutil.copytree(base, bad_source_count)
            manifest_path = bad_source_count / "shards.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_record_count"] = int(manifest["source_record_count"]) + 1
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_compatible_annotations(jsonl_path, csv_path, bad_source_count)

            stale = root / "stale-shard"
            shutil.copytree(base, stale)
            declared = json.loads((stale / "shards.manifest.json").read_text(encoding="utf-8"))["shards"]
            shutil.copyfile(stale / declared[0]["shard"], stale / "stale-unmanifested.parquet")
            with self.assertRaises(ValueError):
                verify_compatible_annotations(jsonl_path, csv_path, stale)

    def test_mixed_synthetic_and_real_optional_fields_share_one_schema(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            jsonl_path, csv_path, master = _write_fixture(
                root / "source", 6, objects_per_schedule=2, prefix="mixed",
            )
            shard_dir = root / "shards"
            manifest = build_parquet_shards(jsonl_path, master, shard_dir, max_schedules=6)
            verified = verify_compatible_annotations(jsonl_path, csv_path, shard_dir)
            self.assertEqual(verified["parquet_count"], 12)

            rows = []
            schemas = []
            for declared in manifest["shards"]:
                path = shard_dir / declared["shard"]
                table = pq.read_table(path)
                schemas.append(table.schema)
                rows.extend(table.to_pylist())
            self.assertTrue(all(schema == schemas[0] for schema in schemas))
            for field in (
                "source_domain", "registration_profile", "registration_high_confidence",
                "text_polygon_margin_px",
            ):
                self.assertIn(field, schemas[0].names)

            synthetic = next(item for item in rows if item["schedule_id"] == "mixed_0000")
            real = next(item for item in rows if item["schedule_id"] == "mixed_0001")
            self.assertEqual(synthetic["source_domain"], "synthetic")
            self.assertIsNone(synthetic["registration_profile"])
            self.assertIsNone(synthetic["registration_high_confidence"])
            self.assertIsNone(synthetic["text_polygon_margin_px"])
            self.assertEqual(real["source_domain"], "real")
            self.assertEqual(real["registration_profile"], "registered_phone_v1")
            self.assertIs(real["registration_high_confidence"], True)
            self.assertEqual(real["text_polygon_margin_px"], 1.25)

            index_rows = pq.read_table(shard_dir / "image_index.parquet").to_pylist()
            domains = {item["schedule_id"]: item["source_domain"] for item in index_rows}
            self.assertEqual(domains["mixed_0000"], "synthetic")
            self.assertEqual(domains["mixed_0001"], "real")

    def test_build_and_verify_100_to_500_schedules_are_streaming(self):
        measurements: dict[int, dict[str, int]] = {}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for schedule_count in (100, 500):
                source = root / f"source-{schedule_count}"
                jsonl_path, csv_path, master = _write_fixture(
                    source,
                    schedule_count,
                    objects_per_schedule=4,
                    payload_bytes=4096,
                    prefix=f"scale{schedule_count}",
                )
                shard_dir = root / f"shards-{schedule_count}"
                manifest, build_peak = _measure_peak(lambda: build_parquet_shards(
                    jsonl_path, master, shard_dir, max_schedules=16, max_objects=128,
                ))
                verified, verify_peak = _measure_peak(lambda: verify_compatible_annotations(
                    jsonl_path, csv_path, shard_dir,
                ))
                baseline_peak = _materialization_peak(jsonl_path)
                expected_objects = schedule_count * 4
                self.assertEqual(manifest["source_record_count"], expected_objects)
                self.assertEqual(sum(item["object_count"] for item in manifest["shards"]), expected_objects)
                self.assertEqual(verified["jsonl_count"], expected_objects)
                self.assertEqual(verified["csv_count"], expected_objects)
                self.assertEqual(verified["parquet_count"], expected_objects)
                measurements[schedule_count] = {
                    "source_size": jsonl_path.stat().st_size,
                    "build_peak": build_peak,
                    "verify_peak": verify_peak,
                    "baseline_peak": baseline_peak,
                }

            large = measurements[500]
            small = measurements[100]
            # Whole-file json.loads retains every 4 KiB payload.  Build and
            # verification should remain comfortably below that baseline.
            self.assertLess(large["build_peak"], large["baseline_peak"] * 0.75, measurements)
            self.assertLess(large["verify_peak"], large["baseline_peak"] * 0.75, measurements)
            source_growth = large["source_size"] - small["source_size"]
            self.assertLess(
                max(0, large["build_peak"] - small["build_peak"]),
                source_growth * 0.50 + 2_000_000,
                measurements,
            )
            self.assertLess(
                max(0, large["verify_peak"] - small["verify_peak"]),
                source_growth * 0.50 + 2_000_000,
                measurements,
            )

    def test_lazy_datasets_have_compact_spawn_payload_and_workers_read_two_batches(self):
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload_sizes = {}
            fixtures = {}
            for schedule_count in (100, 500):
                source = root / f"source-{schedule_count}"
                jsonl_path, _csv_path, master = _write_fixture(
                    source,
                    schedule_count,
                    objects_per_schedule=4,
                    payload_bytes=2048,
                    prefix=f"lazy{schedule_count}",
                    write_images=schedule_count == 500,
                )
                shard_dir = root / f"shards-{schedule_count}"
                build_parquet_shards(
                    jsonl_path, master, shard_dir, max_schedules=16, max_objects=128,
                )
                entries = load_parquet_image_entries([shard_dir], master, purpose="train")
                dense = LazyParquetDenseScheduleDataset(
                    entries,
                    [shard_dir],
                    master,
                    source,
                    purpose="train",
                    kind="dbnet",
                    training=False,
                    long_side=64,
                )
                recognizer = LazyParquetRecognitionCropDataset(
                    [shard_dir],
                    master,
                    source,
                    purpose="train",
                    charset=["D"],
                    training=False,
                )
                sampler = ParquetRecognitionBatchSampler(
                    [shard_dir],
                    master,
                    purpose="train",
                    batch_sizes={160: 2, 320: 2, 640: 2},
                    training=False,
                    shuffle=False,
                    shuffle_buffer_batches=4,
                )
                payload_sizes[schedule_count] = {
                    "dense": _dataset_spawn_payload_size(dense),
                    "recognizer": _dataset_spawn_payload_size(recognizer),
                    "sampler": len(pickle.dumps(sampler, protocol=pickle.HIGHEST_PROTOCOL)),
                    "annotations": jsonl_path.stat().st_size,
                }
                fixtures[schedule_count] = (dense, recognizer, sampler)

            large = payload_sizes[500]
            self.assertLess(large["dense"], large["annotations"] * 0.20, payload_sizes)
            self.assertLess(
                large["recognizer"] + large["sampler"],
                large["annotations"] * 0.20,
                payload_sizes,
            )
            # A five-fold schedule increase may enlarge compact image/master
            # metadata, but it must not reproduce four annotation dictionaries
            # (and their payload) per schedule in every worker.
            self.assertLess(payload_sizes[500]["dense"], payload_sizes[100]["dense"] * 7)
            self.assertLess(
                payload_sizes[500]["recognizer"] + payload_sizes[500]["sampler"],
                (payload_sizes[100]["recognizer"] + payload_sizes[100]["sampler"]) * 7,
            )

            dense, recognizer, sampler = fixtures[500]
            dense_loader = DataLoader(
                dense,
                batch_size=2,
                num_workers=2,
                multiprocessing_context="spawn",
            )
            dense_iterator = iter(dense_loader)
            try:
                dense_batches = [next(dense_iterator), next(dense_iterator)]
            finally:
                if hasattr(dense_iterator, "_shutdown_workers"):
                    dense_iterator._shutdown_workers()
            self.assertTrue(all(batch["image"].shape == (2, 3, 64, 64) for batch in dense_batches))

            recognition_loader = DataLoader(
                recognizer,
                batch_sampler=sampler,
                collate_fn=recognition_collate,
                num_workers=2,
                multiprocessing_context="spawn",
            )
            recognition_iterator = iter(recognition_loader)
            try:
                recognition_batches = [next(recognition_iterator), next(recognition_iterator)]
            finally:
                if hasattr(recognition_iterator, "_shutdown_workers"):
                    recognition_iterator._shutdown_workers()
            self.assertTrue(all(batch["image"].shape[0] == 2 for batch in recognition_batches))
            self.assertTrue(all(batch["display_text"] == ["D", "D"] for batch in recognition_batches))

            # DataLoader's sampler iterator owns a live ParquetFile until the
            # iterator itself is released.  Eager cleanup is required on
            # Windows, where TemporaryDirectory cannot unlink open files.
            del dense_iterator, dense_loader, recognition_iterator, recognition_loader
            del dense_batches, recognition_batches, dense, recognizer, sampler, entries
            fixtures.clear()
            gc.collect()


if __name__ == "__main__":
    unittest.main()
