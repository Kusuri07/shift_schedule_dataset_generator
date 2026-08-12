import multiprocessing
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from shift_ocr.datasets import DenseScheduleDataset, RecognitionCropDataset
from shift_ocr.augmentation import augment_image_and_objects, sample_recipe


def _persistent_epoch_observer(dataset, command_queue, result_queue):
    """Keep one spawned dataset copy alive while its parent changes epoch."""

    while True:
        command = command_queue.get(timeout=10)
        if command == "stop":
            return
        item = dataset[0]
        result_queue.put((
            item.get("augmentation_epoch"),
            item.get("augmentation_seed"),
            item.get("crop_source"),
        ))


def _record():
    return {
        "schedule_id": "schedule_0001",
        "image_path": "schedule.png",
        "object_type": "shift_code",
        "display_text": "D",
        "canonical_code": "D",
        "row_id": "row_1",
        "row_index": 1,
        "day": 1,
        "ignore": False,
        "cell_polygon": [[20, 20], [180, 20], [180, 80], [20, 80]],
        "text_polygon": [[70, 35], [130, 35], [130, 65], [70, 65]],
    }


class DatasetEpochAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        image = np.full((100, 200, 3), 255, np.uint8)
        cv2.putText(image, "D", (80, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.imwrite(str(self.root / "schedule.png"), image)

    def tearDown(self):
        self.temporary.cleanup()

    def test_dense_recipe_is_reproducible_within_epoch_and_changes_across_epochs(self):
        dataset = DenseScheduleDataset(
            [_record()], self.root, kind="dbnet", training=True, long_side=200, seed=17,
        )
        epoch_zero = dataset[0]["augmentation_recipe"]
        self.assertEqual(epoch_zero, dataset[0]["augmentation_recipe"])
        dataset.set_epoch(1)
        epoch_one = dataset[0]["augmentation_recipe"]
        self.assertEqual(epoch_one, dataset[0]["augmentation_recipe"])
        self.assertEqual(epoch_zero["epoch"], 0)
        self.assertEqual(epoch_one["epoch"], 1)
        self.assertNotEqual(epoch_zero["seed"], epoch_one["seed"])
        self.assertNotEqual(epoch_zero["rotation_deg"], epoch_one["rotation_deg"])

    def test_recognizer_mode_and_jitter_seed_follow_absolute_epoch(self):
        dataset = RecognitionCropDataset(
            [_record()], self.root, charset=["D"], training=True, seed=23,
        )
        epoch_zero = dataset[0]
        self.assertEqual(epoch_zero["augmentation_epoch"], 0)
        self.assertEqual(epoch_zero["augmentation_seed"], dataset[0]["augmentation_seed"])
        self.assertEqual(epoch_zero["crop_source"], dataset[0]["crop_source"])
        dataset.set_epoch(7)
        epoch_seven = dataset[0]
        self.assertEqual(epoch_seven["augmentation_epoch"], 7)
        self.assertNotEqual(epoch_zero["augmentation_seed"], epoch_seven["augmentation_seed"])
        # Recreating at the absolute resume epoch must produce the same sample.
        resumed = RecognitionCropDataset(
            [_record()], self.root, charset=["D"], training=True, seed=23,
        )
        resumed.set_epoch(7)
        resumed_item = resumed[0]
        self.assertEqual(epoch_seven["augmentation_seed"], resumed_item["augmentation_seed"])
        self.assertEqual(epoch_seven["crop_source"], resumed_item["crop_source"])
        np.testing.assert_array_equal(epoch_seven["image"].numpy(), resumed_item["image"].numpy())

    def test_shared_epoch_is_pickle_and_spawn_persistent_worker_safe(self):
        dataset = RecognitionCropDataset(
            [_record()], self.root, charset=["D"], training=True, seed=31,
        )
        # Windows DataLoader serializes the dataset through multiprocessing's
        # spawn reducer (plain pickle intentionally cannot serialize shared
        # ctypes outside an active spawn operation).
        context = multiprocessing.get_context("spawn")
        commands = context.Queue()
        results = context.Queue()
        worker = context.Process(
            target=_persistent_epoch_observer,
            args=(dataset, commands, results),
        )
        worker.start()
        try:
            commands.put("sample")
            epoch_zero = results.get(timeout=20)
            dataset.set_epoch(4)
            commands.put("sample")
            epoch_four = results.get(timeout=20)
            self.assertEqual(epoch_zero[0], 0)
            self.assertEqual(epoch_four[0], 4)
            self.assertNotEqual(epoch_zero[1], epoch_four[1])
        finally:
            commands.put("stop")
            worker.join(timeout=20)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        self.assertEqual(worker.exitcode, 0)

    def test_negative_absolute_epoch_is_rejected(self):
        dataset = DenseScheduleDataset(
            [_record()], self.root, kind="table", training=True, long_side=200,
        )
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            dataset.set_epoch(-1)

    def test_augmentation_never_clears_source_ignore(self):
        source = dict(_record(), ignore=True)
        recipe = replace(
            sample_recipe(101), rotation_deg=0.0, scale=1.0,
            translate_x=0.0, translate_y=0.0,
            perspective=((0.0, 0.0),) * 4,
        )
        image = np.full((100, 200, 3), 255, np.uint8)
        _image, objects, _matrix, _details = augment_image_and_objects(
            image, [source], recipe,
        )
        self.assertEqual(len(objects), 1)
        self.assertTrue(objects[0]["ignore"])


if __name__ == "__main__":
    unittest.main()
