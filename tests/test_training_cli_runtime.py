import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train_models import (
    LazyBatchLimit,
    accumulation_settings,
    average_validation_loss,
    batch_candidates,
    dataloader_options,
    epochs_for_run,
    initial_learning_rate_details,
    probe_actual_batches,
    recognizer_batch_sizes,
    resolve_num_workers,
    resolve_storage_paths,
    build_parser,
)


class LazyBatchLimitTests(unittest.TestCase):
    def test_limit_is_lazy_and_reiterable(self):
        class CountingLoader:
            def __init__(self):
                self.yielded = 0

            def __iter__(self):
                for value in range(100):
                    self.yielded += 1
                    yield value

            def __len__(self):
                return 100

        source = CountingLoader()
        limited = LazyBatchLimit(source, 2)
        self.assertEqual(source.yielded, 0)
        self.assertEqual(list(limited), [0, 1])
        self.assertEqual(source.yielded, 2)
        self.assertEqual(list(limited), [0, 1])
        self.assertEqual(source.yielded, 4)
        self.assertEqual(len(limited), 2)


class RuntimeSelectionTests(unittest.TestCase):
    def test_default_training_paths_use_d_drive_storage_root(self):
        args = resolve_storage_paths(build_parser().parse_args(["--model", "dbnet"]))
        self.assertEqual(args.storage_root, Path(r"D:\harudam_model"))
        self.assertEqual(args.shard_dir, [Path(r"D:\harudam_model\training_dataset\shards")])
        self.assertEqual(
            args.master_split,
            Path(r"D:\harudam_model\training_dataset\splits\master_split.jsonl"),
        )
        self.assertEqual(args.image_root, Path(r"D:\harudam_model\training_dataset"))
        self.assertEqual(args.output_dir, Path(r"D:\harudam_model\runs\dbnet_pretrain"))

    def test_explicit_training_paths_override_storage_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = resolve_storage_paths(build_parser().parse_args([
                "--model", "table",
                "--objects", str(root / "objects.jsonl"),
                "--master-split", str(root / "split.jsonl"),
                "--image-root", str(root / "images"),
                "--output-dir", str(root / "run"),
            ]))
        self.assertIsNone(args.shard_dir)
        self.assertEqual(args.output_dir, (root / "run").resolve())

    def test_windows_worker_default_and_loader_options(self):
        self.assertEqual(resolve_num_workers(None, logical_cpu_count=20, platform_name="Windows"), (4, "auto_windows_safe"))
        self.assertEqual(resolve_num_workers(0, logical_cpu_count=20, platform_name="Windows"), (0, "user"))
        with self.assertRaises(ValueError):
            resolve_num_workers(-1)
        zero = dataloader_options(0, cuda=False, persistent=True)
        self.assertFalse(zero["persistent_workers"])
        self.assertNotIn("prefetch_factor", zero)
        workers = dataloader_options(2, cuda=True, persistent=True)
        self.assertTrue(workers["persistent_workers"])
        self.assertEqual(workers["prefetch_factor"], 2)
        self.assertTrue(workers["pin_memory"])

    def test_batch_and_accumulation_settings(self):
        self.assertEqual(batch_candidates(9), [9, 4, 2, 1])
        self.assertEqual(recognizer_batch_sizes(8), {160: 8, 320: 4, 640: 1})
        self.assertEqual(accumulation_settings(3, 32), (11, 33))

    def test_resumed_smoke_targets_exactly_next_epoch(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "last.pt"
            torch.save({"epoch": 6}, checkpoint)
            self.assertEqual(epochs_for_run(requested_epochs=30, smoke=True, resume=checkpoint), 8)
            self.assertEqual(epochs_for_run(requested_epochs=30, smoke=False, resume=checkpoint), 37)
            self.assertEqual(epochs_for_run(requested_epochs=30, smoke=True, resume=None), 1)
            self.assertEqual(epochs_for_run(requested_epochs=30, smoke=False, resume=None), 30)

    def test_runtime_lr_details_restore_checkpoint_or_explicitly_reset(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5)
            checkpoint = Path(tmp) / "last.pt"
            torch.save({"optimizer": optimizer.state_dict(), "epoch": 0}, checkpoint)
            restored, restored_source = initial_learning_rate_details(
                model, 1e-2, resume=checkpoint, resume_policy="restore",
            )
            reset, reset_source = initial_learning_rate_details(
                model, 1e-2, resume=checkpoint, resume_policy="reset",
            )
        self.assertEqual(restored_source, "checkpoint_optimizer")
        self.assertAlmostEqual(restored[0]["lr"], 7e-5)
        self.assertEqual(reset_source, "cli_resume_reset")
        self.assertAlmostEqual(reset[0]["lr"], 1e-2)

    def test_validation_loss_is_weighted_by_actual_batch_size(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(str(exc))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.0))

        loader = [
            {"image": torch.zeros(3, 1, 1, 1)},
            {"image": torch.zeros(1, 1, 1, 1)},
        ]
        with patch(
            "train_models.model_loss",
            side_effect=[torch.tensor(2.0), torch.tensor(8.0)],
        ):
            loss = average_validation_loss(
                Model(), loader, model_kind="dbnet",
                device=torch.device("cpu"), autocast_dtype=None,
            )
        self.assertAlmostEqual(loss, 3.5)


class DenseValidationCollateTests(unittest.TestCase):
    def test_identity_recipe_collates_and_is_json_serializable(self):
        try:
            import cv2
            import numpy as np
            from torch.utils.data import DataLoader
            from shift_ocr.datasets import DenseScheduleDataset
        except ImportError as exc:
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index in range(2):
                image_path = f"schedule_{index}.png"
                cv2.imwrite(str(root / image_path), np.full((24, 32, 3), 255, np.uint8))
                records.append({
                    "schedule_id": f"schedule_{index}",
                    "image_path": image_path,
                    "cell_polygon": [[1, 1], [30, 1], [30, 22], [1, 22]],
                    "text_polygon": [[3, 3], [10, 3], [10, 10], [3, 10]],
                })
            dataset = DenseScheduleDataset(records, root, kind="dbnet", training=False, long_side=32)
            item = dataset[0]
            self.assertEqual(item["augmentation_recipe"]["kind"], "identity")
            self.assertIsInstance(item["augmentation_recipe"].get("epoch", 0), int)
            json.dumps(item["augmentation_recipe"])
            batch = next(iter(DataLoader(dataset, batch_size=2)))
            self.assertEqual(batch["augmentation_recipe"]["kind"], ["identity", "identity"])


class RealBatchProbeTests(unittest.TestCase):
    def test_probe_uses_real_loader_batches(self):
        try:
            import torch
            from torch.utils.data import DataLoader, Dataset
        except ImportError as exc:
            self.skipTest(str(exc))
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        class TinyDataset(Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, index):
                return {
                    "image": torch.full((3, 16, 16), index / 8, dtype=torch.float32),
                    "probability_target": torch.zeros(1, 16, 16),
                    "threshold_target": torch.zeros(1, 16, 16),
                }

        class TinyDbnet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Conv2d(3, 2, 1)

            def forward(self, image):
                output = self.head(image)
                return {
                    "probability": output[:, :1].sigmoid(),
                    "threshold": output[:, 1:2],
                }

        selected, report = probe_actual_batches(
            model_kind="dbnet",
            model_factory=TinyDbnet,
            loader_factory=lambda batch_size, _persistent: DataLoader(TinyDataset(), batch_size=batch_size),
            requested_batch_size=4,
            minimum_batch_size=1,
            iterations=2,
            learning_rate=3e-4,
            device=torch.device("cuda"),
            precision="fp32",
            autocast_dtype=None,
        )
        self.assertEqual(selected, 4)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["attempts"][-1]["iterations_completed"], 2)
        self.assertGreater(report["peak_allocated_mb"], 0)


if __name__ == "__main__":
    unittest.main()
