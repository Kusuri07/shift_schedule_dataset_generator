import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from shift_ocr.training import (
    CheckpointManager, TrainConfig, _make_scheduler, _set_train_loader_epoch,
    apply_unfreezing, fit, model_loss, parameter_groups, resume_selection_decision,
)
from shift_ocr.models import build_model


class DummyDBNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.probability_logit = torch.nn.Parameter(torch.tensor(0.25))
        self.threshold_value = torch.nn.Parameter(torch.tensor(0.4))

    def forward(self, image):
        batch, _channels, height, width = image.shape
        probability = torch.sigmoid(self.probability_logit).expand(batch, 1, height, width).to(image.dtype)
        threshold = self.threshold_value.expand(batch, 1, height, width).to(image.dtype)
        return {"probability": probability, "threshold": threshold}


class DummyTableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(17)
        self.heatmap = torch.nn.Parameter(torch.randn(1, 1, 4, 4, generator=generator))
        self.corners = torch.nn.Parameter(torch.randn(1, 8, 4, 4, generator=generator))
        self.rows = torch.nn.Parameter(torch.randn(1, 8, 4, 4, generator=generator))
        self.columns = torch.nn.Parameter(torch.randn(1, 8, 4, 4, generator=generator))

    def forward(self, image):
        batch = image.shape[0]
        return {
            "cell_heatmap": self.heatmap.expand(batch, -1, -1, -1),
            "corner_offsets": self.corners.expand(batch, -1, -1, -1),
            "row_embedding": self.rows.expand(batch, -1, -1, -1),
            "column_embedding": self.columns.expand(batch, -1, -1, -1),
        }


class TrainingLossTests(unittest.TestCase):
    def test_dbnet_probability_bce_runs_in_fp32_inside_bf16_autocast(self):
        model = DummyDBNet()
        batch = {
            "image": torch.ones(1, 3, 4, 4, dtype=torch.bfloat16),
            "probability_target": torch.ones(1, 1, 2, 2),
            "threshold_target": torch.zeros(1, 1, 2, 2),
        }
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = model_loss("dbnet", model, batch)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.probability_logit.grad)

    def test_table_relation_targets_train_both_embedding_heads(self):
        model = DummyTableModel()
        valid = torch.zeros(1, 1, 4, 4)
        valid[0, 0, 0, :2] = 1
        valid[0, 0, 1, :2] = 1
        relation = torch.zeros(1, 2, 4, 4)
        relation[0, 0, 0, :2] = 1
        relation[0, 0, 1, :2] = 2
        relation[0, 1, :2, 0] = 1
        relation[0, 1, :2, 1] = 2
        batch = {
            "image": torch.ones(1, 3, 4, 4),
            "cell_heatmap_target": torch.zeros(1, 1, 4, 4),
            "corner_target": torch.zeros(1, 8, 4, 4),
            "corner_valid": valid,
            "relation_target": relation,
        }
        loss = model_loss("table", model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.rows.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.columns.grad.abs().sum()), 0.0)


class CheckpointAndFitTests(unittest.TestCase):
    def test_parameter_groups_keep_backbone_neck_and_head_trainable(self):
        model = build_model("dbnet")
        groups = parameter_groups(model, 3e-4)
        counts = {
            group["name"]: sum(parameter.numel() for parameter in group["params"])
            for group in groups
        }
        self.assertGreater(counts["backbone"], 0)
        self.assertGreater(counts["neck"], 0)
        self.assertGreater(counts["head"], 0)
        grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), {id(parameter) for parameter in model.parameters()})

        config = TrainConfig(
            model_kind="dbnet", partial_unfreeze_epoch=0, full_unfreeze_epoch=0,
        )
        self.assertEqual(apply_unfreezing(model, 0, config), "fully_unfrozen")
        optimizer = torch.optim.AdamW(groups)
        backbone_parameter = next(model.backbone.parameters())
        before = backbone_parameter.detach().clone()
        outputs = model(torch.rand(1, 3, 64, 64))
        (outputs["probability"].mean() + outputs["threshold"].mean()).backward()
        self.assertIsNotNone(backbone_parameter.grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, backbone_parameter.detach()))

    def test_checkpoint_restores_scheduler_and_accepts_legacy_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            config = TrainConfig(model_kind="dbnet", epochs=3)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            scheduler.step()
            manager = CheckpointManager(directory, "dbnet")
            checkpoint = manager.save(
                "state", model, optimizer, None, 0,
                {"text_polygon_hmean_iou_0_5": 0.5}, config,
                scheduler=scheduler, best_metrics={"text_polygon_hmean_iou_0_5": 0.5},
            )

            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
            restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=3)
            state = manager.load(
                checkpoint, restored_model, restored_optimizer, device="cpu", scheduler=restored_scheduler,
            )
            self.assertEqual(state["checkpoint_version"], 2)
            self.assertEqual(restored_scheduler.last_epoch, scheduler.last_epoch)
            self.assertEqual(restored_scheduler.get_last_lr(), scheduler.get_last_lr())
            for expected, actual in zip(model.parameters(), restored_model.parameters()):
                self.assertTrue(torch.equal(expected, actual))

            legacy = directory / "legacy.pt"
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 0,
                "metrics": {"text_polygon_hmean_iou_0_5": 0.5}, "config": {},
            }, legacy)
            manager.load(legacy, restored_model, restored_optimizer, device="cpu", scheduler=restored_scheduler)

            # A horizon-bound cosine checkpoint can be opened by the new
            # horizon-independent scheduler without importing its stale T_max.
            migrated_model = torch.nn.Linear(2, 1)
            migrated_optimizer = torch.optim.AdamW(migrated_model.parameters(), lr=0.01)
            migrated_scheduler = _make_scheduler(
                migrated_optimizer, TrainConfig(model_kind="dbnet", epochs=50),
            )
            manager.load(
                checkpoint, migrated_model, migrated_optimizer,
                device="cpu", scheduler=migrated_scheduler,
            )
            self.assertNotIn("T_max", migrated_scheduler.__dict__)
            self.assertEqual(migrated_scheduler.last_epoch, scheduler.last_epoch)
            restored_actual_lr = migrated_optimizer.param_groups[0]["lr"]
            migrated_scheduler.step()
            next_lr = migrated_optimizer.param_groups[0]["lr"]
            self.assertLessEqual(next_lr, restored_actual_lr)
            self.assertAlmostEqual(
                next_lr,
                max(config.scheduler_min_learning_rate, restored_actual_lr * 0.95),
                places=12,
            )

            # Lambda closures are not serialized by PyTorch.  A resume created
            # with a different CLI LR must still honor the checkpoint schedule
            # and its absolute floor instead of using the new closure.
            lambda_model = torch.nn.Linear(2, 1)
            lambda_optimizer = torch.optim.AdamW(lambda_model.parameters(), lr=0.01)
            lambda_config = TrainConfig(
                model_kind="dbnet", epochs=10, scheduler_min_learning_rate=0.001,
            )
            lambda_scheduler = _make_scheduler(lambda_optimizer, lambda_config)
            for _index in range(45):
                lambda_scheduler.step()
            lambda_checkpoint = manager.save(
                "lambda_floor", lambda_model, lambda_optimizer, None, 44,
                {"text_polygon_hmean_iou_0_5": 0.5}, lambda_config,
                scheduler=lambda_scheduler,
            )
            floor_model = torch.nn.Linear(2, 1)
            floor_optimizer = torch.optim.AdamW(floor_model.parameters(), lr=0.1)
            floor_scheduler = _make_scheduler(
                floor_optimizer,
                TrainConfig(model_kind="dbnet", epochs=100, scheduler_min_learning_rate=1e-6),
            )
            manager.load(
                lambda_checkpoint, floor_model, floor_optimizer,
                device="cpu", scheduler=floor_scheduler,
            )
            before_floor_step = floor_optimizer.param_groups[0]["lr"]
            floor_scheduler.step()
            self.assertLessEqual(floor_optimizer.param_groups[0]["lr"], before_floor_step)
            self.assertGreaterEqual(floor_optimizer.param_groups[0]["lr"], 0.001)

            zero_model = torch.nn.Linear(2, 1)
            zero_optimizer = torch.optim.AdamW(zero_model.parameters(), lr=0.01)
            zero_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(zero_optimizer, T_max=1)
            zero_scheduler.step()
            self.assertEqual(zero_optimizer.param_groups[0]["lr"], 0.0)
            zero_checkpoint = manager.save(
                "cosine_zero", zero_model, zero_optimizer, None, 0,
                {"text_polygon_hmean_iou_0_5": 0.5}, config,
                scheduler=zero_scheduler,
            )
            zero_restored_model = torch.nn.Linear(2, 1)
            zero_restored_optimizer = torch.optim.AdamW(zero_restored_model.parameters(), lr=0.01)
            zero_restored_scheduler = _make_scheduler(zero_restored_optimizer, config)
            manager.load(
                zero_checkpoint, zero_restored_model, zero_restored_optimizer,
                device="cpu", scheduler=zero_restored_scheduler,
            )
            zero_restored_scheduler.step()
            self.assertEqual(zero_restored_optimizer.param_groups[0]["lr"], 0.0)

    def test_legacy_empty_optimizer_groups_reset_and_missing_scheduler_is_continuous(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = TrainConfig(model_kind="dbnet", epochs=12)
            manager = CheckpointManager(root, "dbnet")
            model = build_model("dbnet")
            optimizer = torch.optim.AdamW(parameter_groups(model, 3e-4))
            checkpoint = manager.save(
                "prefixed", model, optimizer, None, 9,
                {"text_polygon_hmean_iou_0_5": 0.5}, config,
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state["optimizer"]["param_groups"][0]["params"] = []
            state["optimizer"]["param_groups"][1]["params"] = []
            torch.save(state, checkpoint)

            restored_model = build_model("dbnet")
            restored_optimizer = torch.optim.AdamW(parameter_groups(restored_model, 3e-4))
            restored_scheduler = _make_scheduler(restored_optimizer, config)
            loaded = manager.load(
                checkpoint, restored_model, restored_optimizer,
                device="cpu", scheduler=restored_scheduler,
            )
            self.assertFalse(loaded["_load_metadata"]["optimizer_restored"])
            self.assertEqual(
                loaded["_load_metadata"]["optimizer_restore"],
                "reset_legacy_consumed_parameter_generators",
            )
            self.assertEqual(
                loaded["_load_metadata"]["scheduler_restore"],
                "reset_with_legacy_optimizer_layout",
            )
            self.assertTrue(all(len(group["params"]) > 0 for group in restored_optimizer.param_groups))

            simple_model = torch.nn.Linear(2, 1)
            simple_optimizer = torch.optim.AdamW(simple_model.parameters(), lr=1e-5)
            no_scheduler = manager.save(
                "no_scheduler", simple_model, simple_optimizer, None, 9,
                {"text_polygon_hmean_iou_0_5": 0.5}, config,
            )
            simple_restored = torch.nn.Linear(2, 1)
            simple_restored_optimizer = torch.optim.AdamW(simple_restored.parameters(), lr=3e-4)
            simple_restored_scheduler = _make_scheduler(simple_restored_optimizer, config)
            loaded = manager.load(
                no_scheduler, simple_restored, simple_restored_optimizer,
                device="cpu", scheduler=simple_restored_scheduler,
            )
            before = simple_restored_optimizer.param_groups[0]["lr"]
            simple_restored_scheduler.step()
            after = simple_restored_optimizer.param_groups[0]["lr"]
            self.assertEqual(
                loaded["_load_metadata"]["scheduler_restore"],
                "anchored_to_optimizer_lr_missing_checkpoint_scheduler",
            )
            self.assertAlmostEqual(before, 1e-5)
            self.assertLessEqual(after, before)
            self.assertGreaterEqual(after, config.scheduler_min_learning_rate)

    def test_uninterrupted_and_resumed_loaders_use_the_same_absolute_epoch_order(self):
        from torch.utils.data import DataLoader, Dataset
        from shift_ocr.datasets import WidthBucketBatchSampler

        class IndexDataset(Dataset):
            def __init__(self):
                self.absolute_epoch = None
                self.records = [
                    {
                        "cell_polygon": [[0, 0], [20, 0], [20, 10], [0, 10]],
                    }
                    for _index in range(12)
                ]

            def __len__(self):
                return len(self.records)

            def __getitem__(self, index):
                return index

            def set_epoch(self, epoch):
                self.absolute_epoch = int(epoch)

        dataset = IndexDataset()

        def shuffled_loader():
            return DataLoader(
                dataset, batch_size=3, shuffle=True,
                generator=torch.Generator().manual_seed(41),
            )

        uninterrupted = shuffled_loader()
        _set_train_loader_epoch(uninterrupted, 0, 41)
        self.assertEqual(dataset.absolute_epoch, 0)
        epoch_zero = torch.cat(list(uninterrupted)).tolist()
        _set_train_loader_epoch(uninterrupted, 1, 41)
        self.assertEqual(dataset.absolute_epoch, 1)
        uninterrupted_epoch_one = torch.cat(list(uninterrupted)).tolist()

        resumed = shuffled_loader()
        _set_train_loader_epoch(resumed, 1, 41)
        resumed_epoch_one = torch.cat(list(resumed)).tolist()
        self.assertNotEqual(epoch_zero, uninterrupted_epoch_one)
        self.assertEqual(uninterrupted_epoch_one, resumed_epoch_one)

        def bucket_loader():
            sampler = WidthBucketBatchSampler(
                dataset, {160: 3, 320: 2, 640: 1}, seed=41,
            )
            return DataLoader(dataset, batch_sampler=sampler)

        bucket_uninterrupted = bucket_loader()
        _set_train_loader_epoch(bucket_uninterrupted, 0, 41)
        bucket_epoch_zero = torch.cat(list(bucket_uninterrupted)).tolist()
        _set_train_loader_epoch(bucket_uninterrupted, 1, 41)
        bucket_epoch_one = torch.cat(list(bucket_uninterrupted)).tolist()
        bucket_resumed = bucket_loader()
        _set_train_loader_epoch(bucket_resumed, 1, 41)
        bucket_resumed_epoch_one = torch.cat(list(bucket_resumed)).tolist()
        self.assertNotEqual(bucket_epoch_zero, bucket_epoch_one)
        self.assertEqual(bucket_epoch_one, bucket_resumed_epoch_one)

    def test_fit_steps_final_partial_accumulation_and_writes_epoch_logs(self):
        batches = [{
            "image": torch.ones(1, 3, 4, 4),
            "probability_target": torch.ones(1, 1, 2, 2),
            "threshold_target": torch.zeros(1, 1, 2, 2),
        } for _index in range(3)]
        validation_values = iter([(0.7, 0.4), (0.6, 0.5)])

        def validation(_model):
            score, loss = next(validation_values)
            return {"text_polygon_hmean_iou_0_5": score, "validation_loss": loss}

        original_adamw = torch.optim.AdamW
        optimizers = []

        class CountingAdamW(original_adamw):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.step_calls = 0

            def step(self, *args, **kwargs):
                self.step_calls += 1
                return super().step(*args, **kwargs)

        def make_optimizer(*args, **kwargs):
            optimizer = CountingAdamW(*args, **kwargs)
            optimizers.append(optimizer)
            return optimizer

        config = TrainConfig(
            model_kind="dbnet", epochs=2, requested_batch_size=1,
            target_effective_batch=2, checkpoint_every=5,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            with patch("shift_ocr.training.select_device_and_precision", return_value=(torch.device("cpu"), "fp32", None)), patch.object(
                torch.optim, "AdamW", side_effect=make_optimizer,
            ), patch.object(torch.cuda, "is_available", return_value=False):
                manifest = fit(DummyDBNet(), batches, validation, config, directory)

            self.assertEqual(optimizers[0].step_calls, 4)
            self.assertEqual(manifest["history"][0]["optimizer_step_sample_counts"], [2, 1])
            self.assertEqual([item["best_updated"] for item in manifest["history"]], [True, False])
            self.assertLess(manifest["history"][1]["learning_rate"], manifest["history"][0]["learning_rate"])
            for key in (
                "train_loss", "validation_loss", "learning_rate", "gpu_vram_peak_mb",
                "epoch_duration_seconds", "best_updated",
            ):
                self.assertIn(key, manifest["history"][0])

            with (directory / "training_history.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            history = json.loads((directory / "training_history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(history), 2)
            checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
            self.assertIsNotNone(checkpoint["scheduler"])
            self.assertEqual(checkpoint["scheduler"]["last_epoch"], 2)

    def test_sample_aware_accumulation_handles_mixed_width_micro_batches(self):
        batch_sizes_and_widths = [(4, 160), (2, 320), (1, 640), (4, 160)]
        batches = []
        for batch_size, width in batch_sizes_and_widths:
            batches.append({
                "image": torch.ones(batch_size, 3, 4, width),
                "probability_target": torch.ones(batch_size, 1, 2, max(2, width // 2)),
                "threshold_target": torch.zeros(batch_size, 1, 2, max(2, width // 2)),
            })

        original_adamw = torch.optim.AdamW
        optimizers = []

        class CountingAdamW(original_adamw):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.step_calls = 0

            def step(self, *args, **kwargs):
                self.step_calls += 1
                return super().step(*args, **kwargs)

        def make_optimizer(*args, **kwargs):
            optimizer = CountingAdamW(*args, **kwargs)
            optimizers.append(optimizer)
            return optimizer

        config = TrainConfig(
            model_kind="dbnet", epochs=1, requested_batch_size=4,
            target_effective_batch=5, checkpoint_every=5,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "shift_ocr.training.select_device_and_precision",
                return_value=(torch.device("cpu"), "fp32", None),
            ), patch.object(torch.optim, "AdamW", side_effect=make_optimizer), patch.object(
                torch.cuda, "is_available", return_value=False,
            ):
                manifest = fit(
                    DummyDBNet(), batches,
                    lambda _model: {
                        "text_polygon_hmean_iou_0_5": 0.5,
                        "validation_loss": 0.5,
                    },
                    config, Path(temporary_directory),
                )

        epoch = manifest["history"][0]
        self.assertEqual(optimizers[0].step_calls, 2)
        self.assertEqual(epoch["optimizer_step_sample_counts"], [6, 5])
        self.assertEqual(epoch["train_sample_count"], 11)
        self.assertEqual(epoch["effective_batch_min"], 5)
        self.assertEqual(epoch["effective_batch_max"], 6)

    def test_fit_rejects_resume_target_that_would_run_zero_epochs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            config = TrainConfig(model_kind="dbnet", epochs=1)
            model = DummyDBNet()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            checkpoint = CheckpointManager(directory, "dbnet").save(
                "last", model, optimizer, None, 0,
                {"text_polygon_hmean_iou_0_5": 0.5}, config,
            )
            with patch(
                "shift_ocr.training.select_device_and_precision",
                return_value=(torch.device("cpu"), "fp32", None),
            ), patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(ValueError, "no epochs would run"):
                    fit(
                        DummyDBNet(), [],
                        lambda _model: {"text_polygon_hmean_iou_0_5": 0.5},
                        config, directory, resume=checkpoint,
                    )

    def test_resume_learning_rate_policy_restores_or_explicitly_resets(self):
        batch = {
            "image": torch.ones(1, 3, 4, 4),
            "probability_target": torch.ones(1, 1, 2, 2),
            "threshold_target": torch.zeros(1, 1, 2, 2),
        }
        validation = lambda _model: {
            "text_polygon_hmean_iou_0_5": 0.5,
            "validation_loss": 0.5,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_model = DummyDBNet()
            source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=2e-5)
            checkpoint = CheckpointManager(root, "dbnet").save(
                "source", source_model, source_optimizer, None, 0,
                {"text_polygon_hmean_iou_0_5": 0.5},
                TrainConfig(model_kind="dbnet", epochs=1),
            )
            common_patches = (
                patch(
                    "shift_ocr.training.select_device_and_precision",
                    return_value=(torch.device("cpu"), "fp32", None),
                ),
                patch.object(torch.cuda, "is_available", return_value=False),
            )
            with common_patches[0], common_patches[1]:
                restored = fit(
                    DummyDBNet(), [batch], validation,
                    TrainConfig(
                        model_kind="dbnet", epochs=2, learning_rate=1e-2,
                        resume_learning_rate_policy="restore",
                    ),
                    root / "restored", resume=checkpoint,
                )
            with patch(
                "shift_ocr.training.select_device_and_precision",
                return_value=(torch.device("cpu"), "fp32", None),
            ), patch.object(torch.cuda, "is_available", return_value=False):
                reset = fit(
                    DummyDBNet(), [batch], validation,
                    TrainConfig(
                        model_kind="dbnet", epochs=2, learning_rate=1e-2,
                        resume_learning_rate_policy="reset",
                    ),
                    root / "reset", resume=checkpoint,
                )

        self.assertEqual(restored["learning_rate_state"]["source"], "checkpoint_optimizer")
        self.assertAlmostEqual(
            restored["learning_rate_state"]["initial_group_learning_rates"][0]["lr"],
            2e-5,
        )
        self.assertEqual(reset["learning_rate_state"]["source"], "cli_resume_reset")
        self.assertAlmostEqual(
            reset["learning_rate_state"]["initial_group_learning_rates"][0]["lr"],
            1e-2,
        )

    def test_selection_state_resets_across_phase_or_scope_but_same_phase_inherits(self):
        batch = {
            "image": torch.ones(1, 3, 4, 4),
            "probability_target": torch.ones(1, 1, 2, 2),
            "threshold_target": torch.zeros(1, 1, 2, 2),
        }
        source_metrics = {
            "epoch": 0,
            "text_polygon_hmean_iou_0_5": 0.9,
            "validation_loss": 0.2,
            "best_updated": True,
        }
        validation = lambda _model: {
            "text_polygon_hmean_iou_0_5": 0.1,
            "validation_loss": 0.8,
        }

        def run_resume(checkpoint, output, config):
            with patch(
                "shift_ocr.training.select_device_and_precision",
                return_value=(torch.device("cpu"), "fp32", None),
            ), patch.object(torch.cuda, "is_available", return_value=False):
                return fit(
                    DummyDBNet(), [batch], validation, config, output,
                    resume=checkpoint,
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_config = TrainConfig(
                model_kind="dbnet", epochs=1,
                training_phase="synthetic_pretrain",
                selection_scope="synthetic_validation",
            )
            source_model = DummyDBNet()
            source_optimizer = torch.optim.AdamW(
                source_model.parameters(), lr=source_config.learning_rate,
            )
            source_manager = CheckpointManager(root / "source", "dbnet")
            checkpoint = source_manager.save(
                "last", source_model, source_optimizer, None, 0,
                source_metrics, source_config, best_metrics=source_metrics,
                history=[source_metrics],
            )
            source_manager.save(
                "best", source_model, source_optimizer, None, 0,
                source_metrics, source_config, best_metrics=source_metrics,
                history=[source_metrics],
            )

            same_output = root / "same"
            same = run_resume(
                checkpoint, same_output,
                TrainConfig(
                    model_kind="dbnet", epochs=2,
                    training_phase="synthetic_pretrain",
                    selection_scope="synthetic_validation",
                ),
            )
            real_output = root / "real"
            real = run_resume(
                checkpoint, real_output,
                TrainConfig(
                    model_kind="dbnet", epochs=2,
                    training_phase="real_finetune",
                    selection_scope="real_validation",
                ),
            )
            fold_output = root / "fold"
            fold = run_resume(
                checkpoint, fold_output,
                TrainConfig(
                    model_kind="dbnet", epochs=2,
                    training_phase="synthetic_pretrain",
                    selection_scope="train_cv_fold:0",
                ),
            )

            self.assertTrue(same["selection_state"]["inherited"])
            self.assertEqual([item["epoch"] for item in same["history"]], [0, 1])
            self.assertFalse(same["history"][-1]["best_updated"])
            self.assertTrue((same_output / "best.pt").exists())
            self.assertEqual(
                same["selection_state"]["inherited_best_checkpoint"],
                str(root / "source" / "best.pt"),
            )

            self.assertTrue(real["selection_state"]["reset"])
            self.assertIn("training_phase_changed", real["selection_state"]["reset_reason"])
            self.assertEqual([item["epoch"] for item in real["history"]], [1])
            self.assertTrue(real["history"][0]["best_updated"])
            self.assertTrue((real_output / "best.pt").exists())

            self.assertTrue(fold["selection_state"]["reset"])
            self.assertIn("selection_scope_changed", fold["selection_state"]["reset_reason"])
            self.assertEqual([item["epoch"] for item in fold["history"]], [1])
            self.assertTrue((fold_output / "best.pt").exists())

        explicit = resume_selection_decision(
            {
                "config": {
                    "training_phase": "synthetic_pretrain",
                    "selection_scope": "synthetic_validation",
                }
            },
            TrainConfig(
                model_kind="dbnet", resume_selection_policy="reset",
                training_phase="synthetic_pretrain",
                selection_scope="synthetic_validation",
            ),
        )
        self.assertEqual(explicit["reset_reason"], "explicit_reset_policy")
        legacy_fold = resume_selection_decision(
            {"config": {}},
            TrainConfig(
                model_kind="dbnet", training_phase="synthetic_pretrain",
                selection_scope="train_cv_fold:0",
            ),
        )
        self.assertTrue(legacy_fold["reset"])
        self.assertEqual(legacy_fold["reset_reason"], "legacy_unknown_scope_to:train_cv_fold:0")

    def test_legacy_last_uses_adjacent_manifest_best_before_last_metrics(self):
        batch = {
            "image": torch.ones(1, 3, 4, 4),
            "probability_target": torch.ones(1, 1, 2, 2),
            "threshold_target": torch.zeros(1, 1, 2, 2),
        }
        config = TrainConfig(
            model_kind="dbnet", epochs=2,
            training_phase="synthetic_pretrain", selection_scope="synthetic_validation",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            manager = CheckpointManager(source, "dbnet")
            model = DummyDBNet()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            last_metrics = {"epoch": 0, "text_polygon_hmean_iou_0_5": 0.5}
            checkpoint = manager.save(
                "last", model, optimizer, None, 0, last_metrics, config,
            )
            historical_best = {"epoch": 0, "text_polygon_hmean_iou_0_5": 0.9}
            manager.save(
                "best", model, optimizer, None, 0, historical_best, config,
                best_metrics=historical_best,
            )
            (source / "training_manifest.json").write_text(json.dumps({
                "best_metrics": historical_best,
                "history": [last_metrics],
            }), encoding="utf-8")

            with patch(
                "shift_ocr.training.select_device_and_precision",
                return_value=(torch.device("cpu"), "fp32", None),
            ), patch.object(torch.cuda, "is_available", return_value=False):
                result = fit(
                    DummyDBNet(), [batch],
                    lambda _model: {
                        "text_polygon_hmean_iou_0_5": 0.7,
                        "validation_loss": 0.4,
                    },
                    config, root / "resumed", resume=checkpoint,
                )
            self.assertEqual(result["best_metrics"]["text_polygon_hmean_iou_0_5"], 0.9)
            self.assertFalse(result["history"][-1]["best_updated"])
            self.assertTrue((root / "resumed" / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
