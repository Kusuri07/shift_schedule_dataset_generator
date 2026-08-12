#!/usr/bin/env python3
"""Train DBNet, recognizer or table model without artifact-tool dependencies."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from shift_ocr.charset import load_charset
from shift_ocr.datasets import (
    DenseScheduleDataset, LazyParquetDenseScheduleDataset,
    LazyParquetRecognitionCropDataset, ParquetRecognitionBatchSampler,
    RecognitionCropDataset, WidthBucketBatchSampler, RareCodeCropSampler,
    load_parquet_image_entries, load_records, recognition_collate, train_cv_partition,
)
from shift_ocr.master_split import MasterSplit
from shift_ocr.models import build_model
from shift_ocr.paths import DEFAULT_STORAGE_ROOT, dataset_root, training_run_dir
from shift_ocr.training import (
    TrainConfig, classify_optimizer_group_layout, fit, model_loss, move_to_device, parameter_groups,
    preview_resume_selection, seed_everything, select_device_and_precision,
)
from shift_ocr.validation import validate_dbnet, validate_recognizer, validate_table


class LazyBatchLimit:
    """Re-iterable, lazy view over the first ``limit`` batches of a loader."""

    def __init__(self, loader: Iterable[Any], limit: int) -> None:
        if limit < 1:
            raise ValueError("batch limit must be positive")
        self.loader = loader
        self.limit = limit

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.limit)

    def __len__(self) -> int:
        try:
            return min(len(self.loader), self.limit)  # type: ignore[arg-type]
        except TypeError:
            return self.limit


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be at least 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dbnet", "recognizer", "table"])
    parser.add_argument(
        "--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT,
        help=r"Default dataset, run and checkpoint root (default: D:\harudam_model)",
    )
    parser.add_argument("--objects", type=Path, action="append", help="Legacy/smoke JSONL path; repeat for synthetic and registered-real annotations")
    parser.add_argument(
        "--shard-dir", type=Path, action="append",
        help="Scalable training_shards_v2 directory; repeat for synthetic and registered-real shards",
    )
    parser.add_argument("--master-split", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--charset", type=Path, default=Path("data/korean_charset_v1.txt"))
    parser.add_argument(
        "--epochs", type=positive_int, default=30,
        help="Epochs for a new run; additional epochs when --resume is used (smoke always adds exactly one)",
    )
    parser.add_argument("--batch-size", type=positive_int, default=8, help="Requested physical batch before the CUDA dry-run")
    parser.add_argument("--effective-batch-size", type=positive_int, default=32, help="Target batch after gradient accumulation")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", type=positive_int, default=1280, help="Dense-model long side; CUDA auto-tuning never lowers it")
    parser.add_argument("--num-workers", type=nonnegative_int, help="DataLoader workers; omitted selects a Windows-safe value")
    parser.add_argument("--dry-run-iterations", type=int, choices=[1, 2], default=2)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--resume-lr-policy", choices=["restore", "reset"], default="restore",
        help=(
            "Resume optimizer/scheduler LR from the checkpoint (default), or explicitly "
            "reset optimizer/scheduler state to --learning-rate for a new fine-tuning phase"
        ),
    )
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--cv-fold", type=int, choices=[0, 1, 2], help="Train-only grouped CV fold")
    parser.add_argument("--phase", choices=["synthetic_pretrain", "real_finetune"], default="synthetic_pretrain")
    parser.add_argument(
        "--reset-selection-state", action="store_true",
        help=(
            "Discard checkpoint best/history for a deliberately new validation selection scope; "
            "phase and declared scope changes are reset automatically"
        ),
    )
    parser.add_argument("--table-predictions", type=Path, help="JSONL predicted cell polygons with confidence and gt_iou")
    parser.add_argument("--smoke", action="store_true", help="Run one epoch with two lazy train/validation mini-batches")
    return parser


def resolve_storage_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Fill omitted training paths from the desktop storage root.

    Explicit path arguments always win, preserving custom and smoke workflows.
    """

    storage_root = args.storage_root.expanduser().resolve()
    training_dataset = dataset_root(storage_root)
    path_sources = {
        "storage_root": "cli" if args.storage_root != DEFAULT_STORAGE_ROOT else "default",
        "shard_dir": "cli" if args.shard_dir else "default",
        "master_split": "cli" if args.master_split else "default",
        "image_root": "cli" if args.image_root else "default",
        "output_dir": "cli" if args.output_dir else "default",
    }
    if not args.objects and not args.shard_dir:
        args.shard_dir = [training_dataset / "shards"]
    if args.master_split is None:
        args.master_split = training_dataset / "splits" / "master_split.jsonl"
    if args.image_root is None:
        args.image_root = training_dataset
    if args.output_dir is None:
        args.output_dir = training_run_dir(
            args.model, args.phase, storage_root=storage_root, cv_fold=args.cv_fold,
        )
    args.storage_root = storage_root
    args.master_split = args.master_split.expanduser().resolve()
    args.image_root = args.image_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args._path_sources = path_sources
    return args


def resolve_num_workers(
    requested: int | None,
    *,
    logical_cpu_count: int | None = None,
    platform_name: str | None = None,
) -> tuple[int, str]:
    """Choose a conservative worker count, particularly for Windows spawn."""

    if requested is not None:
        if requested < 0:
            raise ValueError("num_workers cannot be negative")
        return requested, "user"
    logical = max(1, logical_cpu_count or os.cpu_count() or 1)
    system = (platform_name or platform.system()).lower()
    cap = 4 if system.startswith("win") else 8
    # Leave at least one logical CPU for the parent process and avoid copying
    # several 1280px prefetched batches into too many Windows workers.
    selected = min(cap, max(0, (logical - 1) // 2))
    return selected, "auto_windows_safe" if system.startswith("win") else "auto"


def dataloader_options(
    num_workers: int, *, cuda: bool, persistent: bool, seed: int | None = None,
) -> dict[str, Any]:
    import torch

    options: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": cuda,
        "persistent_workers": bool(persistent and num_workers > 0),
    }
    if num_workers > 0:
        options["prefetch_factor"] = 2
        options["worker_init_fn"] = seed_data_worker
    if seed is not None:
        options["generator"] = torch.Generator().manual_seed(seed)
    return options


def seed_data_worker(_worker_id: int) -> None:
    """Derive deterministic Python/NumPy worker seeds from PyTorch's seed."""
    import random
    import torch

    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


def recognizer_batch_sizes(physical_batch_size: int) -> dict[int, int]:
    return {
        160: max(1, physical_batch_size),
        320: max(1, physical_batch_size // 2),
        640: 1,
    }


def batch_candidates(requested: int, minimum: int = 1) -> list[int]:
    values: list[int] = []
    candidate = requested
    while candidate >= minimum:
        if candidate not in values:
            values.append(candidate)
        if candidate == minimum:
            break
        candidate = max(minimum, candidate // 2)
    return values


def accumulation_settings(physical_batch_size: int, target_effective_batch: int) -> tuple[int, int]:
    steps = max(1, math.ceil(target_effective_batch / physical_batch_size))
    return steps, physical_batch_size * steps


def checkpoint_epoch(path: Path) -> int:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if "epoch" not in state:
        raise ValueError(f"resume checkpoint has no epoch: {path}")
    return int(state["epoch"])


def epochs_for_run(*, requested_epochs: int, smoke: bool, resume: Path | None) -> int:
    if resume is None:
        return 1 if smoke else requested_epochs
    # fit() consumes an exclusive total target.  CLI --epochs intentionally
    # means *additional* epochs on resume, while smoke always adds exactly one.
    completed_epochs = checkpoint_epoch(resume) + 1
    return completed_epochs + (1 if smoke else requested_epochs)


def initial_learning_rate_details(
    model: Any, learning_rate: float, *, resume: Path | None,
    resume_policy: str,
) -> tuple[list[dict[str, Any]], str]:
    """Resolve the actual optimizer group LRs that training will start with."""

    if resume_policy not in {"restore", "reset"}:
        raise ValueError("resume_policy must be 'restore' or 'reset'")
    if resume is not None and resume_policy == "restore":
        import torch

        state = torch.load(resume, map_location="cpu", weights_only=False)
        optimizer_state = state.get("optimizer")
        if not optimizer_state or not optimizer_state.get("param_groups"):
            raise ValueError(f"resume checkpoint has no optimizer param groups: {resume}")
        current_groups = parameter_groups(model, learning_rate)
        layout = classify_optimizer_group_layout(
            current_groups, optimizer_state["param_groups"],
        )
        if layout == "legacy_consumed_parameter_generators":
            return [
                {
                    "name": str(group.get("name", f"group_{index}")),
                    "lr": float(group["lr"]),
                }
                for index, group in enumerate(current_groups)
            ], "cli_legacy_optimizer_layout_reset"
        if layout != "compatible":
            raise ValueError(
                f"resume checkpoint optimizer layout is incompatible with the current model: {resume}"
            )
        details = [
            {
                "name": str(group.get("name", f"group_{index}")),
                "lr": float(group["lr"]),
            }
            for index, group in enumerate(optimizer_state["param_groups"])
        ]
        return details, "checkpoint_optimizer"
    details = [
        {
            "name": str(group.get("name", f"group_{index}")),
            "lr": float(group["lr"]),
        }
        for index, group in enumerate(parameter_groups(model, learning_rate))
    ]
    return details, "cli_resume_reset" if resume is not None else "cli_new_run"


def probe_actual_batches(
    *,
    model_kind: str,
    model_factory: Callable[[], Any],
    loader_factory: Callable[[int, bool], Iterable[Mapping[str, Any]]],
    requested_batch_size: int,
    minimum_batch_size: int,
    iterations: int,
    learning_rate: float,
    device: Any,
    precision: str,
    autocast_dtype: Any,
    seed: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Probe real DataLoader batches and lower only physical batch after OOM."""

    import torch

    if device.type != "cuda":
        return requested_batch_size, {
            "status": "skipped_cuda_unavailable",
            "iterations_requested": iterations,
            "attempts": [],
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        }

    attempts: list[dict[str, Any]] = []
    for candidate in batch_candidates(requested_batch_size, minimum_batch_size):
        model = optimizer = scaler = loader = iterator = None
        try:
            if seed is not None:
                seed_everything(seed)
            model = model_factory().to(device)
            optimizer = torch.optim.AdamW(
                parameter_groups(model, learning_rate), weight_decay=1e-4,
            )
            scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
            loader = loader_factory(candidate, False)
            iterator = iter(loader)
            model.train()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            losses: list[float] = []
            for raw_batch in itertools.islice(iterator, iterations):
                optimizer.zero_grad(set_to_none=True)
                batch = move_to_device(raw_batch, device)
                with torch.autocast(
                    device_type="cuda", dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ):
                    loss = model_loss(model_kind, model, batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
            if not losses:
                raise ValueError("training DataLoader yielded no batches for CUDA dry-run")
            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1_000_000
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1_000_000
            attempts.append({
                "physical_batch_size": candidate,
                "status": "passed",
                "iterations_completed": len(losses),
                "losses": losses,
                "peak_allocated_mb": peak_allocated,
                "peak_reserved_mb": peak_reserved,
            })
            return candidate, {
                "status": "passed",
                "iterations_requested": iterations,
                "attempts": attempts,
                "peak_allocated_mb": peak_allocated,
                "peak_reserved_mb": peak_reserved,
            }
        except RuntimeError as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            attempts.append({
                "physical_batch_size": candidate,
                "status": "oom" if is_oom else "error",
                "error": str(exc).splitlines()[0],
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1_000_000,
                "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1_000_000,
            })
            if not is_oom:
                raise
            if candidate == minimum_batch_size:
                raise RuntimeError(
                    f"{model_kind} CUDA dry-run OOM at physical batch {candidate}; "
                    "image size was intentionally not reduced"
                ) from exc
        finally:
            del iterator, loader, scaler, optimizer, model
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f"no safe physical batch was found for {model_kind}")


def average_validation_loss(
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    *,
    model_kind: str,
    device: Any,
    autocast_dtype: Any,
) -> float:
    import torch

    model.eval()
    weighted_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_to_device(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                loss = model_loss(model_kind, model, batch)
            actual_batch_size = int(batch["image"].shape[0])
            if actual_batch_size < 1:
                raise ValueError("validation batch contains no images")
            weighted_loss += float(loss.detach()) * actual_batch_size
            sample_count += actual_batch_size
    if sample_count == 0:
        raise ValueError("validation DataLoader yielded no batches")
    return weighted_loss / sample_count


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    args = resolve_storage_paths(build_parser().parse_args())
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.resume is None and args.resume_lr_policy != "restore":
        raise ValueError("--resume-lr-policy reset requires --resume")
    if bool(args.objects) == bool(args.shard_dir):
        raise ValueError("provide exactly one of --shard-dir or --objects")
    if args.phase == "real_finetune" and args.resume is None:
        raise ValueError("real_finetune requires --resume from synthetic pre-training")
    seed_everything(args.seed)

    master = MasterSplit.load(args.master_split)
    lazy_parquet = bool(args.shard_dir)
    training_domain = "real" if args.phase == "real_finetune" else "synthetic"
    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    train_entries = []
    validation_entries = []
    validation_domain: str | None = training_domain if args.cv_fold is not None else None
    if lazy_parquet:
        shard_dirs = [path.resolve() for path in args.shard_dir]
        if args.cv_fold is not None:
            train_entries = load_parquet_image_entries(
                shard_dirs, master, purpose="cv", source_domain=training_domain,
                exclude_cv_fold=args.cv_fold,
            )
            validation_entries = load_parquet_image_entries(
                shard_dirs, master, purpose="cv", source_domain=training_domain,
                include_cv_fold=args.cv_fold,
            )
        else:
            train_entries = load_parquet_image_entries(
                shard_dirs, master, purpose="train", source_domain=training_domain,
            )
            real_validation_entries = load_parquet_image_entries(
                shard_dirs, master, purpose="select", source_domain="real",
            )
            if real_validation_entries:
                validation_entries = real_validation_entries
                validation_domain = "real"
            else:
                validation_entries = load_parquet_image_entries(
                    shard_dirs, master, purpose="select", source_domain="synthetic",
                )
                validation_domain = "synthetic"
    else:
        train_records = [
            item for path in args.objects
            for item in load_records(path, master, purpose="train")
        ]
        train_records = [
            item for item in train_records
            if str(item.get("source_domain") or "synthetic") == training_domain
        ]
        if args.cv_fold is not None:
            train_records, validation_records = train_cv_partition(train_records, args.cv_fold)
        else:
            all_validation_records = [
                item for path in args.objects
                for item in load_records(path, master, purpose="select")
            ]
            real_validation_records = [
                item for item in all_validation_records if item.get("source_domain") == "real"
            ]
            validation_records = real_validation_records or all_validation_records
            validation_domain = "real" if real_validation_records else "synthetic"
    if not (train_entries if lazy_parquet else train_records):
        raise ValueError(f"no {args.phase} Train records were found")
    if not (validation_entries if lazy_parquet else validation_records):
        raise ValueError("no authorized Validation records were found")
    if args.phase == "real_finetune" and args.cv_fold is None and validation_domain != "real":
        raise ValueError("real_finetune checkpoint selection requires real Validation records")

    charset = load_charset(args.charset)
    table_predictions = {}
    if args.table_predictions:
        from shift_ocr.shards import iter_jsonl
        for item in iter_jsonl(args.table_predictions):
            table_predictions[(str(item["schedule_id"]), str(item.get("row_id")), int(item.get("day") or 0))] = item

    device, precision, autocast_dtype = select_device_and_precision()
    num_workers, worker_source = resolve_num_workers(args.num_workers)
    cuda = device.type == "cuda"
    model_factory = lambda: build_model(
        args.model, class_count=len(charset) + 1,
        attention=args.attention, top_k=2048,
    )

    if args.model == "recognizer":
        if lazy_parquet:
            dataset = LazyParquetRecognitionCropDataset(
                shard_dirs, master, args.image_root, purpose="cv" if args.cv_fold is not None else "train",
                charset=charset, training=True,
                table_prediction_records=table_predictions, seed=args.seed,
            )

            def training_loader_factory(batch_size: int, persistent: bool):
                sampler = ParquetRecognitionBatchSampler(
                    shard_dirs, master,
                    purpose="cv" if args.cv_fold is not None else "train",
                    batch_sizes=recognizer_batch_sizes(batch_size), training=True,
                    source_domain=training_domain, exclude_cv_fold=args.cv_fold,
                    seed=args.seed,
                )
                return DataLoader(
                    dataset, batch_sampler=sampler, collate_fn=recognition_collate,
                    **dataloader_options(
                        num_workers, cuda=cuda, persistent=persistent, seed=args.seed,
                    ),
                )
        else:
            dataset = RecognitionCropDataset(
                train_records, args.image_root, charset=charset, training=True,
                table_prediction_records=table_predictions, seed=args.seed,
            )
            if not dataset.records:
                raise ValueError("no shift-code/name objects were found for recognizer training")
            epoch_samples = min(
                len(dataset.records),
                len({str(item["schedule_id"]) for item in dataset.records}) * 128,
            )
            oversampled_indices = RareCodeCropSampler(
                dataset.records, rare_weight=3.0, max_per_schedule=128, seed=args.seed,
            ).sample_indices(epoch_samples)

            def training_loader_factory(batch_size: int, persistent: bool):
                sampler = WidthBucketBatchSampler(
                    dataset, recognizer_batch_sizes(batch_size), indices=oversampled_indices,
                    seed=args.seed,
                )
                return DataLoader(
                    dataset, batch_sampler=sampler, collate_fn=recognition_collate,
                    **dataloader_options(
                        num_workers, cuda=cuda, persistent=persistent, seed=args.seed,
                    ),
                )

    else:
        if lazy_parquet:
            dataset = LazyParquetDenseScheduleDataset(
                train_entries, shard_dirs, master, args.image_root,
                purpose="cv" if args.cv_fold is not None else "train", kind=args.model,
                training=True, long_side=args.image_size, seed=args.seed,
            )
        else:
            dataset = DenseScheduleDataset(
                train_records, args.image_root, kind=args.model,
                training=True, long_side=args.image_size, seed=args.seed,
            )

        def training_loader_factory(batch_size: int, persistent: bool):
            return DataLoader(
                dataset, batch_size=max(1, batch_size), shuffle=True,
                **dataloader_options(
                    num_workers, cuda=cuda, persistent=persistent, seed=args.seed,
                ),
            )

    selected_batch_size, dry_run = probe_actual_batches(
        model_kind=args.model,
        model_factory=model_factory,
        loader_factory=training_loader_factory,
        requested_batch_size=args.batch_size,
        minimum_batch_size=1,
        iterations=args.dry_run_iterations,
        learning_rate=args.learning_rate,
        device=device,
        precision=precision,
        autocast_dtype=autocast_dtype,
        seed=args.seed,
    )

    # The probe creates a model, advances loader generators and consumes CUDA
    # RNG.  Reset all seeds so probing cannot alter the actual run.
    seed_everything(args.seed)
    loader = training_loader_factory(selected_batch_size, True)
    model = model_factory()
    initial_group_lrs, learning_rate_source = initial_learning_rate_details(
        model, args.learning_rate, resume=args.resume,
        resume_policy=args.resume_lr_policy,
    )
    if args.model == "recognizer":
        if lazy_parquet:
            validation_dataset = LazyParquetRecognitionCropDataset(
                shard_dirs, master, args.image_root,
                purpose="cv" if args.cv_fold is not None else "select",
                charset=charset, training=False, seed=args.seed,
            )
            validation_sampler = ParquetRecognitionBatchSampler(
                shard_dirs, master,
                purpose="cv" if args.cv_fold is not None else "select",
                batch_sizes=recognizer_batch_sizes(selected_batch_size), training=False,
                source_domain=validation_domain, include_cv_fold=args.cv_fold,
                shuffle=False, seed=args.seed,
            )
        else:
            validation_dataset = RecognitionCropDataset(
                validation_records, args.image_root, charset=charset, training=False,
                seed=args.seed,
            )
            validation_sampler = WidthBucketBatchSampler(
                validation_dataset, recognizer_batch_sizes(selected_batch_size), shuffle=False,
                seed=args.seed,
            )
        validation_loader = DataLoader(
            validation_dataset, batch_sampler=validation_sampler,
            collate_fn=recognition_collate,
            **dataloader_options(
                num_workers, cuda=cuda, persistent=True, seed=args.seed + 1,
            ),
        )
    else:
        if lazy_parquet:
            validation_dataset = LazyParquetDenseScheduleDataset(
                validation_entries, shard_dirs, master, args.image_root,
                purpose="cv" if args.cv_fold is not None else "select", kind=args.model,
                training=False, long_side=args.image_size, seed=args.seed,
            )
        else:
            validation_dataset = DenseScheduleDataset(
                validation_records, args.image_root, kind=args.model,
                training=False, long_side=args.image_size, seed=args.seed,
            )
        validation_loader = DataLoader(
            validation_dataset, batch_size=1, shuffle=False,
            **dataloader_options(
                num_workers, cuda=cuda, persistent=True, seed=args.seed + 1,
            ),
        )

    if args.smoke:
        loader = LazyBatchLimit(loader, 2)
        validation_loader = LazyBatchLimit(validation_loader, 2)

    def validation(network):
        validation_device = next(network.parameters()).device
        if args.model == "dbnet":
            metrics = dict(validate_dbnet(network, validation_loader, validation_device))
        elif args.model == "recognizer":
            metrics = dict(validate_recognizer(network, validation_loader, charset, validation_device))
        else:
            metrics = dict(validate_table(network, validation_loader, validation_device))
        metrics["validation_loss"] = average_validation_loss(
            network, validation_loader, model_kind=args.model,
            device=validation_device, autocast_dtype=autocast_dtype,
        )
        return metrics

    target_total_epochs = epochs_for_run(
        requested_epochs=args.epochs, smoke=args.smoke, resume=args.resume,
    )
    validation_scope = "train_cv_fold" if args.cv_fold is not None else (
        "real_validation" if validation_domain == "real" else "synthetic_validation"
    )
    selection_scope = (
        f"train_cv_fold:{args.cv_fold}" if args.cv_fold is not None else validation_scope
    )
    config = TrainConfig(
        model_kind=args.model,
        epochs=target_total_epochs,
        learning_rate=args.learning_rate,
        requested_batch_size=selected_batch_size,
        target_effective_batch=args.effective_batch_size,
        seed=args.seed,
        resume_learning_rate_policy=args.resume_lr_policy,
        training_phase=args.phase,
        selection_scope=selection_scope,
        resume_selection_policy="reset" if args.reset_selection_state else "auto",
    )
    selection_state = preview_resume_selection(args.resume, config)
    accumulation_steps, nominal_effective_batch_size = accumulation_settings(
        selected_batch_size, args.effective_batch_size,
    )
    recognizer_accumulation = None
    if args.model == "recognizer":
        recognizer_accumulation = {
            str(width): {
                "physical_batch_size": bucket_batch_size,
                "micro_batches_to_reach_target_if_bucket_is_repeated": math.ceil(
                    args.effective_batch_size / bucket_batch_size
                ),
                "nominal_samples_at_step_if_bucket_is_repeated": (
                    math.ceil(args.effective_batch_size / bucket_batch_size) * bucket_batch_size
                ),
            }
            for width, bucket_batch_size in recognizer_batch_sizes(selected_batch_size).items()
        }
    reasons = [
        f"precision={precision} was selected from CUDA capability",
        f"num_workers={num_workers} source={worker_source}",
        f"learning-rate source={learning_rate_source}",
    ]
    if args.model == "recognizer":
        reasons.append("recognizer kept 48px height and 160/320/640 width-bucket micro-batches")
    else:
        reasons.append(f"dense image long side remained fixed at {args.image_size}px during every probe")
    if selected_batch_size < args.batch_size:
        reasons.append(
            f"CUDA OOM reduced physical batch from {args.batch_size} to {selected_batch_size}; "
            "image size was not reduced"
        )
    else:
        reasons.append(f"requested physical batch {args.batch_size} passed the real-batch CUDA probe")
    reasons.append(
        "gradient accumulation is sample-aware: each optimizer step averages the actual "
        f"micro-batch samples after reaching target={args.effective_batch_size}"
    )

    runtime_settings: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "storage_root": str(args.storage_root),
        "resolved_paths": {
            "master_split": str(args.master_split),
            "image_root": str(args.image_root),
            "output_dir": str(args.output_dir),
            "sources": args._path_sources,
        },
        "model": args.model,
        "table_candidate_top_k": 2048 if args.model == "table" else None,
        "data_loading_mode": "lazy_parquet_image_index" if lazy_parquet else "legacy_jsonl_memory",
        "shard_dirs": [str(path) for path in shard_dirs] if lazy_parquet else None,
        "train_image_count": len(train_entries) if lazy_parquet else len({
            str(item["image_path"]) for item in train_records
        }),
        "validation_image_count": len(validation_entries) if lazy_parquet else len({
            str(item["image_path"]) for item in validation_records
        }),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
        "gpu_total_vram_mb": (
            torch.cuda.get_device_properties(device).total_memory / 1_000_000 if cuda else None
        ),
        "precision": precision,
        "dense_image_long_side": args.image_size if args.model != "recognizer" else None,
        "recognizer_image_height": 48 if args.model == "recognizer" else None,
        "recognizer_width_buckets": [160, 320, 640] if args.model == "recognizer" else None,
        "requested_physical_batch_size": args.batch_size,
        "selected_physical_batch_size": selected_batch_size,
        "recognizer_physical_batch_sizes": (
            recognizer_batch_sizes(selected_batch_size) if args.model == "recognizer" else None
        ),
        "target_effective_batch_size": args.effective_batch_size,
        "gradient_accumulation_mode": "sample_aware",
        "gradient_accumulation_steps": (
            None if args.model == "recognizer" else accumulation_steps
        ),
        "nominal_effective_batch_size": (
            None if args.model == "recognizer" else nominal_effective_batch_size
        ),
        "recognizer_bucket_accumulation": recognizer_accumulation,
        "final_partial_group_is_sample_averaged": True,
        "learning_rate": args.learning_rate,
        "learning_rate_source": learning_rate_source,
        "resume_learning_rate_policy": args.resume_lr_policy,
        "actual_initial_group_learning_rates": initial_group_lrs,
        "training_phase": args.phase,
        "validation_scope": validation_scope,
        "selection_scope": selection_scope,
        "selection_state": selection_state,
        "seed": args.seed,
        "requested_epochs": args.epochs,
        "resume_epochs_semantics": "additional" if args.resume else "new_run_total",
        "target_total_epochs": target_total_epochs,
        "num_workers": num_workers,
        "num_workers_source": worker_source,
        "persistent_workers": num_workers > 0,
        "dry_run": dry_run,
        "smoke": args.smoke,
        "smoke_train_batch_limit": 2 if args.smoke else None,
        "smoke_validation_batch_limit": 2 if args.smoke else None,
        "reasons": reasons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = args.output_dir / "runtime_settings.json"
    runtime_path.write_text(json.dumps(runtime_settings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runtime_settings": runtime_settings}, ensure_ascii=False, indent=2), flush=True)

    result = fit(model, loader, validation, config, args.output_dir, resume=args.resume)
    result["phase"] = args.phase
    result["cv_fold"] = args.cv_fold
    result["validation_scope"] = validation_scope
    result["runtime_settings"] = runtime_settings
    learning_rate_state = result.get("learning_rate_state", {})
    runtime_settings["learning_rate_source"] = learning_rate_state.get(
        "source", runtime_settings["learning_rate_source"],
    )
    runtime_settings["actual_initial_group_learning_rates"] = learning_rate_state.get(
        "initial_group_learning_rates", runtime_settings["actual_initial_group_learning_rates"],
    )
    runtime_settings["actual_final_group_learning_rates"] = learning_rate_state.get(
        "actual_group_learning_rates"
    )
    runtime_settings["selection_state"] = result.get("selection_state")
    runtime_path.write_text(
        json.dumps(runtime_settings, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Required for Windows DataLoader spawn workers.
    main()
