#!/usr/bin/env python3
"""Train DBNet, recognizer or table model without artifact-tool dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shift_ocr.charset import load_charset
from shift_ocr.datasets import (
    DenseScheduleDataset, RecognitionCropDataset, WidthBucketBatchSampler,
    RareCodeCropSampler, load_records, recognition_collate, train_cv_partition,
)
from shift_ocr.master_split import MasterSplit
from shift_ocr.models import build_model
from shift_ocr.training import TrainConfig, dry_run_batch_search, fit, model_loss, select_device_and_precision
from shift_ocr.validation import validate_dbnet, validate_recognizer, validate_table


def main() -> None:
    import torch
    from torch.utils.data import DataLoader, Subset

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dbnet", "recognizer", "table"])
    parser.add_argument("--objects", required=True, type=Path, action="append", help="Repeat for synthetic and registered-real JSONL")
    parser.add_argument("--master-split", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--charset", type=Path, default=Path("data/korean_charset_v1.txt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--cv-fold", type=int, choices=[0, 1, 2], help="Train-only grouped CV fold")
    parser.add_argument("--phase", choices=["synthetic_pretrain", "real_finetune"], default="synthetic_pretrain")
    parser.add_argument("--table-predictions", type=Path, help="JSONL predicted cell polygons with confidence and gt_iou")
    parser.add_argument("--smoke", action="store_true", help="Run only two mini-batches and one epoch")
    args = parser.parse_args()

    master = MasterSplit.load(args.master_split)
    train_records = [item for path in args.objects for item in load_records(path, master, purpose="train")]
    if args.phase == "real_finetune":
        if args.resume is None:
            raise ValueError("real_finetune requires --resume from synthetic pre-training")
        train_records = [item for item in train_records if item.get("source_domain") == "real"]
    else:
        train_records = [item for item in train_records if item.get("source_domain", "synthetic") == "synthetic"]
    if args.cv_fold is not None:
        train_records, validation_records = train_cv_partition(train_records, args.cv_fold)
    else:
        all_validation_records = [item for path in args.objects for item in load_records(path, master, purpose="select")]
        real_validation_records = [item for item in all_validation_records if item.get("source_domain") == "real"]
        validation_records = real_validation_records or all_validation_records
        if args.phase == "real_finetune" and not real_validation_records:
            raise ValueError("real_finetune checkpoint selection requires real Validation records")
    if not train_records:
        raise ValueError(f"no {args.phase} Train records were found")
    charset = load_charset(args.charset)
    table_predictions = {}
    if args.table_predictions:
        from shift_ocr.shards import iter_jsonl
        for item in iter_jsonl(args.table_predictions):
            table_predictions[(str(item["schedule_id"]), str(item.get("row_id")), int(item.get("day") or 0))] = item
    model = build_model(args.model, class_count=len(charset) + 1, attention=args.attention, top_k=1500)
    device, _precision, _dtype = select_device_and_precision()
    selected_batch_size = args.batch_size
    if device.type == "cuda":
        model.to(device)
        def probe_batch(batch_size):
            width = 320 if args.model == "recognizer" else 1280
            height = 48 if args.model == "recognizer" else 1280
            batch = {"image": torch.zeros(batch_size, 3, height, width)}
            if args.model == "recognizer":
                batch.update({
                    "labels": torch.ones(batch_size, dtype=torch.long),
                    "label_lengths": torch.ones(batch_size, dtype=torch.long),
                })
            else:
                target = torch.zeros(batch_size, 1, height // 4, width // 4)
                batch.update({"probability_target": target, "threshold_target": target})
                if args.model == "table":
                    batch.update({
                        "cell_heatmap_target": target,
                        "corner_target": torch.zeros(batch_size, 8, height // 4, width // 4),
                        "corner_valid": target,
                    })
            return batch
        selected_batch_size = dry_run_batch_search(
            model, probe, lambda network, batch: model_loss(args.model, network, batch),
            requested=args.batch_size, minimum=1, device=device,
        )
    if args.model == "recognizer":
        dataset = RecognitionCropDataset(
            train_records, args.image_root, charset=charset, training=True,
            table_prediction_records=table_predictions,
        )
        epoch_samples = min(
            len(dataset.records),
            len({str(item["schedule_id"]) for item in dataset.records}) * 128,
        )
        oversampled_indices = RareCodeCropSampler(
            dataset.records, rare_weight=3.0, max_per_schedule=128, seed=20260723
        ).sample_indices(epoch_samples)
        sampler = WidthBucketBatchSampler(
            dataset, {160: selected_batch_size, 320: max(1, selected_batch_size // 2), 640: 1},
            indices=oversampled_indices,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=recognition_collate, num_workers=0)
        validation_dataset = RecognitionCropDataset(validation_records, args.image_root, charset=charset, training=False)
        validation_sampler = WidthBucketBatchSampler(validation_dataset, {160: selected_batch_size, 320: max(1, selected_batch_size // 2), 640: 1}, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, collate_fn=recognition_collate, num_workers=0)
    else:
        dataset = DenseScheduleDataset(train_records, args.image_root, kind=args.model, training=True)
        loader = DataLoader(dataset, batch_size=max(1, selected_batch_size), shuffle=True, num_workers=0)
        validation_dataset = DenseScheduleDataset(validation_records, args.image_root, kind=args.model, training=False)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)
    if args.smoke:
        loader = list(loader)[:2]

    def validation(_model):
        device = next(_model.parameters()).device
        if args.model == "dbnet":
            return validate_dbnet(_model, validation_loader, device)
        if args.model == "recognizer":
            return validate_recognizer(_model, validation_loader, charset, device)
        return validate_table(_model, validation_loader, device)

    config = TrainConfig(
        model_kind=args.model,
        epochs=1 if args.smoke else args.epochs,
        requested_batch_size=selected_batch_size,
    )
    result = fit(model, loader, validation, config, args.output_dir, resume=args.resume)
    result["phase"] = args.phase
    result["cv_fold"] = args.cv_fold
    result["validation_scope"] = "train_cv_fold" if args.cv_fold is not None else (
        "real_validation" if any(item.get("source_domain") == "real" for item in validation_records) else "synthetic_validation"
    )
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
