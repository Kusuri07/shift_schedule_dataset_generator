"""Mixed-precision training, dry-run batch search, gradual unfreezing and resume."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


BEST_METRICS = {
    "dbnet": {"primary": "text_polygon_hmean_iou_0_5", "mode": "max"},
    "recognizer": {"primary": "cell_exact_accuracy", "mode": "max", "tie_breaker": "cer", "tie_mode": "min"},
    "table": {"primary": "table_composite", "formula": "0.6*cell_polygon_f1+0.4*row_level_accuracy", "mode": "max"},
}


@dataclass
class TrainConfig:
    model_kind: str
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    requested_batch_size: int = 16
    min_batch_size: int = 1
    target_effective_batch: int = 32
    checkpoint_every: int = 5
    freeze_epochs: int = 3
    partial_unfreeze_epoch: int = 4
    full_unfreeze_epoch: int = 8
    seed: int = 20260723


def select_device_and_precision():
    import torch

    if not torch.cuda.is_available():
        return torch.device("cpu"), "fp32", None
    device = torch.device("cuda")
    if torch.cuda.is_bf16_supported():
        return device, "bf16", torch.bfloat16
    return device, "fp16", torch.float16


def dry_run_batch_search(
    model, batch_factory: Callable[[int], Any], loss_function: Callable[[Any, Any], Any],
    *, requested: int, minimum: int, device,
) -> int:
    import torch

    candidate = requested
    while candidate >= minimum:
        try:
            model.zero_grad(set_to_none=True)
            batch = batch_factory(candidate)
            loss = loss_function(model, move_to_device(batch, device))
            loss.backward()
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return candidate
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or candidate == minimum:
                raise
            candidate = max(minimum, candidate // 2)
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return minimum


def move_to_device(value, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def model_loss(kind: str, model, batch):
    import torch
    import torch.nn.functional as functional

    outputs = model(batch["image"])
    if kind == "dbnet":
        probability = functional.interpolate(outputs["probability"], size=batch["probability_target"].shape[-2:], mode="bilinear", align_corners=False)
        threshold = functional.interpolate(outputs["threshold"], size=batch["threshold_target"].shape[-2:], mode="bilinear", align_corners=False)
        return functional.binary_cross_entropy(probability, batch["probability_target"]) + 0.5 * functional.l1_loss(threshold, batch["threshold_target"])
    if kind == "recognizer":
        logits = outputs.transpose(0, 1)
        input_lengths = torch.full((logits.shape[1],), logits.shape[0], dtype=torch.long, device=logits.device)
        return functional.ctc_loss(logits, batch["labels"], input_lengths, batch["label_lengths"], blank=0, zero_infinity=True)
    heatmap = functional.interpolate(outputs["cell_heatmap"], size=batch["cell_heatmap_target"].shape[-2:], mode="bilinear", align_corners=False)
    corners = functional.interpolate(outputs["corner_offsets"], size=batch["corner_target"].shape[-2:], mode="bilinear", align_corners=False)
    heatmap_loss = functional.binary_cross_entropy_with_logits(heatmap, batch["cell_heatmap_target"])
    valid = batch["corner_valid"].expand_as(corners)
    corner_loss = ((corners - batch["corner_target"]).abs() * valid).sum() / valid.sum().clamp_min(1)
    return heatmap_loss + corner_loss


def parameter_groups(model, base_learning_rate: float):
    groups = []
    if hasattr(model, "backbone"):
        groups.append({"params": model.backbone.parameters(), "lr": base_learning_rate * 0.25, "name": "backbone"})
    if hasattr(model, "fpn"):
        groups.append({"params": model.fpn.parameters(), "lr": base_learning_rate * 0.5, "name": "neck"})
    assigned = {id(parameter) for group in groups for parameter in group["params"]}
    head = [parameter for parameter in model.parameters() if id(parameter) not in assigned]
    if head:
        groups.append({"params": head, "lr": base_learning_rate, "name": "head"})
    return groups


def apply_unfreezing(model, epoch: int, config: TrainConfig) -> str:
    if not hasattr(model, "backbone"):
        return "all"
    for parameter in model.backbone.parameters():
        parameter.requires_grad = epoch >= config.full_unfreeze_epoch
    if epoch >= config.partial_unfreeze_epoch and hasattr(model.backbone, "stage16"):
        for parameter in model.backbone.stage16.parameters():
            parameter.requires_grad = True
    if epoch < config.partial_unfreeze_epoch:
        state = "backbone_frozen"
    elif epoch < config.full_unfreeze_epoch:
        state = "last_backbone_stage"
    else:
        state = "fully_unfrozen"
    return state


def is_better(kind: str, metrics: Mapping[str, float], best: Mapping[str, float] | None) -> bool:
    if best is None:
        return True
    spec = BEST_METRICS[kind]
    primary = spec["primary"]
    value, previous = float(metrics[primary]), float(best[primary])
    if not math.isclose(value, previous, abs_tol=1e-12):
        return value > previous if spec["mode"] == "max" else value < previous
    tie = spec.get("tie_breaker")
    return bool(tie and float(metrics[tie]) < float(best[tie]))


class CheckpointManager:
    def __init__(self, directory: Path, kind: str) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.kind = kind

    def save(self, name: str, model, optimizer, scaler, epoch: int, metrics: Mapping[str, float], config: TrainConfig):
        import torch

        path = self.directory / f"{name}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "metrics": dict(metrics),
            "config": asdict(config),
            "rng_python": random.getstate(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, path)
        return path

    def load(self, path: Path, model, optimizer=None, scaler=None, device="cpu") -> dict[str, Any]:
        import torch

        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        random.setstate(state["rng_python"])
        torch.set_rng_state(state["rng_torch"])
        if torch.cuda.is_available() and state.get("rng_cuda"):
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        return state


def fit(
    model, train_loader, validation_function: Callable[[Any], Mapping[str, float]],
    config: TrainConfig, checkpoint_dir: Path, *, resume: Path | None = None,
) -> dict[str, Any]:
    import torch

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device, precision, autocast_dtype = select_device_and_precision()
    model.to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, config.learning_rate), weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    manager = CheckpointManager(checkpoint_dir, config.model_kind)
    start_epoch = 0
    best_metrics = None
    if resume:
        state = manager.load(resume, model, optimizer, scaler, device)
        start_epoch = int(state["epoch"]) + 1
        best_metrics = state.get("metrics")
    accumulation = max(1, math.ceil(config.target_effective_batch / max(1, config.requested_batch_size)))
    history = []
    for epoch in range(start_epoch, config.epochs):
        unfreeze_state = apply_unfreezing(model, epoch, config)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        step_count = 0
        for step, raw_batch in enumerate(train_loader):
            batch = move_to_device(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                loss = model_loss(config.model_kind, model, batch) / accumulation
            scaler.scale(loss).backward()
            if (step + 1) % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach()) * accumulation
            step_count += 1
        metrics = dict(validation_function(model))
        if config.model_kind == "table":
            metrics["table_composite"] = 0.6 * float(metrics["cell_polygon_f1"]) + 0.4 * float(metrics["row_level_accuracy"])
        metrics.update({"train_loss": running_loss / max(1, step_count), "epoch": epoch, "unfreeze_state": unfreeze_state})
        manager.save("last", model, optimizer, scaler, epoch, metrics, config)
        if is_better(config.model_kind, metrics, best_metrics):
            best_metrics = dict(metrics)
            manager.save("best", model, optimizer, scaler, epoch, metrics, config)
        if (epoch + 1) % config.checkpoint_every == 0:
            manager.save(f"epoch_{epoch + 1:04d}", model, optimizer, scaler, epoch, metrics, config)
        history.append(metrics)
    manifest = {
        "model_kind": config.model_kind,
        "best_metric_spec": BEST_METRICS[config.model_kind],
        "best_metrics": best_metrics,
        "precision": precision,
        "device": str(device),
        "history": history,
    }
    (checkpoint_dir / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
