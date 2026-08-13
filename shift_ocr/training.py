"""Mixed-precision training, dry-run batch search, gradual unfreezing and resume."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
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
    scheduler_min_learning_rate: float = 1e-6
    freeze_epochs: int = 3
    partial_unfreeze_epoch: int = 4
    full_unfreeze_epoch: int = 8
    seed: int = 20260723
    resume_learning_rate_policy: str = "restore"
    training_phase: str = "synthetic_pretrain"
    selection_scope: str = "synthetic_validation"
    resume_selection_policy: str = "auto"


def seed_everything(seed: int) -> None:
    """Seed the parent process before model creation, probing, or training."""
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass


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


def _relation_embedding_loss(embedding, labels, valid):
    """Pull equal relations together and push distinct relation centroids apart.

    Dense pairwise comparisons between every table cell become unnecessarily
    expensive for large schedules.  This discriminative formulation compares
    cells with their relation centroid and then compares the (much smaller)
    set of centroids.  It is evaluated independently for each image so row and
    column identifiers do not leak across schedules.
    """
    import torch

    losses = []
    embedding = embedding.float()
    labels = labels.long()
    valid = valid.bool()
    for sample_embedding, sample_labels, sample_valid in zip(embedding, labels, valid):
        if not torch.any(sample_valid):
            continue
        vectors = sample_embedding[:, sample_valid].transpose(0, 1)
        targets = sample_labels[sample_valid]
        identifiers = torch.unique(targets, sorted=False)
        centroids = torch.stack([vectors[targets == identifier].mean(dim=0) for identifier in identifiers])
        # searchsorted needs sorted centroids, whereas identifiers are not
        # guaranteed to be ordered when ``sorted=False``.
        sorted_identifiers, order = torch.sort(identifiers)
        assigned = torch.searchsorted(sorted_identifiers, targets)
        sorted_centroids = centroids[order]
        pull = (vectors - sorted_centroids[assigned]).pow(2).sum(dim=1).mean()
        if len(sorted_centroids) > 1:
            distances = torch.cdist(sorted_centroids, sorted_centroids, p=2)
            off_diagonal = ~torch.eye(len(sorted_centroids), dtype=torch.bool, device=distances.device)
            push = torch.relu(1.0 - distances[off_diagonal]).pow(2).mean()
        else:
            push = vectors.sum() * 0.0
        losses.append(pull + push)
    if not losses:
        return embedding.sum() * 0.0
    return torch.stack(losses).mean()


def model_loss(kind: str, model, batch):
    import torch
    import torch.nn.functional as functional

    outputs = model(batch["image"])
    if kind == "dbnet":
        # BCE on sigmoid probabilities is explicitly unsafe under CUDA AMP.
        # Keep the network under autocast, but evaluate the numerically
        # sensitive interpolation and loss in FP32.
        with torch.autocast(device_type=outputs["probability"].device.type, enabled=False):
            probability = functional.interpolate(
                outputs["probability"].float(), size=batch["probability_target"].shape[-2:],
                mode="bilinear", align_corners=False,
            ).clamp(1e-6, 1.0 - 1e-6)
            threshold = functional.interpolate(
                outputs["threshold"].float(), size=batch["threshold_target"].shape[-2:],
                mode="bilinear", align_corners=False,
            )
            return functional.binary_cross_entropy(probability, batch["probability_target"].float()) + 0.5 * functional.l1_loss(
                threshold, batch["threshold_target"].float()
            )
    if kind == "recognizer":
        logits = outputs.transpose(0, 1)
        input_lengths = torch.full((logits.shape[1],), logits.shape[0], dtype=torch.long, device=logits.device)
        return functional.ctc_loss(logits, batch["labels"], input_lengths, batch["label_lengths"], blank=0, zero_infinity=True)
    heatmap = functional.interpolate(outputs["cell_heatmap"], size=batch["cell_heatmap_target"].shape[-2:], mode="bilinear", align_corners=False)
    corners = functional.interpolate(outputs["corner_offsets"], size=batch["corner_target"].shape[-2:], mode="bilinear", align_corners=False)
    heatmap_loss = functional.binary_cross_entropy_with_logits(heatmap, batch["cell_heatmap_target"])
    valid = batch["corner_valid"].expand_as(corners)
    corner_loss = ((corners - batch["corner_target"]).abs() * valid).sum() / valid.sum().clamp_min(1)
    relation_loss = heatmap.sum() * 0.0
    if "relation_target" in batch and "row_embedding" in outputs and "column_embedding" in outputs:
        target_size = batch["relation_target"].shape[-2:]
        row_embedding = functional.interpolate(outputs["row_embedding"], size=target_size, mode="bilinear", align_corners=False)
        column_embedding = functional.interpolate(outputs["column_embedding"], size=target_size, mode="bilinear", align_corners=False)
        relation_valid = batch["corner_valid"][:, 0] > 0.5
        row_loss = _relation_embedding_loss(row_embedding, batch["relation_target"][:, 0], relation_valid)
        column_loss = _relation_embedding_loss(column_embedding, batch["relation_target"][:, 1], relation_valid)
        relation_loss = 0.1 * (row_loss + column_loss) / 2.0
    return heatmap_loss + corner_loss + relation_loss


def parameter_groups(model, base_learning_rate: float):
    groups = []
    if hasattr(model, "backbone"):
        # ``Module.parameters()`` is a one-shot generator.  Materialize it so
        # building ``assigned`` below cannot consume the same iterator that is
        # later handed to AdamW (which would silently make this group empty).
        backbone_parameters = list(model.backbone.parameters())
        groups.append({"params": backbone_parameters, "lr": base_learning_rate * 0.25, "name": "backbone"})
    if hasattr(model, "fpn"):
        neck_parameters = list(model.fpn.parameters())
        groups.append({"params": neck_parameters, "lr": base_learning_rate * 0.5, "name": "neck"})
    assigned = {id(parameter) for group in groups for parameter in group["params"]}
    head = [parameter for parameter in model.parameters() if id(parameter) not in assigned]
    if head:
        groups.append({"params": head, "lr": base_learning_rate, "name": "head"})
    return groups


def classify_optimizer_group_layout(current_groups, checkpoint_groups) -> str:
    """Classify checkpoint optimizer layout without accepting unknown damage."""

    current_sizes = [len(group.get("params", [])) for group in current_groups]
    checkpoint_sizes = [len(group.get("params", [])) for group in checkpoint_groups]
    if current_sizes == checkpoint_sizes:
        return "compatible"
    if len(current_sizes) == len(checkpoint_sizes) and all(
        saved == current or (saved == 0 and current > 0)
        for saved, current in zip(checkpoint_sizes, current_sizes)
    ) and any(saved == 0 and current > 0 for saved, current in zip(checkpoint_sizes, current_sizes)):
        # Checkpoints produced before the parameter-generator fix contain
        # empty backbone/FPN groups but a valid model state.  Their optimizer
        # moments cannot be reconstructed, so reset only optimizer-related
        # state while keeping model weights and absolute epoch.
        return "legacy_consumed_parameter_generators"
    return "incompatible"


def apply_unfreezing(model, epoch: int, config: TrainConfig) -> str:
    if not hasattr(model, "backbone"):
        return "all"
    if config.training_phase == "synthetic_pretrain":
        # Synthetic pre-training learns the entire representation from
        # scratch.  The gradual freeze schedule belongs exclusively to real
        # photo fine-tuning and must never leak into this phase.
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        return "synthetic_all_trainable"
    if config.training_phase != "real_finetune":
        raise ValueError(f"unsupported training phase: {config.training_phase}")
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

    def save(
        self, name: str, model, optimizer, scaler, epoch: int, metrics: Mapping[str, Any],
        config: TrainConfig, *, scheduler=None, best_metrics: Mapping[str, Any] | None = None,
        history: Iterable[Mapping[str, Any]] | None = None,
    ):
        import torch

        path = self.directory / f"{name}.pt"
        torch.save({
            "checkpoint_version": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scheduler_name": type(scheduler).__name__ if scheduler is not None else None,
            "epoch": epoch,
            # Data order is derived from the absolute epoch rather than from
            # how many times a freshly-created loader has been iterated.  The
            # value is explicit so checkpoint inspection can confirm which
            # epoch a resumed loader must prepare next.
            "next_data_epoch": epoch + 1,
            "metrics": dict(metrics),
            "best_metrics": dict(best_metrics) if best_metrics is not None else None,
            "history": [dict(item) for item in history] if history is not None else None,
            "selection_state": {
                "training_phase": config.training_phase,
                "selection_scope": config.selection_scope,
            },
            "config": asdict(config),
            "rng_python": random.getstate(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, path)
        return path

    def load(
        self, path: Path, model, optimizer=None, scaler=None, device="cpu", scheduler=None,
        *, restore_optimizer: bool = True, restore_scaler: bool = True,
        restore_scheduler: bool = True,
    ) -> dict[str, Any]:
        import torch

        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer_restore = "not_requested"
        legacy_layout_reset = False
        if restore_optimizer and optimizer is not None and state.get("optimizer") is not None:
            layout = classify_optimizer_group_layout(
                optimizer.param_groups, state["optimizer"].get("param_groups", []),
            )
            if layout == "compatible":
                optimizer.load_state_dict(state["optimizer"])
                optimizer_restore = "checkpoint_state"
            elif layout == "legacy_consumed_parameter_generators":
                legacy_layout_reset = True
                optimizer_restore = f"reset_{layout}"
            else:
                raise ValueError(
                    "checkpoint optimizer parameter-group layout is incompatible with the current model"
                )
        scaler_restored = bool(
            restore_scaler and not legacy_layout_reset and scaler is not None and state.get("scaler")
        )
        if scaler_restored:
            scaler.load_state_dict(state["scaler"])
        scheduler_restore = "not_requested"
        if legacy_layout_reset and scheduler is not None:
            scheduler_restore = "reset_with_legacy_optimizer_layout"
        elif restore_scheduler and scheduler is not None and state.get("scheduler"):
            scheduler_state = state["scheduler"]
            source_name = state.get("scheduler_name")
            if source_name is None and "T_max" in scheduler_state:
                source_name = "CosineAnnealingLR"
            target_name = type(scheduler).__name__
            if source_name in {None, target_name}:
                scheduler.load_state_dict(scheduler_state)
                _rebind_lambda_schedule_for_continuity(
                    scheduler, optimizer if optimizer is not None else scheduler.optimizer,
                    float(getattr(scheduler, "_shift_min_learning_rate", 0.0)),
                )
                scheduler_restore = "checkpoint_state"
            else:
                # Pre-v2 development checkpoints may contain a horizon-bound
                # CosineAnnealingLR state.  Keep the restored optimizer LR and
                # migrate it explicitly.  LambdaLR accepts arbitrary keys in
                # load_state_dict(), so relying on an exception would silently
                # import stale T_max state and could make the next LR increase.
                progress = max(0, int(scheduler_state.get("last_epoch", 0)))
                scheduler_optimizer = optimizer if optimizer is not None else scheduler.optimizer
                actual_lrs = [float(group["lr"]) for group in scheduler_optimizer.param_groups]
                decay = float(getattr(scheduler, "_shift_decay", 0.95))
                minimum = float(getattr(scheduler, "_shift_min_learning_rate", 0.0))
                factor = max(decay ** progress, 1e-30)
                base_lrs = [actual_lr / factor if actual_lr > 0.0 else 0.0 for actual_lr in actual_lrs]
                scheduler.base_lrs = base_lrs
                for group, base_lr in zip(scheduler_optimizer.param_groups, base_lrs):
                    group["initial_lr"] = base_lr
                scheduler.last_epoch = progress
                scheduler._step_count = int(scheduler_state.get("_step_count", 0))
                scheduler._last_lr = actual_lrs
                _rebind_lambda_schedule_for_continuity(
                    scheduler, scheduler_optimizer, minimum,
                )
                scheduler_restore = f"migrated_{source_name}_to_{target_name}"
        elif (
            restore_scheduler and scheduler is not None and optimizer_restore == "checkpoint_state"
        ):
            # Some legacy checkpoints persisted optimizer LR but no scheduler.
            # Anchor a fresh horizon-independent schedule at that actual LR so
            # the next epoch decays continuously instead of jumping to CLI LR.
            current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            scheduler.base_lrs = list(current_lrs)
            for group, current_lr in zip(optimizer.param_groups, current_lrs):
                group["initial_lr"] = current_lr
            scheduler.last_epoch = 0
            scheduler._step_count = 1
            scheduler._last_lr = list(current_lrs)
            _rebind_lambda_schedule_for_continuity(
                scheduler, optimizer,
                float(getattr(scheduler, "_shift_min_learning_rate", 0.0)),
            )
            scheduler_restore = "anchored_to_optimizer_lr_missing_checkpoint_scheduler"
        elif not restore_scheduler:
            scheduler_restore = "reset_by_policy"
        state["_load_metadata"] = {
            "optimizer_restored": optimizer_restore == "checkpoint_state",
            "optimizer_restore": optimizer_restore,
            "scaler_restored": scaler_restored,
            "scheduler_restore": scheduler_restore,
        }
        if state.get("rng_python") is not None:
            random.setstate(state["rng_python"])
        # map_location="cuda" also moves the CPU RNG byte tensor.  PyTorch's
        # RNG restoration APIs intentionally require CPU byte tensors.
        if state.get("rng_torch") is not None:
            torch.set_rng_state(state["rng_torch"].cpu())
        if torch.cuda.is_available() and state.get("rng_cuda"):
            torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["rng_cuda"]])
        return state


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (RuntimeError, ValueError):
            pass
    return value


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_training_logs(checkpoint_dir: Path, manifest: Mapping[str, Any]) -> None:
    history = list(manifest.get("history", []))
    _atomic_json(checkpoint_dir / "training_history.json", history)
    _atomic_json(checkpoint_dir / "training_manifest.json", manifest)
    preferred = [
        "epoch", "train_loss", "validation_loss", "learning_rate", "gpu_vram_peak_mb",
        "epoch_duration_seconds", "best_updated", "unfreeze_state",
    ]
    extra = sorted({str(key) for record in history for key in record if key not in preferred})
    fieldnames = preferred + extra
    destination = checkpoint_dir / "training_history.csv"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in history:
            row = {}
            for key in fieldnames:
                value = _json_safe(record.get(key))
                row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            writer.writerow(row)
    os.replace(temporary, destination)


def _make_scheduler(optimizer, config: TrainConfig):
    import torch

    # This schedule deliberately does not depend on the requested run horizon.
    # Consequently adding epochs with --resume cannot leave a loaded cosine
    # scheduler stuck at its former T_max.  LambdaLR checkpoints also retain
    # last_epoch/base_lrs, so legacy checkpoints without scheduler state and
    # current checkpoints both resume safely.
    lambdas = []
    for group in optimizer.param_groups:
        base_learning_rate = max(float(group["lr"]), 1e-12)
        floor = min(1.0, config.scheduler_min_learning_rate / base_learning_rate)
        lambdas.append(lambda epoch, floor=floor: max(floor, 0.95 ** epoch))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    # These values are also used when explicitly migrating a legacy cosine
    # state while preserving the checkpoint optimizer's actual group LRs.
    scheduler._shift_decay = 0.95
    scheduler._shift_min_learning_rate = config.scheduler_min_learning_rate
    return scheduler


def _rebind_lambda_schedule_for_continuity(scheduler, optimizer, minimum_learning_rate: float) -> None:
    """Recreate non-serializable LambdaLR closures after checkpoint load.

    PyTorch stores ``base_lrs`` and progress but not lambda callables.  The
    closures therefore come from the current CLI config, which may differ from
    the checkpoint.  Rebinding them to the restored base/current LRs prevents a
    resume-time LR jump and keeps legacy zero-LR cosine endpoints at zero.
    """

    decay = float(getattr(scheduler, "_shift_decay", 0.95))
    current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    base_lrs = [float(value) for value in scheduler.base_lrs]
    group_floors: list[float] = []
    lambdas = []
    for base_lr, current_lr in zip(base_lrs, current_lrs):
        # If a legacy scheduler already reached zero/below the configured
        # floor, continuity takes precedence over increasing LR on resume.
        effective_floor = max(0.0, min(float(minimum_learning_rate), current_lr))
        group_floors.append(effective_floor)
        if base_lr <= 0.0:
            lambdas.append(lambda _epoch: 0.0)
        else:
            lambdas.append(
                lambda epoch, base_lr=base_lr, floor=effective_floor, decay=decay:
                max(floor / base_lr, decay ** epoch)
            )
    scheduler.lr_lambdas = lambdas
    scheduler._shift_group_min_learning_rates = group_floors


def _set_train_loader_epoch(train_loader: Any, epoch: int, seed: int) -> None:
    """Prepare deterministic epoch order for both shuffled and bucket loaders.

    A reconstructed DataLoader otherwise starts its generator and the custom
    width-bucket sampler at epoch zero after every resume.  Deriving order from
    the absolute epoch makes uninterrupted and resumed runs agree without
    serializing implementation-specific iterator internals.
    """

    epoch_seed = int(seed) + int(epoch)
    pending = [train_loader]
    visited: set[int] = set()
    while pending:
        loader = pending.pop()
        if loader is None or id(loader) in visited:
            continue
        visited.add(id(loader))
        nested = getattr(loader, "loader", None)
        if nested is not None and nested is not loader:
            pending.append(nested)
        dataset = getattr(loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            # Epoch-aware datasets keep the value in shared memory so workers
            # that persist across epochs (including Windows spawn workers) see
            # the new augmentation stream before this epoch's iterator starts.
            dataset.set_epoch(epoch)
        generator = getattr(loader, "generator", None)
        if generator is not None and hasattr(generator, "manual_seed"):
            generator.manual_seed(epoch_seed)
        for sampler_name in ("sampler", "batch_sampler"):
            sampler = getattr(loader, sampler_name, None)
            if sampler is None or id(sampler) in visited:
                continue
            visited.add(id(sampler))
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            elif hasattr(sampler, "seed"):
                # WidthBucketBatchSampler intentionally exposes ``seed`` and
                # creates a fresh random.Random from it in every __iter__.
                sampler.seed = epoch_seed
            child_sampler = getattr(sampler, "sampler", None)
            if child_sampler is not None:
                pending.append(child_sampler)


def resume_selection_decision(
    state: Mapping[str, Any] | None, config: TrainConfig,
) -> dict[str, Any]:
    """Decide whether checkpoint-selection state belongs to this validation scope."""

    if config.resume_selection_policy not in {"auto", "reset"}:
        raise ValueError("resume_selection_policy must be 'auto' or 'reset'")
    current_phase = config.training_phase
    current_scope = config.selection_scope
    if state is None:
        return {
            "source": "new_run",
            "inherited": False,
            "reset": False,
            "reset_reason": "new_run",
            "checkpoint_phase": None,
            "checkpoint_scope": None,
            "current_phase": current_phase,
            "current_scope": current_scope,
        }

    checkpoint_config = state.get("config") or {}
    checkpoint_selection = state.get("selection_state") or {}
    previous_phase = checkpoint_selection.get("training_phase") or checkpoint_config.get(
        "training_phase"
    )
    previous_scope = checkpoint_selection.get("selection_scope") or checkpoint_config.get(
        "selection_scope"
    )
    reset_reason = None
    if config.resume_selection_policy == "reset":
        reset_reason = "explicit_reset_policy"
    elif previous_phase is not None and previous_phase != current_phase:
        reset_reason = f"training_phase_changed:{previous_phase}->{current_phase}"
    elif previous_phase is None and current_phase == "real_finetune":
        # Checkpoints predating phase metadata were produced by the original
        # synthetic-pretraining-only path.  Never compare their best metric to
        # the first real-photo validation result.
        reset_reason = "legacy_checkpoint_to_real_finetune"
    elif previous_scope is None and current_scope != "synthetic_validation":
        # Legacy checkpoints without scope metadata can only be assumed to use
        # the original global synthetic Validation scope.  Never leak that
        # selection state into a CV fold or another explicitly named scope.
        reset_reason = f"legacy_unknown_scope_to:{current_scope}"
    elif previous_scope is not None and previous_scope != current_scope:
        reset_reason = f"selection_scope_changed:{previous_scope}->{current_scope}"

    inherited = reset_reason is None
    return {
        "source": "checkpoint" if inherited else "reset_for_current_scope",
        "inherited": inherited,
        "reset": not inherited,
        "reset_reason": None if inherited else reset_reason,
        "checkpoint_phase": previous_phase,
        "checkpoint_scope": previous_scope,
        "current_phase": current_phase,
        "current_scope": current_scope,
    }


def preview_resume_selection(resume: Path | None, config: TrainConfig) -> dict[str, Any]:
    """Inspect selection-state routing without loading model tensors onto the GPU."""

    if resume is None:
        return resume_selection_decision(None, config)
    import torch

    state = torch.load(resume, map_location="cpu", weights_only=False)
    return resume_selection_decision(state, config)


def _step_sample_average(scaler, optimizer, accumulated_samples: int) -> None:
    """Apply the average gradient for one possibly uneven micro-batch group."""
    if accumulated_samples < 1:
        raise ValueError("accumulated_samples must be positive")
    scaler.unscale_(optimizer)
    inverse_samples = 1.0 / accumulated_samples
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameter.grad.mul_(inverse_samples)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def fit(
    model, train_loader, validation_function: Callable[[Any], Mapping[str, float]],
    config: TrainConfig, checkpoint_dir: Path, *, resume: Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    import torch

    seed_everything(config.seed)
    device, precision, autocast_dtype = select_device_and_precision()
    model.to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, config.learning_rate), weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    scheduler = _make_scheduler(optimizer, config)
    manager = CheckpointManager(checkpoint_dir, config.model_kind)
    training_log_dir = log_dir or checkpoint_dir
    start_epoch = 0
    best_metrics = None
    history = []
    if config.resume_learning_rate_policy not in {"restore", "reset"}:
        raise ValueError("resume_learning_rate_policy must be 'restore' or 'reset'")
    learning_rate_source = "cli_new_run"
    scheduler_restore = "new_scheduler"
    selection_state = resume_selection_decision(None, config)
    if resume:
        restore_state = config.resume_learning_rate_policy == "restore"
        state = manager.load(
            resume, model, optimizer, scaler, device=device, scheduler=scheduler,
            restore_optimizer=restore_state, restore_scaler=restore_state,
            restore_scheduler=restore_state,
        )
        start_epoch = int(state["epoch"]) + 1
        selection_state = resume_selection_decision(state, config)
        source_manifest = None
        source_manifest_path = Path(resume).parent / "training_manifest.json"
        if source_manifest_path.exists():
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if selection_state["inherited"]:
            # Legacy ``last.pt`` files did not persist the historical best.
            # Prefer the adjacent manifest's selection state before falling
            # back to the last epoch's metrics, otherwise a merely better-than-
            # last result could overwrite a stronger historical best.
            best_metrics = state.get("best_metrics") or (
                source_manifest.get("best_metrics") if source_manifest else None
            ) or state.get("metrics")
        load_metadata = state.get("_load_metadata", {})
        if restore_state and load_metadata.get("optimizer_restored"):
            learning_rate_source = "checkpoint_optimizer"
        elif restore_state and str(load_metadata.get("optimizer_restore", "")).startswith("reset_legacy"):
            learning_rate_source = "cli_legacy_optimizer_layout_reset"
        else:
            learning_rate_source = "cli_resume_reset"
        scheduler_restore = load_metadata.get("scheduler_restore", "unknown")
        if selection_state["inherited"]:
            checkpoint_history = state.get("history")
            if checkpoint_history is None:
                # Older checkpoints kept history beside the checkpoint rather
                # than in it.  Use the source directory so same-phase resume
                # into a new output directory still retains prior epochs.
                if source_manifest:
                    checkpoint_history = source_manifest.get("history", [])
            history = [
                item for item in (checkpoint_history or [])
                if int(item.get("epoch", -1)) < start_epoch
            ]
            source_best = Path(resume).parent / "best.pt"
            destination_best = checkpoint_dir / "best.pt"
            if source_best.exists() and source_best.resolve() != destination_best.resolve():
                # A same-scope resume into a new output directory must still
                # have an exportable best checkpoint even if no new epoch beats
                # the inherited metric.
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_best, destination_best)
                selection_state["inherited_best_checkpoint"] = str(source_best)
    if start_epoch >= config.epochs:
        raise ValueError(
            f"no epochs would run: checkpoint resumes at epoch {start_epoch}, "
            f"but target total is {config.epochs}"
        )
    initial_group_learning_rates = [
        {"name": str(group.get("name", f"group_{index}")), "lr": float(group["lr"])}
        for index, group in enumerate(optimizer.param_groups)
    ]
    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        unfreeze_state = apply_unfreezing(model, epoch, config)
        _set_train_loader_epoch(train_loader, epoch, config.seed)
        learning_rate = max(float(group["lr"]) for group in optimizer.param_groups)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        sample_count = 0
        micro_batch_count = 0
        accumulated_samples = 0
        optimizer_step_sample_counts: list[int] = []
        for raw_batch in train_loader:
            batch = move_to_device(raw_batch, device)
            actual_batch_size = int(batch["image"].shape[0])
            if actual_batch_size < 1:
                raise ValueError("training batch contains no images")
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                loss = model_loss(config.model_kind, model, batch)
            # model_loss is a micro-batch mean.  Backpropagate its sample sum,
            # then divide once by the real group size at optimizer step time.
            # This stays correct for recognizer width buckets (for example
            # physical batches 8/4/1) and for a short final DataLoader batch.
            scaler.scale(loss * actual_batch_size).backward()
            accumulated_samples += actual_batch_size
            running_loss += float(loss.detach()) * actual_batch_size
            sample_count += actual_batch_size
            micro_batch_count += 1
            if accumulated_samples >= config.target_effective_batch:
                _step_sample_average(scaler, optimizer, accumulated_samples)
                optimizer_step_sample_counts.append(accumulated_samples)
                accumulated_samples = 0
        if accumulated_samples:
            _step_sample_average(scaler, optimizer, accumulated_samples)
            optimizer_step_sample_counts.append(accumulated_samples)
        metrics = {str(key): _json_safe(value) for key, value in dict(validation_function(model)).items()}
        if config.model_kind == "table":
            metrics["table_composite"] = 0.6 * float(metrics["cell_polygon_f1"]) + 0.4 * float(metrics["row_level_accuracy"])
        if "validation_loss" not in metrics:
            metrics["validation_loss"] = metrics.get("val_loss")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            gpu_vram_peak_mb = torch.cuda.max_memory_allocated(device) / 1_000_000
        else:
            gpu_vram_peak_mb = 0.0
        metrics.update({
            "train_loss": running_loss / max(1, sample_count),
            "train_sample_count": sample_count,
            "train_micro_batch_count": micro_batch_count,
            "optimizer_step_count": len(optimizer_step_sample_counts),
            "optimizer_step_sample_counts": optimizer_step_sample_counts,
            "effective_batch_min": min(optimizer_step_sample_counts) if optimizer_step_sample_counts else 0,
            "effective_batch_max": max(optimizer_step_sample_counts) if optimizer_step_sample_counts else 0,
            "effective_batch_mean": (
                sum(optimizer_step_sample_counts) / len(optimizer_step_sample_counts)
                if optimizer_step_sample_counts else 0.0
            ),
            "learning_rate": learning_rate,
            "gpu_vram_peak_mb": gpu_vram_peak_mb,
            "epoch_duration_seconds": time.perf_counter() - epoch_started,
            "epoch": epoch,
            "unfreeze_state": unfreeze_state,
        })
        best_updated = is_better(config.model_kind, metrics, best_metrics)
        metrics["best_updated"] = best_updated
        if best_updated:
            best_metrics = dict(metrics)
        scheduler.step()
        history.append(metrics)
        manager.save(
            "last", model, optimizer, scaler, epoch, metrics, config,
            scheduler=scheduler, best_metrics=best_metrics, history=history,
        )
        if best_updated:
            manager.save(
                "best", model, optimizer, scaler, epoch, metrics, config,
                scheduler=scheduler, best_metrics=best_metrics, history=history,
            )
        if (epoch + 1) % config.checkpoint_every == 0:
            manager.save(
                f"epoch_{epoch + 1:04d}", model, optimizer, scaler, epoch, metrics, config,
                scheduler=scheduler, best_metrics=best_metrics, history=history,
            )
        manifest = {
            "model_kind": config.model_kind,
            "best_metric_spec": BEST_METRICS[config.model_kind],
            "best_metrics": best_metrics,
            "precision": precision,
            "device": str(device),
            "config": asdict(config),
            "scheduler": {
                "name": type(scheduler).__name__, "step_unit": "epoch",
                "minimum_learning_rate": config.scheduler_min_learning_rate,
                "restore": scheduler_restore,
            },
            "learning_rate_state": {
                "source": learning_rate_source,
                "resume_policy": config.resume_learning_rate_policy,
                "initial_group_learning_rates": initial_group_learning_rates,
                "actual_group_learning_rates": [
                    {"name": str(group.get("name", f"group_{index}")), "lr": float(group["lr"])}
                    for index, group in enumerate(optimizer.param_groups)
                ],
            },
            "selection_state": selection_state,
            "history": history,
        }
        _write_training_logs(training_log_dir, manifest)
    manifest = {
        "model_kind": config.model_kind,
        "best_metric_spec": BEST_METRICS[config.model_kind],
        "best_metrics": best_metrics,
        "precision": precision,
        "device": str(device),
        "config": asdict(config),
        "scheduler": {
            "name": type(scheduler).__name__, "step_unit": "epoch",
            "minimum_learning_rate": config.scheduler_min_learning_rate,
            "restore": scheduler_restore,
        },
        "learning_rate_state": {
            "source": learning_rate_source,
            "resume_policy": config.resume_learning_rate_policy,
            "initial_group_learning_rates": initial_group_learning_rates,
            "actual_group_learning_rates": [
                {"name": str(group.get("name", f"group_{index}")), "lr": float(group["lr"])}
                for index, group in enumerate(optimizer.param_groups)
            ],
        },
        "selection_state": selection_state,
        "history": history,
    }
    _write_training_logs(training_log_dir, manifest)
    return manifest
