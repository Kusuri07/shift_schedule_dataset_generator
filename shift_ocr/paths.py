"""Default storage layout for desktop dataset generation and training."""

from __future__ import annotations

from pathlib import Path


DEFAULT_STORAGE_ROOT = Path(r"D:\harudam_model")


def dataset_root(storage_root: Path = DEFAULT_STORAGE_ROOT) -> Path:
    return storage_root / "training_dataset"


def training_run_dir(
    model: str,
    phase: str,
    *,
    storage_root: Path = DEFAULT_STORAGE_ROOT,
    cv_fold: int | None = None,
) -> Path:
    suffix = "real" if phase == "real_finetune" else "pretrain"
    name = f"{model}_{suffix}"
    if cv_fold is not None:
        name += f"_cv{cv_fold}"
    return storage_root / "runs" / name
