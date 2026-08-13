"""Configurable storage layout for production dataset generation and training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_PATH_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "desktop_paths.json"


@dataclass(frozen=True)
class StorageLayout:
    datasets: Path
    shards: Path
    checkpoints: Path
    logs: Path
    exports: Path
    cache: Path

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StorageLayout":
        paths = value.get("paths", value)
        missing = [name for name in cls.__annotations__ if not paths.get(name)]
        if missing:
            raise ValueError(f"path config is missing: {', '.join(missing)}")
        return cls(**{
            name: Path(str(paths[name])).expanduser().resolve()
            for name in cls.__annotations__
        })

    def dataset(self, name: str) -> Path:
        return self.datasets / safe_dataset_name(name)

    def shard_set(self, name: str) -> Path:
        return self.shards / safe_dataset_name(name)

    def dataset_cache(self, name: str) -> Path:
        return self.cache / safe_dataset_name(name)

    def dataset_logs(self, name: str) -> Path:
        return self.logs / safe_dataset_name(name)

    def checkpoint_run(self, model: str, phase: str, cv_fold: int | None = None) -> Path:
        suffix = "real" if phase == "real_finetune" else "pretrain"
        name = f"{model}_{suffix}"
        if cv_fold is not None:
            name += f"_cv{cv_fold}"
        return self.checkpoints / name

    def training_logs(self, model: str, phase: str, cv_fold: int | None = None) -> Path:
        return self.logs / self.checkpoint_run(model, phase, cv_fold).name


def safe_dataset_name(value: str) -> str:
    name = value.strip()
    if not name or Path(name).name != name or any(char in name for char in "\\/:"):
        raise ValueError(f"dataset name must be one path component: {value!r}")
    return name


def load_storage_layout(path: Path | None = None) -> StorageLayout:
    config_path = (path or DEFAULT_PATH_CONFIG).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"storage path config is required: {config_path}; use --path-config to select another file"
        )
    return StorageLayout.from_mapping(json.loads(config_path.read_text(encoding="utf-8")))
