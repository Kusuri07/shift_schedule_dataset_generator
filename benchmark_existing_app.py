#!/usr/bin/env python3
"""Record the existing app OCR baseline on the exact real Validation split."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from shift_ocr.evaluation import evaluate_cells
from shift_ocr.master_split import MasterSplit
from shift_ocr.shards import iter_jsonl


def git_value(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-repo", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path, help="JSONL emitted by the current Android OCR paths")
    parser.add_argument("--master-split", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path, help="JSONL with path, latency_ms and peak_memory_mb")
    parser.add_argument("--device-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    master = MasterSplit.load(args.master_split)
    truth = [item for item in iter_jsonl(args.ground_truth) if master.require(str(item["schedule_id"])).split == "validation"]
    predictions = [item for item in iter_jsonl(args.predictions) if master.require(str(item["schedule_id"])).split == "validation"]
    runtime = list(iter_jsonl(args.runtime))
    metrics = evaluate_cells(truth, predictions)
    metrics.pop("schedule_values", None)
    latencies = [float(item["latency_ms"]) for item in runtime]
    memories = [float(item["peak_memory_mb"]) for item in runtime]
    runtime_by_path = defaultdict(list)
    for item in runtime:
        runtime_by_path[str(item.get("path", "unknown"))].append(item)
    prediction_paths = Counter(str(item.get("path", "unknown")) for item in predictions)
    failures = Counter(str(item["failure_type"]) for item in predictions if item.get("failure_type"))
    model_dir = args.app_repo / "android" / "app" / "src" / "main" / "assets" / "models"
    models = [{"name": path.name, "size_bytes": path.stat().st_size} for path in sorted(model_dir.glob("*.onnx"))]
    result = {
        "baseline_schema": "existing_app_ocr_v1",
        "split": "validation",
        "app_commit": git_value(args.app_repo, "rev-parse", "HEAD"),
        "app_dirty": bool(git_value(args.app_repo, "status", "--porcelain")),
        "paths": ["recognize:DBNet->SVTR", "recognizeTableGrid:SLANet->SVTR", "DBNet clustering fallback"],
        "metrics": metrics,
        "runtime": {
            "p50_latency_ms": percentile(latencies, 0.50),
            "p95_latency_ms": percentile(latencies, 0.95),
            "peak_memory_mb": max(memories, default=0.0),
            "by_path": {
                path: {
                    "p50_latency_ms": percentile([float(item["latency_ms"]) for item in values], 0.50),
                    "p95_latency_ms": percentile([float(item["latency_ms"]) for item in values], 0.95),
                    "peak_memory_mb": max((float(item["peak_memory_mb"]) for item in values), default=0.0),
                }
                for path, values in runtime_by_path.items()
            },
        },
        "fallback_rate": prediction_paths["dbnet_fallback"] / max(1, sum(prediction_paths.values())),
        "prediction_path_counts": dict(prediction_paths),
        "failure_types": dict(failures),
        "models": models,
        "device": json.loads(args.device_manifest.read_text(encoding="utf-8")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
