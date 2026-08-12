"""ONNX profile export, quantization candidates and mobile usability checks."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ExportProfile:
    name: str
    model_kind: str
    batch: int
    height: int
    width: int
    dynamic_batch: bool
    dynamic_height: bool
    dynamic_width: bool
    intended_ep: tuple[str, ...]


def candidate_profiles(model_kind: str) -> list[ExportProfile]:
    if model_kind == "recognizer":
        profiles = [ExportProfile(
            "recognizer_dynamic_batch", model_kind, 1, 48, 320, True, False, True,
            ("CPU", "XNNPACK"),
        )]
        for width, batches in {160: (1, 4, 8, 16), 320: (1, 2, 4, 8), 640: (1, 2, 4)}.items():
            profiles.extend(ExportProfile(
                f"recognizer_w{width}_b{batch}", model_kind, batch, 48, width, False, False, False,
                ("NNAPI", "CoreML"),
            ) for batch in batches)
        return profiles
    return [
        ExportProfile(f"{model_kind}_dynamic", model_kind, 1, 960, 1280, False, True, True, ("CPU", "XNNPACK")),
        ExportProfile(f"{model_kind}_fixed", model_kind, 1, 960, 1280, False, False, False, ("NNAPI", "CoreML")),
    ]


class OutputAdapter:
    """Wrap dict-returning networks in a tuple suitable for ONNX export."""

    def __new__(cls, model, output_names: Sequence[str]):
        import torch

        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, image):
                result = self.model(image)
                if isinstance(result, dict):
                    return tuple(result[name] for name in output_names)
                return result

        return Adapter()


def output_names(kind: str) -> list[str]:
    if kind == "dbnet":
        return ["probability", "threshold", "binary"]
    if kind == "table":
        return ["cell_heatmap", "corner_offsets", "row_embedding", "column_embedding"]
    return ["log_probs"]


def export_profile(model, profile: ExportProfile, output_path: Path, *, opset: int = 17) -> dict[str, Any]:
    import torch

    names = output_names(profile.model_kind)
    adapter = OutputAdapter(model.eval(), names)
    dummy = torch.zeros(profile.batch, 3, profile.height, profile.width)
    dynamic_axes: dict[str, dict[int, str]] = {}
    if profile.dynamic_batch:
        dynamic_axes["image"] = {0: "batch"}
        for name in names:
            dynamic_axes[name] = {0: "batch"}
    if profile.dynamic_width:
        dynamic_axes.setdefault("image", {})[3] = "width"
        for name in names:
            dynamic_axes.setdefault(name, {})[1 if profile.model_kind == "recognizer" else 3] = "output_width"
    if profile.dynamic_height:
        dynamic_axes.setdefault("image", {})[2] = "height"
        for name in names:
            dynamic_axes.setdefault(name, {})[2] = "output_height"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        adapter, dummy, output_path, input_names=["image"], output_names=names,
        dynamic_axes=dynamic_axes or None, opset_version=opset, do_constant_folding=True,
    )
    return {**asdict(profile), "path": str(output_path), "size_bytes": output_path.stat().st_size}


def quantize_dynamic_recognizer(source: Path, output: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(source, output, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm", "LSTM"])


def usability_check(model_path: Path) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "onnxruntime.tools.mobile_helpers.usability_checker",
        str(model_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "return_code": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-4000:],
    }


def select_mobile_profile(results: Sequence[Mapping[str, Any]], *, max_accuracy_loss: float = 0.005) -> dict[str, Any]:
    if not results or any(str(item.get("split")) != "validation" for item in results):
        raise ValueError("profile selection requires real Validation benchmark results")
    acceptable = [item for item in results if float(item.get("accuracy_loss", 1.0)) <= max_accuracy_loss and bool(item.get("usable", True))]
    if not acceptable:
        raise ValueError("no mobile profile satisfies the accuracy/usability guardrails")
    selected = min(acceptable, key=lambda item: (float(item["p95_latency_ms"]), float(item["peak_memory_mb"]), int(item["model_size_bytes"])))
    return {
        "selected_profile": selected["profile"],
        "selected_execution_provider": selected["execution_provider"],
        "selected_bucket_batches": selected.get("bucket_batches", {"160": 1, "320": 1, "640": 1}),
        "batch1_fallback": True,
        "include_only_selected_profile": True,
        "validation_result": dict(selected),
    }


def write_model_manifest(
    path: Path, *, version: str, model_files: Sequence[Path], charset_version: str,
    normalization: Mapping[str, Any], selected_profile: Mapping[str, Any], quantization: str,
) -> None:
    import hashlib

    files = []
    for model in model_files:
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        files.append({"path": model.name, "sha256": digest, "size_bytes": model.stat().st_size})
    payload = {
        "version": version,
        "charset_version": charset_version,
        "normalization": dict(normalization),
        "quantization": quantization,
        "profile": dict(selected_profile),
        "models": files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
