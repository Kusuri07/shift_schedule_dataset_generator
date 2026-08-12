#!/usr/bin/env python3
"""Create model-family-specific quantization candidates for Validation A/B tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class NpzCalibrationReader:
    def __init__(self, paths: list[Path]):
        self.paths = iter(paths)

    def get_next(self):
        import numpy as np

        try:
            path = next(self.paths)
        except StopIteration:
            return None
        value = np.load(path)
        return {"image": value["image"].astype("float32")}


def main() -> None:
    import onnx
    from onnxconverter_common import float16
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dbnet", "table", "recognizer"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--calibration-dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [{"variant": "fp32", "path": str(args.input)}]
    fp16_path = args.output_dir / f"{args.model}_fp16.onnx"
    fp16_model = float16.convert_float_to_float16(
        onnx.load(args.input), keep_io_types=True, disable_shape_infer=False,
    )
    onnx.save(fp16_model, fp16_path)
    results.append({"variant": "fp16", "path": str(fp16_path)})
    if args.model in {"dbnet", "table"}:
        if not args.calibration_dir:
            raise ValueError("CNN static INT8 requires --calibration-dir with Validation-independent Train samples")
        output = args.output_dir / f"{args.model}_static_int8_qdq.onnx"
        paths = sorted(args.calibration_dir.glob("*.npz"))
        if not paths:
            raise ValueError("no .npz calibration inputs found")
        quantize_static(
            args.input, output, NpzCalibrationReader(paths), quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
        results.append({"variant": "static_int8_qdq", "path": str(output)})
    else:
        output = args.output_dir / "recognizer_dynamic_int8.onnx"
        quantize_dynamic(args.input, output, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm", "LSTM"])
        results.append({"variant": "dynamic_int8", "path": str(output)})
        cnn_only = args.output_dir / "recognizer_cnn_static_int8.onnx"
        if args.calibration_dir and list(args.calibration_dir.glob("*.npz")):
            quantize_static(
                args.input, cnn_only, NpzCalibrationReader(sorted(args.calibration_dir.glob("*.npz"))),
                quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                op_types_to_quantize=["Conv"],
            )
            results.append({"variant": "cnn_only_static_int8_bilstm_fp32", "path": str(cnn_only)})
    for result in results:
        path = Path(result["path"])
        result["size_bytes"] = path.stat().st_size
    (args.output_dir / "quantization_manifest.json").write_text(
        json.dumps({"selection_split": "validation", "max_accuracy_loss": 0.005, "variants": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
