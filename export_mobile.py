#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shift_ocr.charset import load_charset
from shift_ocr.export import candidate_profiles, export_profile, quantize_dynamic_recognizer, usability_check
from shift_ocr.models import build_model


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dbnet", "recognizer", "table"])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--charset", type=Path, default=Path("data/korean_charset_v1.txt"))
    parser.add_argument("--profile", action="append", help="Export only named profile(s); default exports all candidates")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--dynamic-int8", action="store_true", help="Recognizer only; creates a dynamic INT8 comparison model")
    args = parser.parse_args()
    charset = load_charset(args.charset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "korean_charset_v1.txt").write_text(
        "".join(character + "\n" for character in charset), encoding="utf-8"
    )
    model = build_model(args.model, class_count=len(charset) + 1, attention=args.attention, top_k=1500)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state))
    profiles = candidate_profiles(args.model)
    if args.profile:
        profiles = [profile for profile in profiles if profile.name in set(args.profile)]
    if not profiles:
        raise ValueError("no matching export profiles")
    results = []
    for profile in profiles:
        path = args.output_dir / f"{profile.name}.onnx"
        result = export_profile(model, profile, path)
        result["usability"] = usability_check(path)
        results.append(result)
        if args.dynamic_int8 and args.model == "recognizer" and profile.dynamic_batch:
            quantized = path.with_name(path.stem + "_dynamic_int8.onnx")
            quantize_dynamic_recognizer(path, quantized)
            results.append({"name": profile.name + "_dynamic_int8", "path": str(quantized), "size_bytes": quantized.stat().st_size, "usability": usability_check(quantized)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "export_manifest.json").write_text(json.dumps({"profiles": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"profiles": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
