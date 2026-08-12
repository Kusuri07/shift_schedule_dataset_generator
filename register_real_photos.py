#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shift_ocr.registration import register, transfer_objects
from shift_ocr.master_split import MasterSplit
from shift_ocr.shards import iter_jsonl


def main() -> None:
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--photo", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--master-split", required=True, type=Path)
    parser.add_argument("--photo-path-in-dataset", help="Stored relative image path; defaults to the photo filename")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    def oriented(path: Path):
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)

    reference = oriented(args.reference)
    photo = oriented(args.photo)
    if reference is None or photo is None:
        raise ValueError("reference/photo image could not be read")
    result = register(reference, photo, partial=args.partial)
    payload = {"registration": result, "objects": []}
    source_objects = list(iter_jsonl(args.objects))
    schedule_ids = {str(item["schedule_id"]) for item in source_objects}
    if len(schedule_ids) != 1:
        raise ValueError("registration objects must belong to exactly one schedule_id")
    schedule_id = next(iter(schedule_ids))
    master = MasterSplit.load(args.master_split)
    split_record = master.require(schedule_id)
    if result.get("accepted"):
        transferred = transfer_objects(source_objects, result, photo.shape[1], photo.shape[0])
        for item in transferred:
            item.update({
                "source_domain": "real",
                "split": split_record.split,
                "cv_fold": split_record.cv_fold,
                "master_split_sha256": master.metadata["split_sha256"],
                "image_path": args.photo_path_in_dataset or args.photo.name,
                "image_width": photo.shape[1],
                "image_height": photo.shape[0],
            })
        payload["objects"] = transferred
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result.get("accepted"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
