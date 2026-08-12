#!/usr/bin/env python3
"""Merge accepted registration outputs into a leakage-safe real-photo JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shift_ocr.master_split import MasterSplit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration-dir", required=True, type=Path)
    parser.add_argument("--master-split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    master = MasterSplit.load(args.master_split)
    count = ignored = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for path in sorted(args.registration_dir.rglob("*.registration.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("registration", {}).get("accepted"):
                continue
            for item in payload.get("objects", []):
                record = master.require(str(item["schedule_id"]), item.get("split"))
                if item.get("master_split_sha256") != master.metadata["split_sha256"]:
                    raise ValueError(f"master split hash mismatch: {path}")
                item["cv_fold"] = record.cv_fold
                stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
                ignored += bool(item.get("ignore"))
    print(json.dumps({"real_object_count": count, "ignored_count": ignored}, ensure_ascii=False))


if __name__ == "__main__":
    main()
