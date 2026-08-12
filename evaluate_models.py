#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shift_ocr.evaluation import add_confidence_intervals, choose_mobile_route, evaluate_cells
from shift_ocr.master_split import MasterSplit
from shift_ocr.shards import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("metrics")
    evaluate.add_argument("--ground-truth", required=True, type=Path)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--master-split", required=True, type=Path)
    evaluate.add_argument("--split", required=True, choices=["validation", "test", "ood_layout"])
    evaluate.add_argument("--output", required=True, type=Path)
    routes = sub.add_parser("select-route")
    routes.add_argument("--results", required=True, type=Path)
    routes.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "select-route":
        result = choose_mobile_route(json.loads(args.results.read_text(encoding="utf-8")))
    else:
        master = MasterSplit.load(args.master_split)
        truth = [item for item in iter_jsonl(args.ground_truth) if master.require(str(item["schedule_id"])).split == args.split]
        predictions = [item for item in iter_jsonl(args.predictions) if master.require(str(item["schedule_id"])).split == args.split]
        result = add_confidence_intervals(evaluate_cells(truth, predictions), iterations=2000)
        result["split"] = args.split
        result["selection_allowed"] = args.split == "validation"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
