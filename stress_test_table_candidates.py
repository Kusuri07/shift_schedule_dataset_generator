#!/usr/bin/env python3
"""Verify that table post-processing retains more than 1,000 cell candidates."""

from __future__ import annotations

import argparse

from shift_ocr.models import TableStructureModel


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=1100)
    parser.add_argument("--top-k", type=int, default=1500)
    args = parser.parse_args()
    if args.cells < 1000:
        raise ValueError("stress test must contain at least 1,000 cells")
    model = TableStructureModel(top_k=args.top_k)
    side = 192
    heatmap = torch.full((1, 1, side, side), -20.0)
    positions = [(y, x) for y in range(1, side, 3) for x in range(1, side, 3)][: args.cells]
    for y, x in positions:
        heatmap[0, 0, y, x] = 20.0
    decoded = model.decode_candidates({"cell_heatmap": heatmap}, threshold=0.9)[0]
    retained = len(decoded[0])
    if retained != args.cells:
        raise RuntimeError(f"candidate bottleneck: retained {retained}/{args.cells}")
    print({"requested_cells": args.cells, "retained": retained, "top_k": args.top_k})


if __name__ == "__main__":
    main()
