#!/usr/bin/env python3
"""Generate held-out schedule layouts for reporting-only OOD evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import generate_dataset as generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("output_ood_layout"))
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--min-people", type=int, default=18)
    parser.add_argument("--max-people", type=int, default=40)
    args = parser.parse_args()
    config = generator.GeneratorConfig(
        count=args.count,
        seed=args.seed,
        output_dir=str(args.output_dir),
        min_people=args.min_people,
        max_people=args.max_people,
        template_ids=generator.OOD_TEMPLATE_IDS.copy(),
        schedule_id_prefix="ood_schedule",
        scrape_surnames=False,
    )
    generator.generate_dataset(config, force_template_cycle=True)


if __name__ == "__main__":
    main()
