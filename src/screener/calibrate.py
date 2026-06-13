"""Grader-only calibration against examples/{vcp,not_vcp}/."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import get_settings
from .grade import GradeRequest, grade_all


async def _run(settings, positives: list[Path], negatives: list[Path]) -> tuple[list[dict], list[dict]]:
    pos_reqs = [
        GradeRequest(symbol=p.stem, chart_path=p, close=0.0, high_52w=0.0) for p in positives
    ]
    neg_reqs = [
        GradeRequest(symbol=p.stem, chart_path=p, close=0.0, high_52w=0.0) for p in negatives
    ]
    out_dir = settings.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_results = await grade_all(
        pos_reqs, out_dir=out_dir, date_str="calibrate", settings=settings, include_few_shot=False
    )
    neg_results = await grade_all(
        neg_reqs, out_dir=out_dir, date_str="calibrate", settings=settings, include_few_shot=False
    )
    return pos_results, neg_results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("vcp-screener-calibrate")
    ap.add_argument("--min-grade", type=float, default=60.0)
    args = ap.parse_args(argv)

    settings = get_settings()
    positives = sorted((settings.examples_dir / "vcp").glob("*.png"))
    negatives = sorted((settings.examples_dir / "not_vcp").glob("*.png"))

    if not positives and not negatives:
        print(f"No images in {settings.examples_dir}/{{vcp,not_vcp}}.  Drop reference PNGs first.")
        return 1

    print(f"model={settings.vcp_model}  positives={len(positives)}  negatives={len(negatives)}")
    pos_results, neg_results = asyncio.run(_run(settings, positives, negatives))

    print("\n--- POSITIVES ---")
    for r in pos_results:
        print(json.dumps(r, indent=2, default=str))
    print("\n--- NEGATIVES ---")
    for r in neg_results:
        print(json.dumps(r, indent=2, default=str))

    pos_hit = sum(1 for r in pos_results if r.get("grade", 0) >= args.min_grade)
    neg_hit = sum(1 for r in neg_results if r.get("grade", 100) < args.min_grade)
    pos_avg = (sum(r.get("grade", 0) for r in pos_results) / len(pos_results)) if pos_results else 0
    neg_avg = (sum(r.get("grade", 0) for r in neg_results) / len(neg_results)) if neg_results else 0

    print("\n--- SUMMARY ---")
    if positives:
        print(f"positives graded >= {args.min_grade}: {pos_hit}/{len(positives)} ({pos_hit/len(positives)*100:.0f}%)  avg={pos_avg:.1f}")
    if negatives:
        print(f"negatives graded <  {args.min_grade}: {neg_hit}/{len(negatives)} ({neg_hit/len(negatives)*100:.0f}%)  avg={neg_avg:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
