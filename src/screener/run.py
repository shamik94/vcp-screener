"""End-to-end CLI: stage1 -> render -> grade -> report."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as date_cls
from pathlib import Path

from . import db
from .config import COUNTRY_THRESHOLDS, get_settings, resolve_country
from .grade import GradeRequest, grade_all_sync
from .render import render_chart
from .report import build_watchlist, write_csv, write_md
from .trend_template import apply_heuristic_vcp, apply_trend_template, build_universe_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("screener")


def _parse_date(s: str | None) -> date_cls:
    if s is None or s == "today":
        return date_cls.today()
    return date_cls.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("vcp-screener")
    ap.add_argument("--countries", default="us", help="Comma-separated country codes (e.g. us,in)")
    ap.add_argument("--date", default="today", help="As-of date YYYY-MM-DD or 'today'")
    ap.add_argument(
        "--mode",
        choices=["vcp", "trend", "heuristic"],
        default="vcp",
        help=(
            "vcp = full pipeline with LLM grading (default). "
            "trend = pure Minervini trend template (no tightness, no charts, no LLM). "
            "heuristic = trend template + numerical contraction detector "
            "(>=4 tight days in last 10, range contraction, volume dry-up); no LLM."
        ),
    )
    ap.add_argument("--skip-grade", action="store_true", help="Run Stages 1+2 only (ignored in --mode trend)")
    ap.add_argument("--top", type=int, default=50, help="Top-N to print/save in trend mode (default 50)")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of charts/grades for debug")
    args = ap.parse_args(argv)

    settings = get_settings()
    as_of = _parse_date(args.date)
    date_str = as_of.isoformat()
    out_dir = settings.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    countries = [resolve_country(c.strip()) for c in args.countries.split(",") if c.strip()]
    all_candidates = []
    grade_requests: list[tuple[str, GradeRequest]] = []

    for country in countries:
        thresholds = COUNTRY_THRESHOLDS.get(country)
        if thresholds is None:
            logger.error("No thresholds configured for country=%s; skipping", country)
            continue

        logger.info("loading OHLCV for country=%s as_of=%s", country, as_of)
        ohlcv = db.load_ohlcv(country, as_of=as_of)
        if ohlcv.empty:
            logger.warning("no data for country=%s", country)
            continue
        universe_size = ohlcv["symbol"].nunique()

        logger.info("building features for %d symbols", universe_size)
        features = build_universe_features(
            ohlcv, heuristic_window_bars=thresholds.heuristic_window_bars
        )

        if args.mode == "heuristic":
            scored = apply_heuristic_vcp(features, thresholds)
            scored.insert(1, "country", country)
            survivors = scored[scored["passed_heuristic"]].copy()
            logger.info(
                "country=%s mode=heuristic: universe=%d heuristic_survivors=%d",
                country, universe_size, len(survivors),
            )
            if survivors.empty:
                continue
            # Rank: tightest window first (lowest avg range), RS tiebreak.
            ranked = survivors.sort_values(
                ["window_avg_range_pct", "rs_rank"], ascending=[True, False]
            ).head(args.top)
            heur_path = out_dir / f"heuristic_{date_str}_{country}.csv"
            ranked.to_csv(heur_path, index=False)
            logger.info("country=%s: wrote %d heuristic survivors -> %s", country, len(ranked), heur_path)
            all_candidates.append(ranked)
            continue

        scored = apply_trend_template(features, thresholds, require_tightness=(args.mode == "vcp"))
        scored.insert(1, "country", country)
        survivors = scored[scored["passed_stage1"]].copy()

        cand_path = out_dir / f"candidates_{date_str}.csv"
        if cand_path.exists():
            existing = scored.copy()
            existing.to_csv(cand_path, mode="a", header=False, index=False)
        else:
            scored.to_csv(cand_path, index=False)
        logger.info("country=%s mode=%s: universe=%d stage1_survivors=%d", country, args.mode, universe_size, len(survivors))

        if survivors.empty:
            continue

        if args.mode == "trend":
            # Pure trend-template mode: rank by RS, write CSV, no charts/grades.
            ranked = survivors.sort_values("rs_rank", ascending=False).head(args.top)
            trend_path = out_dir / f"trend_{date_str}_{country}.csv"
            ranked.to_csv(trend_path, index=False)
            logger.info("country=%s: wrote %d trend-template survivors -> %s", country, len(ranked), trend_path)
            all_candidates.append(ranked)
            continue

        if args.limit is not None:
            survivors = survivors.head(args.limit)

        chart_dir = out_dir / "charts" / date_str
        for _, row in survivors.iterrows():
            symbol = row["symbol"]
            df_sym = ohlcv[ohlcv["symbol"] == symbol]
            chart_path = chart_dir / f"{symbol}.png"
            try:
                render_chart(df_sym, symbol, chart_path)
            except Exception as e:  # one bad symbol shouldn't kill the run.
                logger.warning("render failed for %s: %s", symbol, e)
                continue
            grade_requests.append(
                (
                    country,
                    GradeRequest(
                        symbol=symbol,
                        chart_path=chart_path,
                        close=float(row["close"]),
                        high_52w=float(row["high_52w"]),
                        tightness_10d=float(row["tightness_10d"]),
                        vol_10d_avg=float(row["vol_10d_avg"]),
                        vol_50d_avg=float(row["vol_50d_avg"]),
                    ),
                )
            )

        all_candidates.append(survivors)

    if not all_candidates:
        logger.info("no survivors; nothing to grade")
        return 0
    candidates_df = (
        all_candidates[0] if len(all_candidates) == 1 else __import__("pandas").concat(all_candidates, ignore_index=True)
    )

    if args.mode in ("trend", "heuristic"):
        total = sum(len(c) for c in all_candidates)
        print(f"SUMMARY  mode={args.mode}  survivors={total}  (no charts, no grading)")
        return 0

    logger.info("rendered %d charts", len(grade_requests))

    if args.skip_grade:
        print(f"SUMMARY  stage1={len(grade_requests)}  (grading skipped)")
        return 0

    if not grade_requests:
        print("SUMMARY  stage1=0  graded=0  final=0")
        return 0

    reqs = [r for _, r in grade_requests]
    logger.info("grading %d charts with model=%s concurrency=%d", len(reqs), settings.vcp_model, settings.concurrency)
    grades = grade_all_sync(reqs, out_dir=out_dir, date_str=date_str)

    grades_path = out_dir / f"grades_{date_str}.json"
    grades_path.write_text(__import__("json").dumps(grades, indent=2, default=str))

    watchlist = build_watchlist(grades, candidates_df)
    write_csv(watchlist, out_dir / f"watchlist_{date_str}.csv")
    summary = (
        f"as_of={date_str}  stage1={len(grade_requests)}  graded={len(grades)}  final={len(watchlist)}"
    )
    write_md(watchlist, out_dir / f"watchlist_{date_str}.md", summary=summary)
    print(f"SUMMARY  {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
