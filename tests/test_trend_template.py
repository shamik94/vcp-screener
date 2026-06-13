"""Unit tests for trend template indicators and pass criteria."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.screener.config import CountryThresholds
from src.screener.trend_template import (
    apply_heuristic_vcp,
    apply_trend_template,
    build_universe_features,
    compute_features,
)
from tests.conftest import make_ohlcv


def test_compute_features_returns_none_for_short_history(short_history_100):
    assert compute_features(short_history_100) is None


def test_sma_stack_passes_for_uptrend(trending_up_300):
    feats = compute_features(trending_up_300)
    assert feats is not None
    assert feats.close > feats.sma50 > feats.sma150 > feats.sma200
    assert feats.sma200_slope > 0


def test_sma_stack_fails_for_downtrend(trending_down_300):
    feats = compute_features(trending_down_300)
    assert feats is not None
    assert feats.sma50 < feats.sma150 < feats.sma200
    assert feats.sma200_slope < 0


def test_52w_high_low_match_window(trending_up_300):
    feats = compute_features(trending_up_300)
    last_year_high = trending_up_300["high"].iloc[-252:].max()
    last_year_low = trending_up_300["low"].iloc[-252:].min()
    assert feats.high_52w == last_year_high
    assert feats.low_52w == last_year_low


def test_tightness_math():
    # Flat 11-bar tail at price 100, high=101, low=99 → range/close = 0.02.
    closes = np.concatenate([np.linspace(50, 100, 290), np.full(15, 100.0)])
    df = make_ohlcv("FLAT", closes)
    feats = compute_features(df)
    assert feats is not None
    # high_mult=1.01, low_mult=0.99 on price 100 → range = 2 → tightness 0.02.
    assert feats.tightness_10d == 2.0 / 100.0


def test_rs_rank_is_relative_not_absolute():
    """Higher 12m return must rank above lower 12m return in the same universe."""
    rng = np.random.default_rng(0)
    big_winner = make_ohlcv("WIN", np.linspace(50, 200, 300) + rng.normal(0, 0.5, 300))
    laggard = make_ohlcv("LAG", np.linspace(100, 110, 300) + rng.normal(0, 0.5, 300))
    ohlcv = pd.concat([big_winner, laggard], ignore_index=True)
    features = build_universe_features(ohlcv)
    by_symbol = features.set_index("symbol")
    assert by_symbol.loc["WIN", "rs_rank"] > by_symbol.loc["LAG", "rs_rank"]
    # Two symbols → percentile ranks should be exactly 100 and 50.
    assert by_symbol.loc["WIN", "rs_rank"] == 100.0


def test_htf_detector_positive(htf_candidate):
    feats = compute_features(htf_candidate)
    assert feats is not None
    assert feats.htf_flag is True


def test_htf_detector_negative(non_htf_slow_grind):
    feats = compute_features(non_htf_slow_grind)
    assert feats is not None
    assert feats.htf_flag is False


def test_apply_trend_template_passes_clean_uptrend(trending_up_300):
    # Build a tiny universe of one tight-base uptrend that should pass everything.
    base = make_ohlcv("TIGHT", np.concatenate([np.linspace(50, 195, 290), np.full(15, 197.0)]))
    # Force low 10d volume vs 50d to satisfy vol-dryup pre-filter.
    base.loc[base.index[-10:], "volume"] = 500_000
    features = build_universe_features(base)
    thresholds = CountryThresholds(liquidity_floor=100_000, rs_min=0)  # solo universe → rank=100
    out = apply_trend_template(features, thresholds)
    assert bool(out["passed_stage1"].iloc[0]) is True


def test_apply_trend_template_rejects_downtrend(trending_down_300):
    features = build_universe_features(trending_down_300)
    thresholds = CountryThresholds(liquidity_floor=0, rs_min=0)
    out = apply_trend_template(features, thresholds)
    assert bool(out["passed_stage1"].iloc[0]) is False


def test_window_range_features_capture_max_and_avg(heuristic_tight_window):
    feats = compute_features(heuristic_tight_window)
    assert feats is not None
    # All last-4 bars built with high_mult=1.02, low_mult=0.98 → range ≈ 4%.
    assert feats.window_max_range_pct == pytest.approx(0.04, abs=0.001)
    assert feats.window_avg_range_pct == pytest.approx(0.04, abs=0.001)


def test_apply_heuristic_vcp_passes_tight_window(heuristic_tight_window):
    features = build_universe_features(heuristic_tight_window)
    thresholds = CountryThresholds(liquidity_floor=0, rs_min=0)
    out = apply_heuristic_vcp(features, thresholds)
    row = out.iloc[0]
    assert bool(row["passed_heuristic"]) is True
    assert row["window_max_range_pct"] <= thresholds.heuristic_max_bar_range


def test_apply_heuristic_vcp_rejects_one_wide_bar_in_window(heuristic_window_with_wide_bar):
    """Strict rule: a single wide bar in the last N kills the signal."""
    features = build_universe_features(heuristic_window_with_wide_bar)
    thresholds = CountryThresholds(liquidity_floor=0, rs_min=0)
    out = apply_heuristic_vcp(features, thresholds)
    row = out.iloc[0]
    assert bool(row["passed_heuristic"]) is False
    assert row["window_max_range_pct"] > thresholds.heuristic_max_bar_range
