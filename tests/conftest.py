"""Synthetic OHLCV fixtures for indicator math tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    symbol: str,
    closes: np.ndarray,
    *,
    start: str = "2024-01-02",
    volume: float | np.ndarray = 1_000_000,
    high_mult: float = 1.01,
    low_mult: float = 0.99,
) -> pd.DataFrame:
    """Build a long-form OHLCV DataFrame from a 1-D close array.
    Highs/lows are derived from close, opens equal previous close (or first close on day 0)."""
    n = len(closes)
    dates = pd.bdate_range(start=start, periods=n)
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = closes * high_mult
    lows = closes * low_mult
    vols = np.full(n, volume, dtype=float) if np.isscalar(volume) else np.asarray(volume, dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        }
    )


@pytest.fixture
def trending_up_300() -> pd.DataFrame:
    """300 bars of steady uptrend with mild noise — passes SMA stack & slope."""
    rng = np.random.default_rng(42)
    trend = np.linspace(100, 200, 300)
    noise = rng.normal(0, 0.5, 300)
    return make_ohlcv("UP", trend + noise)


@pytest.fixture
def trending_down_300() -> pd.DataFrame:
    """300 bars of steady downtrend — fails SMA stack."""
    rng = np.random.default_rng(7)
    trend = np.linspace(200, 100, 300)
    noise = rng.normal(0, 0.5, 300)
    return make_ohlcv("DOWN", trend + noise)


@pytest.fixture
def short_history_100() -> pd.DataFrame:
    """Only 100 bars — insufficient for 252-bar windows."""
    return make_ohlcv("SHORT", np.linspace(100, 110, 100))


@pytest.fixture
def htf_candidate() -> pd.DataFrame:
    """260+ bars: long flat base, then 8-week +100% rip, then a tight 4-week flag."""
    flat = np.full(200, 50.0)
    rip = np.linspace(50, 100, 40)  # +100% in 8 weeks
    flag = np.full(20, 95.0) + np.random.default_rng(1).normal(0, 0.5, 20)  # ~5% pullback band
    closes = np.concatenate([flat, rip, flag])
    return make_ohlcv("HTF", closes)


@pytest.fixture
def non_htf_slow_grind() -> pd.DataFrame:
    """260 bars of slow steady gain — no 8-week +90% leg."""
    return make_ohlcv("SLOW", np.linspace(50, 80, 260))


@pytest.fixture
def heuristic_tight_window() -> pd.DataFrame:
    """300 bars uptrend; last 4 bars all have ~4% range (well under 20%)."""
    rng = np.random.default_rng(11)
    closes = np.linspace(50, 100, 300) + rng.normal(0, 0.1, 300)
    df = make_ohlcv("TIGHT4", closes, high_mult=1.02, low_mult=0.98)  # 4% range bars
    return df


@pytest.fixture
def heuristic_window_with_wide_bar() -> pd.DataFrame:
    """Last 4 bars: three tight (~4%), one wide (~30%). MUST fail the strict rule."""
    rng = np.random.default_rng(12)
    closes = np.linspace(50, 100, 300) + rng.normal(0, 0.1, 300)
    df = make_ohlcv("MIXED4", closes, high_mult=1.02, low_mult=0.98)
    wide_idx = df.index[-3]
    df.at[wide_idx, "high"] = df.at[wide_idx, "close"] * 1.18
    df.at[wide_idx, "low"] = df.at[wide_idx, "close"] * 0.88   # range = 30%
    return df
