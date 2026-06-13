"""Stage 1: deterministic Minervini-style trend-template filter + HTF detector.

Input: long-form OHLCV DataFrame (symbol, date, open, high, low, close, volume).
Output: per-symbol feature DataFrame with pass flags.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import CountryThresholds


_MIN_HISTORY_BARS = 252  # need a full year for RS / 52w stats


@dataclass
class SymbolFeatures:
    symbol: str
    close: float
    sma50: float
    sma150: float
    sma200: float
    sma200_slope: float
    high_52w: float
    low_52w: float
    ret_3m: float
    ret_6m: float
    ret_9m: float
    ret_12m: float
    weighted_return: float
    dollar_vol_20d: float
    tightness_10d: float
    vol_10d_avg: float
    vol_50d_avg: float
    htf_flag: bool
    # Heuristic VCP features (range_pct = (high-low)/close per bar):
    window_max_range_pct: float   # max range_pct in the last N bars (pass condition)
    window_avg_range_pct: float   # avg range_pct in the last N bars (ranking key)


def _safe_pct_change(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return np.nan
    past = series.iloc[-periods - 1]
    last = series.iloc[-1]
    if past == 0 or pd.isna(past):
        return np.nan
    return (last / past) - 1.0


def _slope(series: pd.Series) -> float:
    """Ordinary-least-squares slope of `series` vs its positional index."""
    n = len(series)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    y = series.to_numpy(dtype=float)
    if np.any(np.isnan(y)):
        return np.nan
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _detect_htf(close: pd.Series, high: pd.Series, low: pd.Series) -> bool:
    """High Tight Flag: prior 8-week (40-bar) gain >= 90% followed by a 15-25 bar
    flag with depth <= 25%. Looks at the last ~12 weeks (60 bars)."""
    bars_8w = 40
    flag_min, flag_max = 15, 25
    if len(close) < bars_8w + flag_min:
        return False
    tail = close.iloc[-(bars_8w + flag_max):]
    for flag_len in range(flag_min, flag_max + 1):
        if len(tail) < bars_8w + flag_len:
            continue
        run_close = tail.iloc[-(bars_8w + flag_len)]
        run_peak = tail.iloc[-flag_len - 1]
        if run_close == 0 or pd.isna(run_close):
            continue
        run_gain = (run_peak / run_close) - 1.0
        if run_gain < 0.90:
            continue
        flag_high = high.iloc[-flag_len:].max()
        flag_low = low.iloc[-flag_len:].min()
        if flag_high == 0:
            continue
        depth = (flag_high - flag_low) / flag_high
        if depth <= 0.25:
            return True
    return False


def _window_range_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    window_bars: int,
) -> tuple[float, float]:
    """Per-bar range over the last N bars.

    Returns (window_max_range_pct, window_avg_range_pct), where
      range_pct[i] = (high[i] - low[i]) / close[i].
    Max is what the pass condition checks; avg is the ranking key.
    """
    if len(close) < window_bars:
        return np.nan, np.nan
    range_pct = ((high - low) / close).replace([np.inf, -np.inf], np.nan)
    window = range_pct.iloc[-window_bars:]
    return float(window.max()), float(window.mean())


def compute_features(df: pd.DataFrame, *, heuristic_window_bars: int = 4) -> SymbolFeatures | None:
    """Compute features for one symbol. Returns None if history is insufficient."""
    if len(df) < _MIN_HISTORY_BARS:
        return None

    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].fillna(0.0)

    sma50 = close.rolling(50).mean().iloc[-1]
    sma150 = close.rolling(150).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    sma200_series = close.rolling(200).mean().iloc[-21:]
    sma200_slope = _slope(sma200_series.dropna())

    high_52w = high.iloc[-252:].max()
    low_52w = low.iloc[-252:].min()

    # Returns: 3m=63, 6m=126, 9m=189, 12m=252 trading bars.
    ret_3m = _safe_pct_change(close, 63)
    ret_6m = _safe_pct_change(close, 126)
    ret_9m = _safe_pct_change(close, 189)
    ret_12m = _safe_pct_change(close, 252)

    weighted_return = (
        0.4 * (ret_3m if pd.notna(ret_3m) else 0)
        + 0.2 * (ret_6m if pd.notna(ret_6m) else 0)
        + 0.2 * (ret_9m if pd.notna(ret_9m) else 0)
        + 0.2 * (ret_12m if pd.notna(ret_12m) else 0)
    )

    dollar_vol_20d = (close.iloc[-20:] * volume.iloc[-20:]).median()

    range_10 = high.iloc[-10:].max() - low.iloc[-10:].min()
    tightness_10d = range_10 / close.iloc[-1] if close.iloc[-1] else np.nan

    vol_10d_avg = volume.iloc[-10:].mean()
    vol_50d_avg = volume.iloc[-50:].mean()

    htf_flag = _detect_htf(close, high, low)
    window_max_range_pct, window_avg_range_pct = _window_range_features(
        high, low, close, window_bars=heuristic_window_bars
    )

    return SymbolFeatures(
        symbol=df["symbol"].iloc[-1],
        close=float(close.iloc[-1]),
        sma50=float(sma50),
        sma150=float(sma150),
        sma200=float(sma200),
        sma200_slope=float(sma200_slope),
        high_52w=float(high_52w),
        low_52w=float(low_52w),
        ret_3m=float(ret_3m) if pd.notna(ret_3m) else np.nan,
        ret_6m=float(ret_6m) if pd.notna(ret_6m) else np.nan,
        ret_9m=float(ret_9m) if pd.notna(ret_9m) else np.nan,
        ret_12m=float(ret_12m) if pd.notna(ret_12m) else np.nan,
        weighted_return=float(weighted_return),
        dollar_vol_20d=float(dollar_vol_20d),
        tightness_10d=float(tightness_10d),
        vol_10d_avg=float(vol_10d_avg),
        vol_50d_avg=float(vol_50d_avg),
        htf_flag=bool(htf_flag),
        window_max_range_pct=float(window_max_range_pct) if pd.notna(window_max_range_pct) else np.nan,
        window_avg_range_pct=float(window_avg_range_pct) if pd.notna(window_avg_range_pct) else np.nan,
    )


def build_universe_features(ohlcv: pd.DataFrame, *, heuristic_window_bars: int = 4) -> pd.DataFrame:
    """Compute features for every symbol; assign RS rank percentile across the universe."""
    if ohlcv.empty:
        return pd.DataFrame()

    rows = []
    for symbol, group in ohlcv.groupby("symbol", sort=False):
        feats = compute_features(group, heuristic_window_bars=heuristic_window_bars)
        if feats is None:
            continue
        rows.append(asdict(feats))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["rs_rank"] = df["weighted_return"].rank(pct=True) * 100.0
    return df


def apply_trend_template(
    features: pd.DataFrame,
    thresholds: CountryThresholds,
    *,
    require_tightness: bool = True,
) -> pd.DataFrame:
    """Apply pass criteria; add boolean columns and `passed_stage1`.

    When `require_tightness=False`, returns the pure Minervini trend template:
    only the 6 base criteria (SMA stack/slope, near-high, above-low, RS, liquidity).
    """
    if features.empty:
        return features.assign(passed_stage1=False)

    df = features.copy()

    df["pass_sma_stack"] = (df["close"] > df["sma50"]) & (df["sma50"] > df["sma150"]) & (df["sma150"] > df["sma200"])
    df["pass_sma200_slope"] = df["sma200_slope"] > 0
    df["pass_above_low"] = df["close"] >= thresholds.above_low_min * df["low_52w"]
    df["pass_near_high"] = df["close"] >= thresholds.near_high_min * df["high_52w"]
    df["pass_rs"] = df["rs_rank"] >= thresholds.rs_min
    df["pass_liquidity"] = df["dollar_vol_20d"] >= thresholds.liquidity_floor
    df["pass_tightness"] = (df["tightness_10d"] <= thresholds.tightness_max) & (df["vol_10d_avg"] < df["vol_50d_avg"])

    base = (
        df["pass_sma_stack"]
        & df["pass_sma200_slope"]
        & df["pass_above_low"]
        & df["pass_near_high"]
        & df["pass_rs"]
        & df["pass_liquidity"]
    )
    if require_tightness:
        # HTF candidates bypass the tightness pre-filter (flags are wider than VCPs).
        df["passed_stage1"] = base & (df["pass_tightness"] | df["htf_flag"])
    else:
        df["passed_stage1"] = base
    return df


def apply_heuristic_vcp(features: pd.DataFrame, thresholds: CountryThresholds) -> pd.DataFrame:
    """Numerical-only VCP detector. No charts, no LLM.

    Pass criteria: the trend-template base AND every bar in the last
    `heuristic_window_bars` (default 4) must have (high-low)/close <=
    `heuristic_max_bar_range` (default 0.20) AND price within
    `heuristic_near_high_min` of the 52w high.
    Rank by `window_avg_range_pct` ascending (tightest window first),
    `rs_rank` descending as tiebreak.
    """
    if features.empty:
        return features.assign(passed_heuristic=False)

    df = features.copy()

    # Base trend-template flags (mirror apply_trend_template, no tightness pre-filter).
    df["pass_sma_stack"] = (df["close"] > df["sma50"]) & (df["sma50"] > df["sma150"]) & (df["sma150"] > df["sma200"])
    df["pass_sma200_slope"] = df["sma200_slope"] > 0
    df["pass_above_low"] = df["close"] >= thresholds.above_low_min * df["low_52w"]
    df["pass_rs"] = df["rs_rank"] >= thresholds.rs_min
    df["pass_liquidity"] = df["dollar_vol_20d"] >= thresholds.liquidity_floor

    # Heuristic-specific flags.
    df["pass_near_high_strict"] = df["close"] >= thresholds.heuristic_near_high_min * df["high_52w"]
    # Equivalent to "every bar in window has range/close <= max_bar_range".
    df["pass_window_tight"] = df["window_max_range_pct"] <= thresholds.heuristic_max_bar_range

    df["passed_heuristic"] = (
        df["pass_sma_stack"]
        & df["pass_sma200_slope"]
        & df["pass_above_low"]
        & df["pass_near_high_strict"]
        & df["pass_rs"]
        & df["pass_liquidity"]
        & df["pass_window_tight"]
    )
    return df
