"""Stage 2: composite chart rendering via mplfinance.

Produces one PNG per symbol with three panels:
  - Top: weekly candles (2y), log scale, 10/30/40-week SMAs
  - Middle: daily candles (~6m), SMA50/150/200, 52w-high pivot hline
  - Bottom: daily volume + 50d volume MA
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for batch runs.

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd


_STYLE = mpf.make_mpf_style(
    base_mpf_style="yahoo",
    gridstyle="",
    facecolor="white",
    rc={"axes.edgecolor": "#888", "axes.labelcolor": "#333"},
)


def _to_mpf(df: pd.DataFrame) -> pd.DataFrame:
    """mplfinance wants a DatetimeIndex and columns Open/High/Low/Close/Volume."""
    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.set_index("date").sort_index()
    # Some symbols have duplicate (symbol, date) rows in the loader; keep the last.
    out = out[~out.index.duplicated(keep="last")]
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    return out


def _weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    rule = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    return df_daily.resample("W-FRI").agg(rule).dropna()


def render_chart(df_daily: pd.DataFrame, symbol: str, out_path: Path) -> Path:
    """Render the composite chart for `symbol` and write a PNG to `out_path`.

    `df_daily` is a long-form OHLCV frame for ONE symbol (cols: date, open, high, low, close, volume).
    Returns the resolved output path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    daily = _to_mpf(df_daily)
    weekly_full = _weekly(daily)
    weekly = weekly_full.iloc[-104:] if len(weekly_full) > 104 else weekly_full
    daily_6m = daily.iloc[-126:] if len(daily) > 126 else daily

    # 52-week high (pivot proxy) computed from the FULL daily window.
    pivot = float(daily["High"].iloc[-252:].max()) if len(daily) >= 1 else float("nan")

    fig = plt.figure(figsize=(12, 14), dpi=100, facecolor="white")
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[4, 4, 2],
        hspace=0.25,
        left=0.07,
        right=0.97,
        top=0.96,
        bottom=0.05,
    )
    ax_weekly = fig.add_subplot(gs[0])
    ax_daily = fig.add_subplot(gs[1])
    ax_vol = fig.add_subplot(gs[2])

    # --- Top: weekly with log scale + 10/30/40-week MAs ---
    weekly_mas = [m for m in (10, 30, 40) if len(weekly) >= m]
    mpf.plot(
        weekly,
        type="candle",
        style=_STYLE,
        ax=ax_weekly,
        mav=tuple(weekly_mas) if weekly_mas else (),
        warn_too_much_data=10_000,
        axtitle=f"{symbol}  —  Weekly (log)  •  10/30/40W MA",
    )
    ax_weekly.set_yscale("log")

    # --- Middle: daily with SMA50/150/200 + 52w-high hline ---
    daily_for_mas = daily.iloc[-(126 + 200):] if len(daily) > 126 else daily
    sma_overlays = []
    for window, color in [(50, "#1f77b4"), (150, "#ff7f0e"), (200, "#d62728")]:
        if len(daily_for_mas) >= window:
            sma_series = daily_for_mas["Close"].rolling(window).mean().reindex(daily_6m.index)
            sma_overlays.append(
                mpf.make_addplot(sma_series, ax=ax_daily, color=color, width=1.0)
            )
    mpf.plot(
        daily_6m,
        type="candle",
        style=_STYLE,
        ax=ax_daily,
        addplot=sma_overlays if sma_overlays else None,
        warn_too_much_data=10_000,
        axtitle=f"Daily (last 6m)  •  SMA 50/150/200  •  pivot ≈ {pivot:.2f}",
    )
    ax_daily.axhline(pivot, color="#444", linewidth=1.0, linestyle="--", alpha=0.7)

    # --- Bottom: daily volume + 50d MA ---
    vol_window = daily.iloc[-126:] if len(daily) > 126 else daily
    ax_vol.bar(vol_window.index, vol_window["Volume"], color="#888", width=1.0)
    if len(daily) >= 50:
        vol_ma = daily["Volume"].rolling(50).mean().reindex(vol_window.index)
        ax_vol.plot(vol_window.index, vol_ma, color="#d62728", linewidth=1.2, label="50d Vol MA")
        ax_vol.legend(loc="upper left", fontsize=8, frameon=False)
    ax_vol.set_title("Volume", loc="left", fontsize=10)
    ax_vol.grid(False)

    fig.savefig(out_path, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path
