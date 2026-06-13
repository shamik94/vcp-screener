"""Streamlit UI: browse + sort screener survivors, view charts, opt-in LLM grading.

Launch with `make ui` (or `streamlit run src/screener/ui.py`).

Pipeline runs once per (country, as_of) and is cached in memory via @st.cache_data.
Charts render lazily on row selection and are reused from out/charts/{date}/{symbol}.png
when present (so CLI runs and UI share the same on-disk cache).
"""
from __future__ import annotations

import sys
from datetime import date as date_cls
from pathlib import Path

# Streamlit invokes this file as a script (not `python -m`), so the repo root
# isn't on sys.path by default — add it before importing src.screener.*.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.screener import db
from src.screener.config import COUNTRY_THRESHOLDS, get_settings
from src.screener.grade import GradeRequest, grade_all_sync
from src.screener.render import render_chart
from src.screener.report import build_watchlist
from src.screener.trend_template import (
    apply_heuristic_vcp,
    apply_trend_template,
    build_universe_features,
)

st.set_page_config(page_title="VCP Screener", layout="wide")


@st.cache_data(show_spinner="Loading OHLCV and computing features…")
def load_pipeline(country: str, as_of_iso: str):
    as_of = date_cls.fromisoformat(as_of_iso)
    ohlcv = db.load_ohlcv(country, as_of=as_of)
    thresholds = COUNTRY_THRESHOLDS[country]
    feats = build_universe_features(ohlcv, heuristic_window_bars=thresholds.heuristic_window_bars)
    trend_pure = apply_trend_template(feats, thresholds, require_tightness=False)
    trend_vcp = apply_trend_template(feats, thresholds, require_tightness=True)
    heur = apply_heuristic_vcp(feats, thresholds)
    return ohlcv, trend_pure, trend_vcp, heur


@st.cache_data(show_spinner=False)
def load_info(country: str) -> pd.DataFrame:
    return db.load_stock_info(country)


@st.cache_data(show_spinner="Rendering chart…")
def get_chart_path(symbol: str, as_of_iso: str, _ohlcv: pd.DataFrame) -> str:
    """Render (or reuse) {symbol}.png for the given date. _ohlcv prefix skips hashing."""
    out_dir = get_settings().out_dir / "charts" / as_of_iso
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}.png"
    if not path.exists():
        df_sym = _ohlcv[_ohlcv["symbol"] == symbol]
        render_chart(df_sym, symbol, path)
    return str(path)


def _pie_charts(df: pd.DataFrame, label: str) -> None:
    """Render side-by-side sector and industry pie charts for the given DataFrame."""
    if df.empty:
        return
    has_sector = "sector" in df.columns
    has_industry = "industry" in df.columns
    if not has_sector and not has_industry:
        return

    st.subheader(f"{label} — Industry & Sector Breakdown")
    col1, col2 = st.columns(2)

    if has_sector:
        counts = df["sector"].fillna("Unknown").value_counts().reset_index()
        counts.columns = ["sector", "count"]
        with col1:
            fig = px.pie(counts, values="count", names="sector", title="By Sector")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    if has_industry:
        counts = df["industry"].fillna("Unknown").value_counts().reset_index()
        counts.columns = ["industry", "count"]
        with col2:
            fig = px.pie(counts, values="count", names="industry", title="By Industry")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)


# ---------- Sidebar ----------
st.sidebar.title("VCP Screener")
country = st.sidebar.selectbox("Country", list(COUNTRY_THRESHOLDS.keys()), index=0)
as_of = st.sidebar.date_input("As of", value=date_cls.today())
mode = st.sidebar.radio("Mode", ["Heuristic", "Trend", "VCP (LLM)"], index=0)

# Load info early so sector/industry options can be populated in sidebar.
info = load_info(country)
_all_sectors = sorted(info["sector"].dropna().unique().tolist()) if not info.empty else []

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")
min_rs = st.sidebar.slider("Min RS rank", 0, 100, 70)
max_avg_range = st.sidebar.slider(
    "Max window avg range (heuristic/VCP)", 0.01, 0.30, 0.10, 0.005,
    help="Average per-bar (high-low)/close over the heuristic window. Ignored in Trend mode.",
)
symbol_search = st.sidebar.text_input("Symbol", placeholder="e.g. AAPL").upper().strip()
selected_sectors = st.sidebar.multiselect("Sector", _all_sectors)
# Industry list cascades from selected sectors.
_industry_pool = (
    info[info["sector"].isin(selected_sectors)] if selected_sectors else info
)
_all_industries = sorted(_industry_pool["industry"].dropna().unique().tolist()) if not _industry_pool.empty else []
selected_industries = st.sidebar.multiselect("Industry", _all_industries)
htf_only = st.sidebar.checkbox("HTF setups only")

if st.sidebar.button("Refresh data", help="Clear cache and re-pull from Postgres"):
    load_pipeline.clear()
    load_info.clear()
    get_chart_path.clear()
    st.session_state.pop("watchlist", None)
    st.session_state.pop("selected_idx", None)
    st.rerun()

# ---------- Load ----------
as_of_iso = as_of.isoformat()
ohlcv, trend_pure, trend_vcp, heur = load_pipeline(country, as_of_iso)

# ---------- Mode → source df + default sort ----------
if mode == "Heuristic":
    src = heur[heur["passed_heuristic"]].copy()
    sort_col, sort_asc = "window_avg_range_pct", True
    apply_tightness_filter = True
elif mode == "Trend":
    src = trend_pure[trend_pure["passed_stage1"]].copy()
    sort_col, sort_asc = "rs_rank", False
    apply_tightness_filter = False
else:  # VCP (LLM)
    src = trend_vcp[trend_vcp["passed_stage1"]].copy()
    sort_col, sort_asc = "rs_rank", False
    apply_tightness_filter = True

# Numeric filters
src = src[src["rs_rank"] >= min_rs]
if apply_tightness_filter and "window_avg_range_pct" in src.columns:
    src = src[src["window_avg_range_pct"] <= max_avg_range]

# Merge sector / industry
if not info.empty:
    src = src.merge(info, on="symbol", how="left")

# Categorical / text filters (applied after merge so sector/industry columns exist)
if symbol_search:
    src = src[src["symbol"].str.contains(symbol_search, case=False, na=False)]
if selected_sectors and "sector" in src.columns:
    src = src[src["sector"].isin(selected_sectors)]
if selected_industries and "industry" in src.columns:
    src = src[src["industry"].isin(selected_industries)]
if htf_only and "htf_flag" in src.columns:
    src = src[src["htf_flag"].astype(bool)]

display_cols = [
    "symbol", "sector", "industry", "close", "high_52w", "rs_rank",
    "window_avg_range_pct", "window_max_range_pct",
    "tightness_10d", "htf_flag",
]
display_cols = [c for c in display_cols if c in src.columns]
df_display = src[display_cols].sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

# Reset chart navigation when the active filter set changes.
_filter_key = (country, as_of_iso, mode, min_rs, max_avg_range,
               symbol_search, tuple(selected_sectors), tuple(selected_industries), htf_only)
if st.session_state.get("_filter_key") != _filter_key:
    st.session_state["_filter_key"] = _filter_key
    st.session_state.pop("selected_idx", None)

# ---------- Header ----------
st.title(f"{mode} — {country.upper()} — {as_of_iso}")
st.caption(
    f"universe={ohlcv['symbol'].nunique()}  •  survivors={len(df_display)}  "
    f"•  sort={sort_col} ({'asc' if sort_asc else 'desc'})"
)

# ---------- Table ----------
event = st.dataframe(
    df_display,
    width="stretch",
    on_select="rerun",
    selection_mode="single-row",
    hide_index=False,
    column_config={
        "close": st.column_config.NumberColumn(format="%.2f"),
        "high_52w": st.column_config.NumberColumn(format="%.2f"),
        "rs_rank": st.column_config.NumberColumn(format="%.1f"),
        "window_avg_range_pct": st.column_config.NumberColumn(format="%.3f"),
        "window_max_range_pct": st.column_config.NumberColumn(format="%.3f"),
        "tightness_10d": st.column_config.NumberColumn(format="%.3f"),
    },
)

# ---------- Sector / Industry pie charts ----------
_pie_charts(df_display, mode)

# ---------- VCP grading button ----------
if mode == "VCP (LLM)":
    settings = get_settings()
    if st.button(
        f"Grade {len(df_display)} charts with {settings.vcp_model}",
        disabled=df_display.empty,
    ):
        with st.spinner(f"Grading {len(df_display)} charts…"):
            reqs = []
            for row in src.itertuples():
                chart_path = Path(get_chart_path(row.symbol, as_of_iso, ohlcv))
                reqs.append(
                    GradeRequest(
                        symbol=row.symbol,
                        chart_path=chart_path,
                        close=float(row.close),
                        high_52w=float(row.high_52w),
                        tightness_10d=float(getattr(row, "tightness_10d", 0.0) or 0.0),
                        vol_10d_avg=float(getattr(row, "vol_10d_avg", 0.0) or 0.0),
                        vol_50d_avg=float(getattr(row, "vol_50d_avg", 0.0) or 0.0),
                    )
                )
            grades = grade_all_sync(reqs, out_dir=settings.out_dir, date_str=as_of_iso)
            st.session_state["watchlist"] = build_watchlist(grades, src)
    if "watchlist" in st.session_state:
        watchlist = st.session_state["watchlist"].copy()
        if not info.empty:
            watchlist = watchlist.merge(info, on="symbol", how="left")
        st.subheader("Graded watchlist")
        st.dataframe(watchlist, width="stretch")
        _pie_charts(watchlist, "Watchlist")

# ---------- Selected chart ----------
# Sync dataframe click → session state, but not if a nav button just fired.
df_selected_rows = event.selection.rows if event and event.selection else []
nav_triggered = st.session_state.pop("_nav_triggered", False)
if not nav_triggered and df_selected_rows:
    st.session_state["selected_idx"] = df_selected_rows[0]

selected_idx = st.session_state.get("selected_idx")
if selected_idx is not None and selected_idx < len(df_display):
    symbol = df_display.iloc[selected_idx]["symbol"]

    col_prev, col_sym, col_next = st.columns([1, 10, 1])
    with col_prev:
        if st.button("◀", disabled=(selected_idx == 0), use_container_width=True):
            st.session_state["selected_idx"] = selected_idx - 1
            st.session_state["_nav_triggered"] = True
            st.rerun()
    with col_sym:
        st.subheader(f"{symbol}  ({selected_idx + 1} / {len(df_display)})")
    with col_next:
        if st.button("▶", disabled=(selected_idx >= len(df_display) - 1), use_container_width=True):
            st.session_state["selected_idx"] = selected_idx + 1
            st.session_state["_nav_triggered"] = True
            st.rerun()

    path = get_chart_path(symbol, as_of_iso, ohlcv)
    st.image(path)
else:
    st.info("Click a row to view its chart.")
