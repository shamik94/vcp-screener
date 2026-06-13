"""Stage 4: merge grades with Stage-1 features, filter, rank, write CSV+MD."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


_MD_COLS = [
    "symbol",
    "country",
    "grade",
    "setup",
    "pivot_price_estimate",
    "pct_below_pivot",
    "contraction_depths_pct",
    "rs_rank",
    "notes",
]


def _setup_label(row: pd.Series) -> str:
    tags = []
    if row.get("is_vcp"):
        tags.append("VCP")
    if row.get("is_htf"):
        tags.append("HTF")
    return "/".join(tags) or "?"


def build_watchlist(
    grades: list[dict[str, Any]],
    candidates: pd.DataFrame,
    *,
    min_grade: float = 60.0,
    top_n: int = 50,
) -> pd.DataFrame:
    if not grades:
        return pd.DataFrame(columns=_MD_COLS)
    g = pd.DataFrame(grades)

    # Defensive: missing fields default to safe values.
    for col, default in [
        ("is_vcp", False),
        ("is_htf", False),
        ("already_broken_out", False),
        ("grade", 0),
        ("pivot_price_estimate", float("nan")),
        ("pct_below_pivot", float("nan")),
        ("contraction_depths_pct", None),
        ("notes", ""),
    ]:
        if col not in g.columns:
            g[col] = default

    merged = g.merge(
        candidates[["symbol", "rs_rank"]].assign(country="usa"),
        on="symbol",
        how="left",
    )
    keep = (
        (merged["is_vcp"].fillna(False) | merged["is_htf"].fillna(False))
        & (~merged["already_broken_out"].fillna(False))
        & (merged["grade"].fillna(0) >= min_grade)
    )
    filtered = merged[keep].copy()
    if filtered.empty:
        return pd.DataFrame(columns=_MD_COLS)

    filtered["setup"] = filtered.apply(_setup_label, axis=1)
    filtered = filtered.sort_values(["grade", "rs_rank"], ascending=[False, False]).head(top_n)
    return filtered[_MD_COLS]


def write_csv(watchlist: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    watchlist.to_csv(path, index=False)


def write_md(watchlist: pd.DataFrame, path: Path, *, summary: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if summary:
        lines.append(f"_{summary}_\n")
    if watchlist.empty:
        lines.append("No qualifying setups today.\n")
    else:
        headers = ["#"] + _MD_COLS
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for i, (_, row) in enumerate(watchlist.iterrows(), start=1):
            cells = [str(i)]
            for col in _MD_COLS:
                val = row[col]
                if isinstance(val, float):
                    cells.append(f"{val:.2f}")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
