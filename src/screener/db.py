"""Postgres data loader. Single bulk query per country."""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def load_ohlcv(country: str, *, as_of: date, lookback_days: int = 730) -> pd.DataFrame:
    """Load long-form OHLCV for all symbols in a country within lookback window.

    Returns DataFrame with columns: symbol, date, open, high, low, close, volume.
    Sorted by (symbol, date). Volume is float to tolerate nulls/zeros.
    """
    start = as_of - timedelta(days=lookback_days)
    sql = text(
        """
        SELECT symbol, date, open, high, low, close, volume
        FROM stock_data
        WHERE country = :country
          AND date >= :start
          AND date <= :as_of
        ORDER BY symbol, date
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params={"country": country, "start": start, "as_of": as_of})

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def latest_trading_date(country: str) -> date:
    sql = text("SELECT MAX(date) FROM stock_data WHERE country = :country")
    with get_engine().connect() as conn:
        result = conn.execute(sql, {"country": country}).scalar()
    if result is None:
        raise RuntimeError(f"No data for country={country}")
    return result
