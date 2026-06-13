"""Smoke test: render a synthetic chart and assert the PNG exists and is non-empty."""
from __future__ import annotations

import numpy as np

from src.screener.render import render_chart
from tests.conftest import make_ohlcv


def test_render_chart_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    closes = np.linspace(50, 200, 504) + rng.normal(0, 1.0, 504)
    df = make_ohlcv("TEST", closes, start="2023-01-02")
    out = tmp_path / "TEST.png"
    result = render_chart(df, "TEST", out)
    assert result.exists()
    assert result.stat().st_size > 5_000  # any real PNG with three panels exceeds 5KB.
