"""Configuration: thresholds, settings, env overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CountryThresholds:
    liquidity_floor: float
    rs_min: float = 70.0
    tightness_max: float = 0.10
    near_high_min: float = 0.75
    above_low_min: float = 1.30
    # Heuristic-VCP params (used by --mode heuristic):
    heuristic_window_bars: int = 4             # check the last N bars
    heuristic_max_bar_range: float = 0.20      # every bar in the window must have range/close <= this
    heuristic_near_high_min: float = 0.85      # close >= ratio * 52w high (tighter than trend's 0.75)


COUNTRY_THRESHOLDS: dict[str, CountryThresholds] = {
    "usa": CountryThresholds(liquidity_floor=5_000_000),
    "india": CountryThresholds(liquidity_floor=50_000_000),  # INR placeholder
}

CLI_TO_DB_COUNTRY = {"us": "usa", "in": "india", "eu": "eu"}


def _default_database_url() -> str:
    if url := os.getenv("DATABASE_URL"):
        return url
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5433")
    name = os.getenv("DB_NAME", "stockdata")
    user = os.getenv("DB_USER", "user")
    pwd = os.getenv("DB_PASSWORD", "password")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{name}"


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=_default_database_url)
    vcp_model: str = field(default_factory=lambda: os.getenv("VCP_MODEL", "gemini/gemini-flash-latest"))
    concurrency: int = field(default_factory=lambda: int(os.getenv("VCP_CONCURRENCY", "4")))
    grade_timeout_s: int = field(default_factory=lambda: int(os.getenv("VCP_GRADE_TIMEOUT", "60")))
    examples_dir: Path = field(default_factory=lambda: REPO_ROOT / "examples")
    out_dir: Path = field(default_factory=lambda: REPO_ROOT / "out")


def get_settings() -> Settings:
    return Settings()


def resolve_country(cli_code: str) -> str:
    code = cli_code.lower().strip()
    if code in COUNTRY_THRESHOLDS:
        return code
    if code in CLI_TO_DB_COUNTRY:
        return CLI_TO_DB_COUNTRY[code]
    raise ValueError(f"Unknown country code: {cli_code}")
