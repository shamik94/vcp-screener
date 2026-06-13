# VCP Vision Screener

Two-stage nightly screener: deterministic Minervini trend-template filter, then a vision LLM grades chart images for VCP / HTF setups. Outputs a ranked top-50 watchlist.

## Setup

```bash
make install
```

The screener reads from Postgres `stock_data(symbol, date, open, high, low, close, volume, country)`. Default connection: `postgresql://user:password@localhost:5433/stockdata`. Override via `DATABASE_URL` or `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD`.

## Reference images for grading

Drop hand-picked chart PNGs into:
- `examples/vcp/` — valid VCP setups (used as positive few-shot examples, up to 4)
- `examples/not_vcp/` — non-VCP charts (optional negatives, up to 2)

These are also the ground truth for `make calibrate`.

## Run

```bash
make screen        # full pipeline: trend template -> charts -> LLM grading -> watchlist
make heuristic     # numerical VCP detector (tight-day count + range contraction); no LLM
make trend         # Minervini trend template ONLY (no charts, no LLM); top 50 by RS rank
make calibrate     # grader-only, against examples/
make ui            # Streamlit UI to browse survivors + view charts (localhost:8501)
make test          # unit tests
```

## UI

`make ui` launches a Streamlit app at `http://localhost:8501`. The pipeline runs
once per (country, as-of-date) and is held in memory — flipping between Heuristic
/ Trend / VCP modes does not re-query Postgres. Features:

- Sortable table (RS rank, tightness, etc. — click any column header)
- Sidebar filters: min RS, max window-avg-range
- Click a row → composite chart appears below (cached at `out/charts/{date}/`,
  so CLI runs and UI share the same on-disk cache)
- VCP mode: opt-in "Grade with LLM" button; results land in `st.session_state`
- "Refresh data" button to bust the cache and re-pull

CLI flags:

```bash
python -m src.screener.run --countries us --date 2026-06-12 [options]
  --mode {vcp,trend,heuristic}
       vcp       = full pipeline with LLM grading (default)
       trend     = pure Minervini, no tightness pre-filter, no charts, no grading
       heuristic = trend template + numerical VCP detector, no LLM
  --top N         Top-N to keep in trend/heuristic mode (default 50)
  --skip-grade    Stages 1+2 only (renders charts, skips LLM). vcp mode only.
  --limit N       Cap charts/grades for debug
```

### Three modes

- **`--mode vcp`** (default): adds a tightness pre-filter (10-day range ≤ 10% AND
  10d volume < 50d volume) to bias toward bases, then renders composite charts
  and asks a vision LLM to grade each one for VCP/HTF structure. Expect ~10-50
  charts/day out of a ~1k-symbol universe.
- **`--mode heuristic`**: trend template + a per-bar tight-window rule, no
  LLM. A symbol passes when EVERY one of the last N bars (default N=4) has
  daily range `(high-low)/close ≤ 0.20` AND price is within 15% of the 52w
  high. Survivors are ranked by `window_avg_range_pct` ascending (tightest
  window first), with `rs_rank` as tiebreak. The 0.20 ceiling is a coarse
  filter — its job is to drop wild-volatility days; the real signal is the
  ranking. To require a more compressed window, lower
  `heuristic_max_bar_range` in `config.py`. Writes `out/heuristic_{date}_{country}.csv`.
- **`--mode trend`**: pure Minervini trend template — the 6 base criteria only
  (SMA stack, SMA200 slope, near 52w high, above 52w low, RS ≥ 70, liquidity).
  No tightness filter, no rendering, no LLM. Useful as a quick daily scan or
  as input to a different downstream process. Writes `out/trend_{date}_{country}.csv`.

All heuristic thresholds live in `CountryThresholds` in `src/screener/config.py`
(prefixed `heuristic_*`) and are tunable without code changes.

## Model selection

Set `VCP_MODEL` to any LiteLLM-supported vision model:
- `gemini/gemini-flash-latest` (default; needs `GEMINI_API_KEY`)
- `anthropic/claude-haiku-latest` (needs `ANTHROPIC_API_KEY`)
- `ollama/qwen2.5vl:7b` (local)

Concurrency via `VCP_CONCURRENCY` (default 4), timeout via `VCP_GRADE_TIMEOUT` (default 60s).

## Output

Everything lands in `out/`:
- `candidates_{date}.csv` — every symbol scored, with pass flags (vcp/trend modes)
- `heuristic_{date}_{country}.csv` — top-N heuristic VCP survivors (`--mode heuristic`)
- `trend_{date}_{country}.csv` — top-N trend-template survivors (`--mode trend`)
- `charts/{date}/{symbol}.png` — composite weekly/daily/volume charts (vcp mode)
- `grades_{date}.json` — raw grader output (vcp mode)
- `watchlist_{date}.{csv,md}` — final ranked top 50 (vcp mode)
- `grade_failures_{date}.log` — JSON parse / timeout failures (vcp mode)
