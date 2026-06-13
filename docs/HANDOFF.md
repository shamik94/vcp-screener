# VCP Screener — Handoff

Last verified: 2026-06-12

## What this is

A nightly screener that finds US stocks currently sitting in a Volatility
Contraction Pattern (VCP) or High Tight Flag (HTF) base **before** they break
out, ranks them, and writes a top-50 watchlist.

Pipeline:

```
Postgres stock_data  →  Stage 1 trend template (deterministic)
                          ↓ survivors (~30-50 / day from ~1k universe)
                        Stage 2 render composite chart PNG (mplfinance)
                          ↓
                        Stage 3 vision LLM grades each chart (LiteLLM)
                          ↓ grade ≥ 60, not broken out
                        Stage 4 rank + write watchlist (CSV + MD)
```

USA-only for v1. Country-keyed config (`COUNTRY_THRESHOLDS` in
`src/screener/config.py`) makes India/EU a config addition once their data
warrants it.

## Current state

| Stage | Status | Verified |
|---|---|---|
| 1 — trend template | Done | 11 unit tests + live run: 1174 symbols → 40 survivors |
| 2 — render | Done | Spot-checked one chart (`out/charts/2026-06-12/ABCB.png`) |
| 3 — grader | Done | Code path imports; **not yet run** — needs reference images + API key |
| 4 — report | Done | Code path; **not yet run end-to-end** |

The grader and report have not been exercised against the live model. Stage 3
is the next thing to validate; see "What to do next."

## Repo layout

```
src/screener/
  config.py        Settings, thresholds, env overrides, country mapping
  db.py            SQLAlchemy engine + single-query OHLCV loader
  trend_template.py  Stage 1: features + pass criteria + HTF detector
  render.py        Stage 2: 3-panel composite via mplfinance
  grade.py         Stage 3: async LiteLLM calls + JSON parsing + retries
  report.py        Stage 4: rank + CSV + MD
  run.py           End-to-end CLI (python -m src.screener.run)
  calibrate.py     Grader-only CLI (python -m src.screener.calibrate)
  prompt.txt       System prompt for the grader — edit freely, no code change needed
tests/             pytest, synthetic OHLCV fixtures
examples/
  vcp/             positive reference PNGs (up to 4 sent as few-shot)
  not_vcp/         negative reference PNGs (up to 2 sent as few-shot)
out/               gitignored: candidates_*.csv, charts/*, grades_*.json, watchlist_*
prompt.txt is at src/screener/prompt.txt
```

## How to operate

```bash
make install     # one-time, in a venv
make test        # unit tests for Stage 1
make calibrate   # grade reference images only, prints accuracy
make screen      # full pipeline (VCP mode); writes to out/
make heuristic   # numerical VCP detector; no charts, no LLM, top 50 by tightest contraction
make trend       # Minervini trend template only — no charts, no LLM, top 50 by RS
make ui          # Streamlit visualizer (localhost:8501); pipeline cached in memory
```

CLI flags on `run.py`:

```
--countries us                comma-separated; 'us' → DB country='usa'
--date YYYY-MM-DD             defaults to today
--mode {vcp,trend,heuristic}  vcp = full pipeline (default).
                              trend = pure Minervini, no tightness filter, no charts, no LLM.
                              heuristic = trend + numerical VCP detector, no LLM.
--top N                       top-N in trend/heuristic mode (default 50)
--skip-grade                  Stages 1+2 only; renders charts but skips LLM (vcp mode)
--limit N                     cap charts/grades for debug
```

### Heuristic mode internals

Defined in `trend_template.apply_heuristic_vcp`. A symbol passes when the trend
template base criteria pass AND:
- **Tight window**: for every bar in the last `heuristic_window_bars` (default 4),
  `(high - low) / close ≤ heuristic_max_bar_range` (default 0.20). One wide bar
  in the window kills the signal — there is NO shakeout tolerance in this mode
  (you trade tolerance for the simplicity of an exact per-bar rule).
- **Near pivot**: `close ≥ heuristic_near_high_min × high_52w` (default 0.85,
  tighter than the trend template's 0.75).

The 0.20 ceiling is intentionally loose; its only job is to drop bars with
extreme intraday moves (gap-and-fail days, news events). The real signal is the
ranking: results are sorted by `window_avg_range_pct` ascending (tightest
window first), with `rs_rank` descending as the tiebreak.

All thresholds live in `CountryThresholds` (`config.py`). To get fewer/cleaner
candidates, lower `heuristic_max_bar_range` (e.g. 0.05 for 5%-or-tighter bars).
To require a longer base, raise `heuristic_window_bars`.

### Connecting to Postgres

Default `postgresql://user:password@localhost:5433/stockdata` (the existing
docker container). Override with `DATABASE_URL` or the `DB_*` env vars listed
in `config.py`.

### Selecting a model

`VCP_MODEL` env var. Any LiteLLM-supported vision model:
- `gemini/gemini-flash-latest` (default; needs `GEMINI_API_KEY`)
- `anthropic/claude-haiku-latest` (needs `ANTHROPIC_API_KEY`)
- `ollama/qwen2.5vl:7b` (local; needs `ollama serve`)

Other knobs: `VCP_CONCURRENCY` (default 4), `VCP_GRADE_TIMEOUT` (seconds, default 60).

## Where to make changes

| If you want to... | Edit |
|---|---|
| Change Stage 1 thresholds (RS, liquidity, tightness, 52w-high cushion) | `src/screener/config.py` → `CountryThresholds` |
| Add a country | `src/screener/config.py` → `COUNTRY_THRESHOLDS` and `CLI_TO_DB_COUNTRY` |
| Tune the grader prompt (definitions, hard rules, output schema) | `src/screener/prompt.txt` — no code change required |
| Change which features are passed to the grader | `GradeRequest` + `_format_features` in `grade.py`, populate in `run.py` |
| Change chart layout | `src/screener/render.py` |
| Change the final filter (min grade, top N) | `src/screener/report.py` → `build_watchlist` defaults |
| Tweak UI (columns, filters, layout) | `src/screener/ui.py` |

## Architectural decisions worth knowing

1. **One bulk query per country**, not per symbol. `db.load_ohlcv` returns
   long-form OHLCV for the whole country; Stage 1 groups in pandas. Scales
   to ~10k symbols comfortably.

2. **Stage 1 is the funnel**. ~3% pass rate is typical. If you find you're
   sending too many or too few charts to the grader, tune `tightness_max`,
   `near_high_min`, or `rs_min` in `config.py` — don't try to fix it in
   the LLM prompt.

3. **The grader prompt is in a separate file** (`prompt.txt`). This is
   deliberate: prompt iteration happens far more often than code changes,
   and putting it in code makes diffs noisy. The prompt is loaded once at
   import (`@lru_cache`).

4. **Precomputed features are passed to the grader.** The prompt instructs
   the model to trust them over visual measurement. Currently passing
   `latest_close`, `high_52w`, `pct_below_52w_high`, `range_10d_pct`, and
   `vol_10d/vol_50d` ratio. Add more by extending `GradeRequest` and
   `_format_features`.

5. **Calibration mode skips few-shot** (`include_few_shot=False`). Otherwise
   the model would just echo whatever's in `examples/vcp/`, defeating the
   purpose of measuring accuracy on those images.

6. **Rendering is sequential.** mplfinance shares global pyplot state and
   isn't thread-safe. For ~50-200 charts/run this is fine (sub-second per
   chart). If the universe grows to >1000 survivors, move to a process pool.

7. **Grader concurrency uses asyncio.** `asyncio.Semaphore(VCP_CONCURRENCY)`
   bounds in-flight calls. JSON parse failures get one retry with a stricter
   system message; everything else (timeouts, provider errors) is logged to
   `out/grade_failures_{date}.log` and dropped from the watchlist.

8. **Sub-process robustness**: one bad symbol (render error, grade timeout,
   missing feature) never kills the run. It's logged and skipped.

## Known gotchas

- **Duplicate (symbol, date) rows in the DB.** At least `ABCB` has them as of
  writing. `render.py` dedupes with `keep="last"`. If duplicates become
  common, fix the loader instead.
- **EU has no data yet.** `--countries eu` will run but find nothing.
- **India has 9 symbols.** RS percentile is meaningless at that universe size.
  Wait for the loader to thicken India coverage before enabling it.
- **`min_periods` not set on rolling SMAs.** Symbols with < 200 bars get NaN
  SMA200 and are dropped by `compute_features` (insufficient history). This is
  intentional.
- **HTF detector is conservative.** Looks back only ~12 weeks. If you're
  hunting for HTFs aggressively, widen `bars_8w + flag_max` window in
  `trend_template._detect_htf`.
- **Prompt file path is hardcoded** as `src/screener/prompt.txt`. If you move
  the prompt, update `_PROMPT_PATH` in `grade.py`.

## What to do next

In order of priority:

1. **Drop 3-5 reference VCP charts** into `examples/vcp/` and 1-2 negatives
   into `examples/not_vcp/`. PNGs only. Use charts you'd grade ~90+ yourself.
2. **`export GEMINI_API_KEY=...`** (or set `VCP_MODEL` to a different provider).
3. **`make calibrate`** — read the per-image JSON output. Iterate on
   `prompt.txt` until positives grade ≥ 60 and negatives grade < 60.
4. **`make screen`** — full pipeline. Spot-check the top 10 in
   `out/watchlist_*.md`. If too many false positives, tighten the prompt or
   raise `min_grade` in `report.py`.
5. **Schedule it** — a cron entry that runs `make screen` ~30 min after the
   loader finishes each market day.

Deferred (not blocked, just not in scope yet):

- Backtest the watchlist's 1/2/4-week forward performance to score the grader.
- Add `--countries in` once India universe is >100 symbols.
- Email / Slack delivery of the daily watchlist MD.
- Web UI to browse historical watchlists.

## How to verify a working install

```bash
make test                                          # 11 tests, all pass
.venv/bin/python -m src.screener.run \
    --countries us --skip-grade --limit 5         # should log "rendered N charts"
ls out/charts/$(date +%F)/                         # 5 PNGs
open out/charts/$(date +%F)/$(ls out/charts/$(date +%F) | head -1)  # eyeball one chart
```

If those three steps work, Stages 1 and 2 are healthy. Then add a reference
image and an API key and run `make calibrate` to validate Stage 3.

## Useful references

- The original task spec: see git history / the initial prompt.
- Plan file (architectural decisions): `~/.claude/plans/quizzical-prancing-glacier.md`
- Minervini, *Trade Like a Stock Market Wizard* — the source for the
  trend template and VCP/HTF definitions.
