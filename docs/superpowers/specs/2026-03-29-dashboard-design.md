# Cassandra Dashboard — Design Spec

**Date:** 2026-03-29
**Phase:** 6 — Reflex Dashboard (implemented as Streamlit)
**Scope:** Personal, local-only. Not public-facing. No authentication required.

---

## Goal

Build a two-page Streamlit dashboard that lets the owner monitor live bot signals and evaluate strategy readiness before committing real capital. The dashboard is the primary go/no-go decision tool for the two-week live evaluation period.

---

## Architecture

```
dashboard/
├── app.py                        # Entry point: streamlit run dashboard/app.py
├── pages/
│   ├── 1_Live_Monitor.py         # Page 1: price chart, signals, balance
│   └── 2_Strategy_Evaluation.py  # Page 2: calibration, backtest log, shadow pipeline
└── utils.py                      # Shared helpers: data loading, log parsing
```

**No FastAPI, no backend server, no authentication.** Streamlit runs as a single Python process and imports project modules directly. All external calls (Binance public OHLCV, Anthropic Claude, Binance testnet balance) are made from within Streamlit.

**Auto-refresh:** Use `st.cache_data(ttl=60)` on all data-fetching functions so cached results expire every 60 seconds. A `st_autorefresh(interval=60000)` call (from the `streamlit-autorefresh` package) triggers a page rerun every 60 seconds without blocking the UI thread. This keeps the dashboard fully interactive — tab switching and scrolling work normally during the refresh interval.

**Async bridge:** Streamlit runs its own asyncio event loop internally (via Tornado). Calling `asyncio.run()` directly inside a Streamlit script raises `RuntimeError: This event loop is already running`. The fix is `nest_asyncio`, which patches the running loop to allow nested calls. Applied once at module level: `nest_asyncio.apply()`. After that, `asyncio.run(scrape_crypto_headlines())` works correctly within the Streamlit process.

**New dependencies:** `streamlit`, `streamlit-autorefresh`, and `nest_asyncio` added to `requirements.txt`.

---

## Page 1 — Live Monitor

Three tabs. Data fetched fresh on every 60-second refresh via `st.cache_data(ttl=60)`.

### Tab 1: Price Chart

- Fetch last 500 BTC/USDT 1h candles via `fetch_ohlcv_bulk` (Binance public API, no key required)
- Compute Bollinger Bands via `compute_bollinger_bands(close)` and RSI via `compute_rsi(close)`
- Plot close price as a line using `st.plotly_chart` with Bollinger Band upper/lower overlays
- RSI subplot rendered below the main chart
- Chart title shows symbol, timeframe, and last updated timestamp

### Tab 2: Signals

Displays the full decision chain for the current candle. Signal computation uses `generate_signal(close, volume)` — not `rolling_signals` — which evaluates only the latest candle and avoids the O(n²) cost of iterating the full series.

| Section | Source | Fields shown |
|---|---|---|
| Statistical Swarm | `generate_signal(close, volume)` | Signal (bullish / bearish / neutral) |
| Macro Swarm | `asyncio.run(scrape_crypto_headlines())` → `classify_sentiment(headlines)` | Signal, confidence (%), reasoning |
| Consensus Decision | `evaluate_consensus(close, volume, macro, capital_usdt)` | Action (displayed as uppercase: BUY / SELL / HOLD), EV, Kelly fraction, position size (USDT), reasoning |

**Async bridge:** `scrape_crypto_headlines()` is an async Playwright function. It is called via `asyncio.run()` after `nest_asyncio.apply()` has been applied at module level (see Architecture section). Error handling covers Playwright not installed and network timeouts — both fall back to `[]` headlines and `classify_sentiment([])` returns a neutral `MacroSignal` with confidence 0.5 without making an Anthropic API call.

Macro sentiment is the only cost-incurring operation (~$0.00025 per Claude Haiku call). It is wrapped in `st.cache_data(ttl=60)` so it fires at most once per 60-second cycle.

Layout: three `st.metric` blocks at the top for the three signals, expandable `st.expander` sections below for reasoning text.

### Tab 3: Balance

- Calls `build_exchange()` then `fetch_balance(exchange)` using testnet credentials from `.env`
- `build_exchange()` requires `BINANCE_TESTNET_API_KEY` and `BINANCE_TESTNET_SECRET` — distinct from the public OHLCV path which needs no key
- Displays free USDT balance as a large `st.metric`
- Shows last updated timestamp
- If credentials are missing, shows `st.error` with setup instructions rather than crashing

---

## Page 2 — Strategy Evaluation

Three tabs. Reads local files only — no API calls on this page.

### Tab 1: Calibration

Reads the most recent row from `agents/memory/backtest_log.md` via `latest_backtest_stats()` and evaluates it against all three calibration thresholds from `backtest/calibration.py`.

| Metric | Threshold | Source constant |
|---|---|---|
| Sharpe Ratio | >= 1.0 | `MIN_SHARPE` |
| Profit Factor | >= 1.2 | `MIN_PROFIT_FACTOR` |
| Max Drawdown | display only | no gate |
| Brier Score | display only if present | `MAX_BRIER = 0.25` shown for reference |

**Note:** The backtest log does not record Brier Score (it requires LLM probability predictions which are not part of the CLI backtest runner). The Brier Score threshold is shown as a reference value only. The live-readiness verdict is based on Sharpe and Profit Factor from the log.

Verdict displayed prominently as a large header using `st.success("LIVE READY")` or `st.error("NOT LIVE READY")`. No emojis.

### Tab 2: Backtest History

- Parses `agents/memory/backtest_log.md` Markdown table into a pandas DataFrame via `parse_backtest_log()`
- Displays with `st.dataframe`, newest-first
- Columns: Date, Symbol, Timeframe, Candles, Period, Return, Sharpe, PF, Drawdown, Trades, Verdict
- If the file does not exist or has no data rows, shows `st.info("No backtest runs logged yet")`

### Tab 3: Shadow Pipeline

- Reads `agents/memory/PENDING_UPGRADES.md` as raw text
- Renders with `st.markdown`
- If the file is empty or has no entries beyond the header, shows: "No variants have beaten the baseline yet."

---

## Data Flow Summary

```
Page 1 (auto-refreshes every 60s, non-blocking)
  Binance public API  →  fetch_ohlcv_bulk()                 →  Price Chart tab
  compute_rsi / compute_bollinger_bands                      →  Price Chart tab
  generate_signal(close, volume)                             →  Signals tab (statistical)
  asyncio.run(scrape_crypto_headlines())                     →  Signals tab (macro input)
    → classify_sentiment(headlines) → Anthropic Claude Haiku →  Signals tab (macro output)
  evaluate_consensus(close, volume, macro, capital_usdt)     →  Signals tab (consensus)
  build_exchange() → fetch_balance(exchange) [testnet]       →  Balance tab

Page 2 (auto-refreshes every 60s, reads local files only — no API calls)
  agents/memory/backtest_log.md    →  Calibration tab + Backtest History tab
  agents/memory/PENDING_UPGRADES.md  →  Shadow Pipeline tab
```

---

## Shared Utilities (`dashboard/utils.py`)

| Function | Purpose |
|---|---|
| `load_ohlcv() -> tuple[pd.Series, pd.Series]` | Fetch 500 BTC/USDT 1h candles via `fetch_ohlcv_bulk`, return `(close, volume)` |
| `parse_backtest_log() -> pd.DataFrame` | Parse `backtest_log.md` Markdown table into DataFrame, newest-first |
| `latest_backtest_stats() -> dict` | Return the most recent row as a dict for calibration tab |
| `load_pending_upgrades() -> str` | Read `PENDING_UPGRADES.md` as a string; return empty-state message if absent |

All functions decorated with `@st.cache_data(ttl=60)`.

---

## Error Handling

| Failure | Behaviour |
|---|---|
| Binance OHLCV API fails | `st.warning` with error message; chart tab shows empty state |
| Playwright not installed or network timeout | Headlines fall back to `[]`; macro signal shows as neutral with note |
| Anthropic API fails | `st.warning("Macro sentiment unavailable")`; macro signal shows as neutral |
| `BINANCE_TESTNET_API_KEY` / `SECRET` missing | `st.error` with setup instructions on Balance tab; other tabs unaffected |
| `backtest_log.md` missing | `st.info("No backtest runs logged yet")` on Calibration and History tabs |
| `PENDING_UPGRADES.md` missing | Shadow Pipeline tab shows empty-state message |

---

## Running the Dashboard

```bash
# Install new dependencies
pip install streamlit streamlit-autorefresh nest_asyncio

# Run
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` in the browser. No cloud, no sharing, no public URL.

---

## Definition of Done

- [ ] `streamlit run dashboard/app.py` opens without error
- [ ] Page 1 Tab 1 renders a price chart with Bollinger Bands and RSI
- [ ] Page 1 Tab 2 shows live statistical signal, macro signal, and consensus decision
- [ ] Page 1 Tab 3 shows testnet USDT balance
- [ ] Page 2 Tab 1 shows calibration metrics and live-readiness verdict
- [ ] Page 2 Tab 2 renders backtest history table from `backtest_log.md`
- [ ] Page 2 Tab 3 renders contents of `PENDING_UPGRADES.md`
- [ ] Auto-refreshes every 60 seconds without blocking the UI
- [ ] Graceful error messages when any data source is unavailable
- [ ] No emojis in UI text
