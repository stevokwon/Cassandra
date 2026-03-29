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

**Auto-refresh:** 60-second interval using `time.sleep(60)` + `st.rerun()` at the bottom of each page. Appropriate for a 1h-candle strategy — the signal cannot change faster than once per candle.

**New dependency:** `streamlit` added to `requirements.txt`.

---

## Page 1 — Live Monitor

Three tabs. Data fetched fresh on every 60-second refresh.

### Tab 1: Price Chart

- Fetch last 500 BTC/USDT 1h candles via `fetch_ohlcv_bulk` (Binance public API, no key)
- Plot close price as a line using `st.plotly_chart`
- Overlay Bollinger Band upper and lower bands (from `compute_bollinger_bands`)
- RSI subplot below the main chart (from `compute_rsi`)
- Chart title shows symbol, timeframe, and last updated timestamp

### Tab 2: Signals

Displays the full decision chain for the current candle:

| Section | Source | Fields shown |
|---|---|---|
| Statistical Swarm | `rolling_signals` on last 500 candles | Signal (bullish / bearish / neutral) |
| Macro Swarm | `classify_sentiment` via Claude Haiku | Signal, confidence (%), reasoning |
| Consensus Decision | `evaluate_consensus` | Action (BUY / SELL / HOLD), EV, Kelly fraction, position size (USDT), reasoning |

Macro sentiment call hits the Anthropic API on each refresh — the only cost-incurring operation. Claude Haiku pricing is negligible (~$0.00025 per call), but the call is made only once per 60-second cycle.

Layout: three `st.metric` blocks at the top for the three signals, expandable detail section below for reasoning text.

### Tab 3: Balance

- Calls `fetch_balance` via testnet `build_exchange`
- Displays free USDT balance as a large metric
- Shows last updated timestamp
- Simple — no chart needed

---

## Page 2 — Strategy Evaluation

Three tabs. Data read from local files and the last backtest log entry. No API calls on this page.

### Tab 1: Calibration

Reads the most recent row from `agents/memory/backtest_log.md` and reconstructs calibration metrics.

| Metric | Threshold | Pass condition |
|---|---|---|
| Sharpe Ratio | >= 1.0 | `sharpe >= MIN_SHARPE` |
| Profit Factor | >= 1.2 | `pf >= MIN_PROFIT_FACTOR` |
| Max Drawdown | display only | no gate |

Verdict displayed prominently: `LIVE READY` or `NOT LIVE READY` as a large header. No status emojis — use colour via `st.success` / `st.error`.

### Tab 2: Backtest History

- Parses `agents/memory/backtest_log.md` Markdown table into a pandas DataFrame
- Displays with `st.dataframe`, newest-first
- Columns: Date, Symbol, Timeframe, Candles, Period, Return, Sharpe, PF, Drawdown, Trades, Verdict
- If the file does not exist or is empty, shows an informational message

### Tab 3: Shadow Pipeline

- Reads `agents/memory/PENDING_UPGRADES.md` as raw text
- Renders with `st.markdown`
- If the file is empty or has no entries beyond the header, shows a message: "No variants have beaten the baseline yet."

---

## Data Flow Summary

```
Page 1 (every 60s)
  Binance public API  →  fetch_ohlcv_bulk()     →  Price Chart tab
  compute_rsi / compute_bollinger_bands          →  Price Chart tab
  rolling_signals()                              →  Signals tab (statistical)
  classify_sentiment() → Anthropic Claude Haiku →  Signals tab (macro)
  evaluate_consensus()                           →  Signals tab (consensus)
  fetch_balance() → Binance testnet             →  Balance tab

Page 2 (every 60s, but reads local files only — no API calls)
  agents/memory/backtest_log.md   →  Calibration tab + Backtest History tab
  agents/memory/PENDING_UPGRADES.md  →  Shadow Pipeline tab
```

---

## Shared Utilities (`dashboard/utils.py`)

| Function | Purpose |
|---|---|
| `load_ohlcv() -> tuple[pd.Series, pd.Series]` | Fetch 500 BTC/USDT 1h candles, return (close, volume) |
| `parse_backtest_log() -> pd.DataFrame` | Parse backtest_log.md Markdown table into DataFrame |
| `latest_backtest_stats() -> dict` | Return the most recent row as a dict for calibration |

---

## Error Handling

- If Binance API call fails: show `st.warning` with the error, display last known data if cached, do not crash
- If Anthropic API call fails: show `st.warning("Macro sentiment unavailable")`, mark macro signal as neutral
- If `backtest_log.md` does not exist: show `st.info("No backtest runs logged yet")`
- If `.env` missing credentials: show `st.error` with setup instructions

---

## Running the Dashboard

```bash
# Install new dependency
pip install streamlit

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
- [ ] Auto-refreshes every 60 seconds
- [ ] Graceful error messages when any data source is unavailable
- [ ] No emojis in UI text
