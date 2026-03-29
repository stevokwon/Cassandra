# Phase 6: Streamlit Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-page Streamlit dashboard with auto-refresh that shows live signals, price chart, balance, calibration status, backtest history, and shadow pipeline results.

**Architecture:** `dashboard/utils.py` holds all data-loading helpers (pure functions + one `@st.cache_data`-wrapped OHLCV loader). Two Streamlit pages (`1_Live_Monitor.py`, `2_Strategy_Evaluation.py`) each with three tabs. Entry point is `dashboard/app.py`. Auto-refreshes every 60 seconds via `streamlit-autorefresh` without blocking the UI. Playwright async is bridged with `nest_asyncio`.

**Tech Stack:** Python 3.11+, `streamlit`, `streamlit-autorefresh`, `nest_asyncio`, `plotly`, `pandas`, existing project modules.

---

## File Map

| Path | Action | Responsibility |
|------|--------|----------------|
| `dashboard/__init__.py` | Create | Package marker — empty |
| `dashboard/app.py` | Create | Entry point, page config, landing text |
| `dashboard/pages/__init__.py` | Create | Package marker — empty |
| `dashboard/pages/1_Live_Monitor.py` | Create | Price chart + signals + balance |
| `dashboard/pages/2_Strategy_Evaluation.py` | Create | Calibration + backtest history + shadow pipeline |
| `dashboard/utils.py` | Create | `load_ohlcv`, `parse_backtest_log`, `latest_backtest_stats`, `load_pending_upgrades` |
| `tests/test_dashboard_utils.py` | Create | TDD suite for utils (6 tests) |
| `requirements.txt` | Modify | Add `streamlit`, `streamlit-autorefresh`, `nest_asyncio` |

---

## Existing code to know

- `strategy/statistical_arb.py`:
  - `generate_signal(close, volume) -> Literal["bullish","bearish","neutral"]` — single-candle signal
  - `compute_rsi(close, period=14) -> pd.Series` — RSI values
  - `compute_bollinger_bands(close, period=20, std_dev=2.0) -> pd.DataFrame` — columns: `upper`, `middle`, `lower`
- `strategy/macro_sentiment.py`: `classify_sentiment(headlines: list[str]) -> MacroSignal`
- `strategy/consensus.py`: `evaluate_consensus(close, volume, macro, capital_usdt) -> TradeDecision`
  - `TradeDecision.action` is lowercase: `"buy"`, `"sell"`, `"hold"` — display as `.upper()`
- `agents/tools/news_scraper.py`: `scrape_crypto_headlines()` — **async**, must be called via `asyncio.run()` after `nest_asyncio.apply()`
- `execution/ccxt_client.py`:
  - `fetch_ohlcv_bulk(symbol, timeframe, total_candles) -> pd.DataFrame` — public Binance, no key needed
  - `build_exchange() -> ccxt.binance` — raises `ValueError` if testnet env vars missing
  - `fetch_balance(exchange) -> float`
- `backtest/calibration.py`: `MIN_SHARPE = 1.0`, `MIN_PROFIT_FACTOR = 1.2`, `MAX_BRIER = 0.25`
- `agents/memory/backtest_log.md` — Markdown table, columns: `Date (UTC)`, `Symbol`, `TF`, `Candles`, `Period`, `Return`, `Sharpe`, `PF`, `DD`, `Trades`, `Verdict`
- `agents/memory/PENDING_UPGRADES.md` — Markdown prose, appended by shadow pipeline

---

## Task 1: Setup

**Files:**
- Modify: `requirements.txt`
- Create: `dashboard/__init__.py`, `dashboard/pages/__init__.py`

- [ ] **Step 1: Create branch**

```bash
cd /Users/a91956/Documents/GitHub/Cassandra
git checkout -b phase-6/dashboard
```

- [ ] **Step 2: Add dependencies to requirements.txt**

Add these three lines after the existing entries:

```
streamlit>=1.35.0
streamlit-autorefresh>=1.0.1
nest_asyncio>=1.6.0
```

- [ ] **Step 3: Install new dependencies**

```bash
.venv/bin/pip install streamlit streamlit-autorefresh nest_asyncio --quiet
```

- [ ] **Step 4: Create dashboard package directories**

```bash
mkdir -p /Users/a91956/Documents/GitHub/Cassandra/dashboard/pages
touch /Users/a91956/Documents/GitHub/Cassandra/dashboard/__init__.py
touch /Users/a91956/Documents/GitHub/Cassandra/dashboard/pages/__init__.py
```

- [ ] **Step 5: Verify imports work**

```bash
.venv/bin/python -c "import streamlit; import streamlit_autorefresh; import nest_asyncio; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt dashboard/__init__.py dashboard/pages/__init__.py
git commit -m "feat: add dashboard package skeleton and new dependencies"
```

---

## Task 2: Shared Utilities (TDD)

**Files:**
- Create: `dashboard/utils.py`
- Create: `tests/test_dashboard_utils.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/a91956/Documents/GitHub/Cassandra/tests/test_dashboard_utils.py`:

```python
"""TDD tests for dashboard shared utilities."""
from pathlib import Path

import pandas as pd
import pytest


def test_parse_backtest_log_returns_dataframe(tmp_path, monkeypatch) -> None:
    """parse_backtest_log() returns a DataFrame from a valid log file."""
    import dashboard.utils as utils

    log = tmp_path / "backtest_log.md"
    log.write_text(
        "# Backtest Log\n\n"
        "| Date (UTC) | Symbol | TF | Candles | Period | Return | Sharpe | PF | DD | Trades | Verdict |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| 2026-03-01 10:00 | BTC/USDT | 1h | 500 | 2024-01-01->2024-03-01 | +5.00% | 1.200 | 1.30 | 3.00% | 45 | LIVE READY |\n"
    )
    monkeypatch.setattr(utils, "_BACKTEST_LOG", log)
    df = utils.parse_backtest_log()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_parse_backtest_log_returns_empty_when_file_missing(tmp_path, monkeypatch) -> None:
    """parse_backtest_log() returns empty DataFrame when file does not exist."""
    import dashboard.utils as utils

    monkeypatch.setattr(utils, "_BACKTEST_LOG", tmp_path / "nonexistent.md")
    df = utils.parse_backtest_log()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_parse_backtest_log_newest_first(tmp_path, monkeypatch) -> None:
    """parse_backtest_log() returns rows newest-first."""
    import dashboard.utils as utils

    log = tmp_path / "backtest_log.md"
    log.write_text(
        "# Backtest Log\n\n"
        "| Date (UTC) | Symbol | TF | Candles | Period | Return | Sharpe | PF | DD | Trades | Verdict |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| 2026-01-01 10:00 | BTC/USDT | 1h | 500 | 2024-01-01->2024-03-01 | +5.00% | 1.200 | 1.30 | 3.00% | 45 | LIVE READY |\n"
        "| 2026-02-01 10:00 | ETH/USDT | 4h | 500 | 2024-03-01->2024-06-01 | +3.00% | 0.900 | 1.10 | 5.00% | 30 | NOT LIVE READY |\n"
    )
    monkeypatch.setattr(utils, "_BACKTEST_LOG", log)
    df = utils.parse_backtest_log()
    assert "2026-02-01" in df.iloc[0]["Date (UTC)"]


def test_latest_backtest_stats_returns_dict(tmp_path, monkeypatch) -> None:
    """latest_backtest_stats() returns a non-empty dict from a populated log."""
    import dashboard.utils as utils

    log = tmp_path / "backtest_log.md"
    log.write_text(
        "# Backtest Log\n\n"
        "| Date (UTC) | Symbol | TF | Candles | Period | Return | Sharpe | PF | DD | Trades | Verdict |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| 2026-03-01 10:00 | BTC/USDT | 1h | 500 | 2024-01-01->2024-03-01 | +5.00% | 1.200 | 1.30 | 3.00% | 45 | LIVE READY |\n"
    )
    monkeypatch.setattr(utils, "_BACKTEST_LOG", log)
    result = utils.latest_backtest_stats()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_latest_backtest_stats_empty_when_no_log(tmp_path, monkeypatch) -> None:
    """latest_backtest_stats() returns empty dict when log file is absent."""
    import dashboard.utils as utils

    monkeypatch.setattr(utils, "_BACKTEST_LOG", tmp_path / "nonexistent.md")
    assert utils.latest_backtest_stats() == {}


def test_load_pending_upgrades_returns_message_when_file_missing(tmp_path, monkeypatch) -> None:
    """load_pending_upgrades() returns empty-state message when file is absent."""
    import dashboard.utils as utils

    monkeypatch.setattr(utils, "_PENDING_UPGRADES", tmp_path / "nonexistent.md")
    result = utils.load_pending_upgrades()
    assert "No variants" in result
```

- [ ] **Step 2: Run — confirm FAIL**

```bash
cd /Users/a91956/Documents/GitHub/Cassandra
.venv/bin/pytest tests/test_dashboard_utils.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'dashboard.utils'`

- [ ] **Step 3: Implement `dashboard/utils.py`**

Create `/Users/a91956/Documents/GitHub/Cassandra/dashboard/utils.py`:

```python
"""Shared data-loading utilities for the Cassandra dashboard.

`load_ohlcv` is wrapped with @st.cache_data(ttl=60) so results are cached
for 60 seconds and refreshed automatically on each Streamlit rerun.
File-reading helpers (parse_backtest_log etc.) are plain functions — fast
enough to not need caching.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow imports from repo root when running inside dashboard/
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.ccxt_client import fetch_ohlcv_bulk

_BACKTEST_LOG = Path(__file__).parent.parent / "agents" / "memory" / "backtest_log.md"
_PENDING_UPGRADES = Path(__file__).parent.parent / "agents" / "memory" / "PENDING_UPGRADES.md"

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
CANDLES = 500


@st.cache_data(ttl=60)
def load_ohlcv() -> tuple[pd.Series, pd.Series]:
    """Fetch 500 BTC/USDT 1h candles from the Binance public API.

    Cached for 60 seconds. No API key required.

    Returns:
        Tuple of (close, volume) as DatetimeIndex pd.Series.
    """
    df = fetch_ohlcv_bulk(symbol=SYMBOL, timeframe=TIMEFRAME, total_candles=CANDLES)
    ts_index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    close = pd.Series(df["close"].to_numpy(), index=ts_index, dtype=float)
    volume = pd.Series(df["volume"].to_numpy(), index=ts_index, dtype=float)
    return close, volume


def parse_backtest_log() -> pd.DataFrame:
    """Parse agents/memory/backtest_log.md into a DataFrame, newest-first.

    Returns:
        DataFrame with columns matching the log table header.
        Empty DataFrame if file does not exist or has no data rows.
    """
    if not _BACKTEST_LOG.exists():
        return pd.DataFrame()

    lines = [
        ln.strip()
        for ln in _BACKTEST_LOG.read_text().splitlines()
        if ln.strip().startswith("|")
    ]

    if len(lines) < 3:  # need header + separator + at least one data row
        return pd.DataFrame()

    def _parse_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    headers = _parse_row(lines[0])
    data_rows = [_parse_row(ln) for ln in lines[2:] if "---" not in ln]

    if not data_rows:
        return pd.DataFrame()

    df = pd.DataFrame(data_rows, columns=headers)
    return df.iloc[::-1].reset_index(drop=True)


def latest_backtest_stats() -> dict:
    """Return the most recent backtest log row as a dict.

    Returns:
        Dict keyed by column names, or empty dict if log is absent/empty.
    """
    df = parse_backtest_log()
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def load_pending_upgrades() -> str:
    """Read PENDING_UPGRADES.md as a string.

    Returns:
        File contents, or an empty-state message if the file is absent or
        contains no entries beyond the header.
    """
    _EMPTY = "No variants have beaten the baseline yet."

    if not _PENDING_UPGRADES.exists():
        return _EMPTY

    text = _PENDING_UPGRADES.read_text().strip()
    non_header = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not non_header:
        return _EMPTY

    return text
```

- [ ] **Step 4: Run — confirm PASS**

```bash
.venv/bin/pytest tests/test_dashboard_utils.py -v 2>&1
```

Expected: 6 tests PASSED.

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/pytest tests/ --tb=short 2>&1 | tail -5
```

Expected: **86 tests PASSED** (80 existing + 6 new).

- [ ] **Step 6: Commit**

```bash
git add dashboard/utils.py tests/test_dashboard_utils.py
git commit -m "feat: add dashboard utils with TDD suite"
```

---

## Task 3: Entry Point

**Files:**
- Create: `dashboard/app.py`

No tests — this is a one-screen landing page.

- [ ] **Step 1: Create `dashboard/app.py`**

Create `/Users/a91956/Documents/GitHub/Cassandra/dashboard/app.py`:

```python
"""Cassandra Dashboard — entry point.

Run with:
    streamlit run dashboard/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Cassandra",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Cassandra Trading System")
st.write(
    "Use the sidebar to navigate between pages.\n\n"
    "- **Live Monitor** — real-time price chart, signals, and account balance\n"
    "- **Strategy Evaluation** — calibration status, backtest history, shadow pipeline"
)
st.caption("All data refreshes automatically every 60 seconds.")
```

- [ ] **Step 2: Syntax check**

```bash
.venv/bin/python -m py_compile dashboard/app.py && echo "OK"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add dashboard entry point"
```

---

## Task 4: Page 1 — Live Monitor

**Files:**
- Create: `dashboard/pages/1_Live_Monitor.py`

- [ ] **Step 1: Create `dashboard/pages/1_Live_Monitor.py`**

Create `/Users/a91956/Documents/GitHub/Cassandra/dashboard/pages/1_Live_Monitor.py`:

```python
"""Page 1 — Live Monitor.

Tabs:
    Price Chart  — BTC/USDT 1h close price with Bollinger Bands + RSI subplot
    Signals      — statistical swarm, macro swarm, consensus decision
    Balance      — testnet USDT balance
"""
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import nest_asyncio
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# Patch the running asyncio loop so asyncio.run() works inside Streamlit
nest_asyncio.apply()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.tools.news_scraper import scrape_crypto_headlines
from backtest.calibration import MIN_PROFIT_FACTOR, MIN_SHARPE
from dashboard.utils import load_ohlcv
from execution.ccxt_client import build_exchange, fetch_balance
from strategy.consensus import evaluate_consensus
from strategy.macro_sentiment import MacroSignal, classify_sentiment
from strategy.statistical_arb import (
    compute_bollinger_bands,
    compute_rsi,
    generate_signal,
)

st.set_page_config(page_title="Live Monitor — Cassandra", layout="wide")
st_autorefresh(interval=60_000, key="live_refresh")

st.title("Live Monitor")
st.caption(
    f"Auto-refreshes every 60 seconds. "
    f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
)

tab_chart, tab_signals, tab_balance = st.tabs(["Price Chart", "Signals", "Balance"])

# ── Fetch OHLCV (shared across Chart and Signals tabs) ────────────────────────
try:
    close, volume = load_ohlcv()
    ohlcv_ok = True
except Exception as exc:
    st.warning(f"Could not fetch price data: {exc}")
    ohlcv_ok = False

# ── Tab 1: Price Chart ────────────────────────────────────────────────────────
with tab_chart:
    if not ohlcv_ok:
        st.info("Price data unavailable.")
    else:
        bb = compute_bollinger_bands(close)
        rsi = compute_rsi(close)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=("BTC/USDT 1h", "RSI (14)"),
        )

        fig.add_trace(
            go.Scatter(x=close.index, y=close, name="Close", line=dict(color="#1f77b4")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=bb.index, y=bb["upper"], name="BB Upper",
                       line=dict(color="#aec7e8", dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bb.index, y=bb["lower"], name="BB Lower",
                line=dict(color="#aec7e8", dash="dash"),
                fill="tonexty", fillcolor="rgba(174,199,232,0.1)",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="#ff7f0e")),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

        fig.update_layout(height=600, showlegend=True, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Signals ────────────────────────────────────────────────────────────
with tab_signals:
    if not ohlcv_ok:
        st.info("Signal data unavailable — price feed required.")
    else:
        # Statistical swarm — single candle, no O(n^2) rolling
        stat_signal = generate_signal(close, volume)

        # Macro swarm — async Playwright scraper + Claude Haiku
        try:
            headlines = asyncio.run(scrape_crypto_headlines())
            macro: MacroSignal = classify_sentiment(headlines)
        except Exception as exc:
            st.warning(f"Macro sentiment unavailable: {exc}")
            macro = MacroSignal(
                signal="neutral",
                confidence=0.5,
                reasoning="Scraper error — falling back to neutral.",
            )

        # Balance for consensus sizing (fallback to 1000 if testnet unreachable)
        try:
            exchange = build_exchange()
            capital = fetch_balance(exchange)
        except Exception:
            capital = 1_000.0

        decision = evaluate_consensus(close, volume, macro, capital_usdt=capital)

        col1, col2, col3 = st.columns(3)
        col1.metric("Statistical Swarm", stat_signal.upper())
        col2.metric(
            "Macro Swarm",
            macro.signal.upper(),
            f"{macro.confidence:.0%} confidence",
        )
        col3.metric(
            "Consensus",
            decision.action.upper(),
            f"EV {decision.expected_value:+.4f}",
        )

        with st.expander("Macro reasoning"):
            st.write(macro.reasoning)

        with st.expander("Consensus reasoning"):
            st.write(decision.reasoning)
            c1, c2 = st.columns(2)
            c1.metric("Kelly Fraction", f"{decision.kelly_fraction:.4f}")
            c2.metric("Position Size", f"${decision.position_size_usdt:,.2f} USDT")

# ── Tab 3: Balance ────────────────────────────────────────────────────────────
with tab_balance:
    try:
        exchange = build_exchange()
        balance = fetch_balance(exchange)
        st.metric(
            "Free USDT Balance (Testnet)",
            f"${balance:,.2f}",
        )
        st.caption(
            f"Source: Binance testnet — "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
    except ValueError as exc:
        st.error(
            f"Cannot connect to testnet: {exc}\n\n"
            "Check that BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET "
            "are set in your .env file."
        )
    except Exception as exc:
        st.warning(f"Balance fetch failed: {exc}")
```

- [ ] **Step 2: Syntax check**

```bash
.venv/bin/python -m py_compile dashboard/pages/1_Live_Monitor.py && echo "OK"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add dashboard/pages/1_Live_Monitor.py
git commit -m "feat: add Live Monitor page with price chart, signals, balance tabs"
```

---

## Task 5: Page 2 — Strategy Evaluation

**Files:**
- Create: `dashboard/pages/2_Strategy_Evaluation.py`

- [ ] **Step 1: Create `dashboard/pages/2_Strategy_Evaluation.py`**

Create `/Users/a91956/Documents/GitHub/Cassandra/dashboard/pages/2_Strategy_Evaluation.py`:

```python
"""Page 2 — Strategy Evaluation.

Tabs:
    Calibration      — Sharpe / PF thresholds and live-readiness verdict
    Backtest History — full sortable table from backtest_log.md
    Shadow Pipeline  — PENDING_UPGRADES.md contents
"""
import sys
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.calibration import MAX_BRIER, MIN_PROFIT_FACTOR, MIN_SHARPE
from dashboard.utils import latest_backtest_stats, load_pending_upgrades, parse_backtest_log

st.set_page_config(page_title="Strategy Evaluation — Cassandra", layout="wide")
st_autorefresh(interval=60_000, key="eval_refresh")

st.title("Strategy Evaluation")

tab_cal, tab_hist, tab_shadow = st.tabs(
    ["Calibration", "Backtest History", "Shadow Pipeline"]
)

# ── Tab 1: Calibration ────────────────────────────────────────────────────────
with tab_cal:
    stats = latest_backtest_stats()

    if not stats:
        st.info(
            "No backtest runs logged yet. "
            "Run `python scripts/run_backtest.py` to populate."
        )
    else:
        def _parse_float(val: object) -> float:
            """Strip formatting characters and convert to float."""
            return float(str(val).replace("%", "").replace("+", "").strip())

        try:
            sharpe = _parse_float(stats.get("Sharpe", "nan"))
            pf_raw = str(stats.get("PF", "0")).strip()
            pf = float("inf") if pf_raw in ("inf", "∞") else _parse_float(pf_raw)
            dd = _parse_float(stats.get("DD", "nan"))
            total_return = _parse_float(stats.get("Return", "nan"))
            trades = int(stats.get("Trades", 0))
        except (ValueError, TypeError) as exc:
            st.warning(f"Could not parse calibration values: {exc}")
            st.stop()

        sharpe_ok = sharpe >= MIN_SHARPE
        pf_ok = pf == float("inf") or pf >= MIN_PROFIT_FACTOR
        live_ready = sharpe_ok and pf_ok

        if live_ready:
            st.success("LIVE READY — all thresholds met")
        else:
            st.error("NOT LIVE READY — thresholds not met")

        st.divider()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Sharpe Ratio",
            f"{sharpe:.3f}",
            f"Need >= {MIN_SHARPE}",
            delta_color="normal" if sharpe_ok else "inverse",
        )
        col2.metric(
            "Profit Factor",
            "inf" if pf == float("inf") else f"{pf:.3f}",
            f"Need >= {MIN_PROFIT_FACTOR}",
            delta_color="normal" if pf_ok else "inverse",
        )
        col3.metric("Max Drawdown", f"{dd:.2f}%", "Display only")
        col4.metric("Total Return", f"{total_return:+.2f}%")

        st.caption(
            f"Brier Score threshold reference: <= {MAX_BRIER} "
            f"(not recorded in backtest log — requires LLM probability predictions)"
        )
        st.caption(
            f"Latest run: {stats.get('Date (UTC)', 'unknown')}  |  "
            f"{stats.get('Symbol', '')} {stats.get('TF', '')}  |  "
            f"{trades} trades"
        )

# ── Tab 2: Backtest History ───────────────────────────────────────────────────
with tab_hist:
    df = parse_backtest_log()
    if df.empty:
        st.info(
            "No backtest runs logged yet. "
            "Run `python scripts/run_backtest.py` to populate."
        )
    else:
        st.dataframe(df, use_container_width=True)
        st.caption(f"{len(df)} runs logged — newest first")

# ── Tab 3: Shadow Pipeline ────────────────────────────────────────────────────
with tab_shadow:
    content = load_pending_upgrades()
    st.markdown(content)
    st.caption(
        "Source: agents/memory/PENDING_UPGRADES.md — "
        "updated by `python scripts/run_shadow_pipeline.py`"
    )
```

- [ ] **Step 2: Syntax check**

```bash
.venv/bin/python -m py_compile dashboard/pages/2_Strategy_Evaluation.py && echo "OK"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add dashboard/pages/2_Strategy_Evaluation.py
git commit -m "feat: add Strategy Evaluation page with calibration, history, shadow pipeline tabs"
```

---

## Task 6: Final Verification + Merge

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/a91956/Documents/GitHub/Cassandra
.venv/bin/pytest tests/ -v 2>&1 | tail -20
```

Expected: **86 tests PASSED, 0 failed.**

- [ ] **Step 2: Verify all imports**

```bash
.venv/bin/python -c "
from dashboard.utils import load_ohlcv, parse_backtest_log, latest_backtest_stats, load_pending_upgrades
import ast, pathlib
for p in ['dashboard/app.py', 'dashboard/pages/1_Live_Monitor.py', 'dashboard/pages/2_Strategy_Evaluation.py']:
    ast.parse(pathlib.Path(p).read_text())
print('All dashboard files syntax OK')
"
```

Expected: `All dashboard files syntax OK`

- [ ] **Step 3: Merge to main**

```bash
git checkout main
git merge --no-ff phase-6/dashboard -m "feat: Phase 6 complete — Streamlit dashboard (86/86 tests)"
```

- [ ] **Step 4: Verify on main**

```bash
.venv/bin/pytest tests/ --tb=short 2>&1 | tail -5
git log --oneline -6
```

- [ ] **Step 5: Print run instructions**

```bash
echo "Run the dashboard with:"
echo "  streamlit run dashboard/app.py"
echo "Opens at http://localhost:8501"
```

---

## Definition of Done

- [ ] `streamlit run dashboard/app.py` opens without error at `http://localhost:8501`
- [ ] Page 1 Tab 1 renders BTC/USDT price chart with Bollinger Bands and RSI
- [ ] Page 1 Tab 2 shows statistical signal, macro signal, and consensus decision
- [ ] Page 1 Tab 3 shows testnet USDT balance
- [ ] Page 2 Tab 1 shows calibration metrics with LIVE READY / NOT LIVE READY verdict
- [ ] Page 2 Tab 2 renders backtest history table from `backtest_log.md`
- [ ] Page 2 Tab 3 renders `PENDING_UPGRADES.md` contents
- [ ] Auto-refreshes every 60 seconds without blocking UI
- [ ] Graceful error messages when any data source is unavailable
- [ ] No emojis in UI text
- [ ] `pytest tests/ -v` shows **86/86 PASSED**
- [ ] All work on `phase-6/dashboard`, merged to `main` via `--no-ff`
