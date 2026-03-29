# Cassandra

**Fully Automated Multi-Agentic AI Crypto Trading Bot**

Cassandra is a modular, multi-agent trading system that combines deterministic technical analysis with LLM-powered macro sentiment to generate, validate, and continuously improve trading decisions — all while enforcing strict human approval before any strategy change goes live.

---

## How It Works

Cassandra uses two independent signal sources ("swarms") that must agree before a trade is executed:

1. **Statistical Swarm** — RSI + Bollinger Bands + Volume Z-Score on live OHLCV data
2. **Macro Swarm** — Playwright scrapes crypto news headlines → Claude LLM classifies sentiment

A trade only fires when both swarms agree on direction **and** the expected value is strictly positive. Position sizing uses Fractional Kelly (25%) to keep risk conservative.

In the background, a **Shadow Pipeline** continuously backtests alternative parameter sets. If a variant beats the current strategy's Sharpe Ratio, it is written to `agents/memory/PENDING_UPGRADES.md` for human review — nothing is ever auto-merged.

---

## Architecture

```
Cassandra/
├── strategy/
│   ├── statistical_arb.py     # Phase 2 — RSI, Bollinger Bands, Volume Z-Score signals
│   ├── macro_sentiment.py     # Phase 3 — LLM sentiment classifier (Claude Haiku)
│   ├── consensus.py           # Phase 4 — dual-swarm gate + EV check → TradeDecision
│   └── optimizer.py           # Phase 7 — parameterised variant catalogue (8 configs)
│
├── risk_engine/
│   ├── kelly_size.py          # Expected value, Fractional Kelly, position sizing
│   ├── validate_risk.py       # EV gate, drawdown guard
│   └── metrics.py             # Brier score, profit factor
│
├── backtest/
│   ├── vectorbt_engine.py     # VectorBT wrapper — returns {return, sharpe, drawdown, trades}
│   ├── run_backtest.py        # Full-pipeline backtest runner
│   └── calibration.py        # CalibrationReport — live-readiness gate (Sharpe≥1.0, PF≥1.2)
│
├── execution/
│   ├── ccxt_client.py         # Binance testnet client + fetch_ohlcv_bulk (public prod data)
│   └── ccxt_stream.py         # WebSocket price stream
│
├── agents/
│   ├── orchestrator.py        # Shadow pipeline runner + PENDING_UPGRADES writer
│   └── tools/
│       └── news_scraper.py    # Playwright async headline scraper
│
├── scripts/
│   ├── run_backtest.py        # CLI — run a single backtest, logs to agents/memory/
│   └── run_shadow_pipeline.py # CLI — run all 8 variants, write winners to PENDING_UPGRADES
│
└── main.py                    # Startup healthcheck
```

---

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | Foundation & Infrastructure | ✅ Complete |
| 2 | Statistical Swarm | ✅ Complete |
| 3 | Macro Swarm (LLM Sentiment) | ✅ Complete |
| 4 | Consensus & Risk Engine | ✅ Complete |
| 5 | Simulation & Calibration | ✅ Complete |
| 6 | Reflex Dashboard | — Pending |
| 7 | Shadow Pipeline (Continuous Evolution) | ✅ Complete |

**Test coverage: 80/80 tests passing**

---

## Prerequisites

- Python 3.11+
- Binance testnet account — [testnet.binance.vision](https://testnet.binance.vision)
- Anthropic API key — [console.anthropic.com](https://console.anthropic.com)

---

## Setup

```bash
# 1. Clone
git clone https://github.com/your-username/Cassandra.git
cd Cassandra

# 2. Bootstrap (creates .venv, installs deps, writes cd() auto-activation hook)
bash setup.sh

# 3. Fill in credentials
cp .env.example .env
# Edit .env — add BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET, ANTHROPIC_API_KEY

# 4. Install Playwright browsers (needed for macro sentiment scraping)
.venv/bin/playwright install chromium
```

**VS Code / Cursor users:** the venv is auto-selected via `.vscode/settings.json`. No extra steps.

**CLI users:** after running `setup.sh`, open a new terminal (or `source ~/.zshrc`) — the `cd()` hook will auto-activate the venv whenever you enter the project directory.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|---|---|
| `BINANCE_TESTNET_API_KEY` | API key from testnet.binance.vision |
| `BINANCE_TESTNET_SECRET` | Secret from testnet.binance.vision |
| `ANTHROPIC_API_KEY` | Anthropic API key for macro sentiment (Claude Haiku) |

---

## Running

### Healthcheck

```bash
.venv/bin/python main.py
```

Verifies testnet connectivity, VectorBT import, and agent memory files.

### Run a backtest

```bash
# Default: BTC/USDT 1h, 500 candles
.venv/bin/python scripts/run_backtest.py

# Custom
.venv/bin/python scripts/run_backtest.py --symbol ETH/USDT --timeframe 4h --limit 1000 --capital 5000
```

Results are printed and appended to `agents/memory/backtest_log.md`.

### Run the shadow pipeline

Tests all 8 strategy parameter variants against historical data. Writes any that beat the baseline Sharpe to `agents/memory/PENDING_UPGRADES.md` for your review.

```bash
.venv/bin/python scripts/run_shadow_pipeline.py

# Custom
.venv/bin/python scripts/run_shadow_pipeline.py --symbol BTC/USDT --timeframe 1h --candles 15000
```

### Run tests

```bash
.venv/bin/pytest tests/ -v
```

---

## Live-Readiness Gate

Before the strategy is considered ready for live trading, `CalibrationReport` enforces:

| Metric | Threshold |
|---|---|
| Sharpe Ratio | ≥ 1.0 |
| Profit Factor | ≥ 1.2 |
| Brier Score (LLM calibration) | ≤ 0.25 |

`is_live_ready = True` only when all applicable thresholds pass.

---

## Trade Logic

```
Tick
 └─ Statistical Swarm (RSI + BB + Volume Z) → bullish / bearish / neutral
 └─ Macro Swarm (headlines → Claude Haiku)  → bullish / bearish / neutral
         │
         ▼
   Both agree & non-neutral?
         │ No → HOLD
         ▼
   EV = p × b − (1 − p) > 0?
         │ No → HOLD
         ▼
   Kelly fraction f* = (p×b − q) / b × 0.25
         │
         ▼
   TradeDecision(action, position_size_usdt, ev, kelly_fraction, reasoning)
```

Execution routes to **Binance testnet** — no real money is at risk during development.

---

## Shadow Pipeline

The shadow pipeline (`scripts/run_shadow_pipeline.py`) is the continuous improvement loop:

1. Fetches 15 000+ candles of historical data
2. Runs the current baseline strategy → records Sharpe
3. Backtests 8 alternative parameter variants (different RSI/BB/Volume periods)
4. Any variant with Sharpe > baseline **and** ≥ 3 trades is written to `agents/memory/PENDING_UPGRADES.md`
5. **Human reviews and manually applies changes** — nothing is auto-merged

---

## Tech Stack

| Library | Purpose |
|---|---|
| `ccxt 4.3.95` | Exchange connectivity (Binance testnet + public API) |
| `vectorbt 0.26.2` | Vectorised backtesting engine |
| `pandas 2.2.2` | Time-series data manipulation |
| `numpy 1.26.4` | Numerical computations |
| `anthropic 0.28.0` | Claude Haiku for macro sentiment classification |
| `playwright 1.44.0` | Headless browser for news scraping |
| `python-dotenv 1.0.1` | Environment variable management |
| `pytest 8.2.0` | Test suite (80 tests) |

> **Note:** `plotly` is pinned to `>=5.0,<6` — vectorbt 0.26.2 depends on `heatmapgl` which was removed in plotly 6.x.

---

## Project Conventions

- **TDD throughout** — failing tests written before every implementation
- **Testnet only** — all order execution targets Binance testnet; public Binance API used for historical OHLCV only (no key required)
- **No auto-merges** — the shadow pipeline proposes; humans decide
- **Pure functions** — all strategy and risk functions are stateless and side-effect free
- **Type hints** — full type annotations on all public functions
