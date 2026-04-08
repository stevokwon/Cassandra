"""Page 1 — Live Monitor.

Tabs:
    Price Chart  — BTC/USDT 1h close price with Bollinger Bands + RSI subplot
    Signals      — statistical swarm, macro swarm, consensus decision
    Balance      — testnet USDT balance
"""
import asyncio
import concurrent.futures
import sys
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.tools.news_scraper import scrape_crypto_headlines
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

# Auto-refresh every 60 seconds via JS — no external package required
components.html(
    '<script>setTimeout(function(){window.location.reload();}, 60000);</script>',
    height=0,
)

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
            vertical_spacing=0.04,
            subplot_titles=("BTC/USDT 1h", "RSI (28)"),
        )

        # Price + Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=close.index, y=close, name="Close",
                line=dict(color="#F0F0F0", width=1.5),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bb.index, y=bb["upper"], name="BB Upper",
                line=dict(color="#FF6600", dash="dash", width=1),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bb.index, y=bb["lower"], name="BB Lower",
                line=dict(color="#FF6600", dash="dash", width=1),
                fill="tonexty", fillcolor="rgba(255,102,0,0.05)",
            ),
            row=1, col=1,
        )

        # RSI
        fig.add_trace(
            go.Scatter(
                x=rsi.index, y=rsi, name="RSI",
                line=dict(color="#FF6600", width=1.5),
            ),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#CC2200", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00AA44", line_width=1, row=2, col=1)

        fig.update_layout(
            height=620,
            showlegend=True,
            margin=dict(t=40, b=20, l=10, r=10),
            paper_bgcolor="#0D0D0D",
            plot_bgcolor="#111111",
            font=dict(color="#D4D4D4", family="Courier New, monospace", size=12),
            legend=dict(
                bgcolor="#1A1A1A", bordercolor="#333333", borderwidth=1,
                font=dict(size=11),
            ),
        )
        fig.update_xaxes(gridcolor="#222222", showgrid=True, zeroline=False, linecolor="#333333")
        fig.update_yaxes(gridcolor="#222222", showgrid=True, zeroline=False, linecolor="#333333")

        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Signals ────────────────────────────────────────────────────────────
with tab_signals:
    if not ohlcv_ok:
        st.info("Signal data unavailable — price feed required.")
    else:
        # Statistical swarm — single candle, no O(n^2) rolling
        stat_signal = generate_signal(close, volume)

        # Macro swarm — run async Playwright scraper in a new thread so it
        # gets its own event loop (Streamlit already runs one internally)
        def _fetch_headlines() -> list[str]:
            return asyncio.run(scrape_crypto_headlines())

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                headlines = pool.submit(_fetch_headlines).result(timeout=30)
            macro: MacroSignal = classify_sentiment(headlines)
        except Exception as exc:
            st.warning(f"Macro sentiment unavailable: {exc}")
            macro = MacroSignal(
                signal="neutral",
                confidence=0.5,
                reasoning="Scraper error — falling back to neutral.",
            )

        # Balance for consensus sizing (fallback if testnet unreachable)
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
