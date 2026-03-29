"""Page 3 — Backtest Runner.

Interactive UI for running backtests with configurable parameters.
Results are displayed in-page and appended to agents/memory/backtest_log.md.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.calibration import MIN_PROFIT_FACTOR, MIN_SHARPE, build_report
from backtest.run_backtest import run_full_backtest
from backtest.vectorbt_engine import load_ohlcv_from_df
from execution.ccxt_client import fetch_ohlcv_bulk

st.set_page_config(page_title="Backtest Runner — Cassandra", layout="wide")


# ── Log helper (mirrors scripts/run_backtest.py) ──────────────────────────────
def _append_log(
    symbol: str,
    timeframe: str,
    candles: int,
    capital: float,
    report: object,
    verdict: str,
    date_from: str,
    date_to: str,
) -> None:
    log_path = Path(__file__).parent.parent.parent / "agents" / "memory" / "backtest_log.md"

    if not log_path.exists():
        log_path.write_text(
            "# Backtest Log\n\n"
            "| Date (UTC) | Symbol | TF | Candles | Period | Return | Sharpe | PF | DD | Trades | Verdict |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n"
        )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    pf = report.profit_factor  # type: ignore[attr-defined]
    pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
    line = (
        f"| {now} | {symbol} | {timeframe} | {candles} | {date_from}→{date_to} "
        f"| {report.total_return:+.2%} | {report.sharpe_ratio:.3f} "  # type: ignore[attr-defined]
        f"| {pf_str} | {report.max_drawdown:.2%} | {report.total_trades} | {verdict} |\n"  # type: ignore[attr-defined]
    )
    with log_path.open("a") as f:
        f.write(line)


st.title("Backtest Runner")
st.caption("Configure parameters and run a backtest. Results are logged automatically.")

# ── Configuration form ────────────────────────────────────────────────────────
with st.form("backtest_form"):
    col1, col2, col3, col4 = st.columns(4)

    symbol = col1.selectbox(
        "Symbol",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
    )
    timeframe = col2.selectbox(
        "Timeframe",
        ["1h", "4h", "1d"],
    )
    candles = col3.select_slider(
        "Candles",
        options=[500, 1000, 2000, 5000, 10000, 15000, 20000],
        value=5000,
    )
    capital = col4.number_input(
        "Starting Capital (USDT)",
        min_value=100.0,
        max_value=1_000_000.0,
        value=1_000.0,
        step=100.0,
    )

    submitted = st.form_submit_button("Run Backtest", use_container_width=True)

# ── Run backtest on submit ────────────────────────────────────────────────────
if submitted:
    with st.spinner(f"Fetching {candles:,} {symbol} {timeframe} candles and running backtest..."):
        try:
            df = fetch_ohlcv_bulk(symbol=symbol, timeframe=timeframe, total_candles=candles)
            close = load_ohlcv_from_df(df)
            ts_index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            volume = pd.Series(df["volume"].to_numpy(), index=ts_index)

            stats = run_full_backtest(close, volume, initial_cash=capital)

            gross_profit = max(stats["total_return"] * capital, 0.0)
            gross_loss = max(-stats["total_return"] * capital, 1.0)
            report = build_report(stats, gross_profit=gross_profit, gross_loss=gross_loss)

            date_from = close.index[0].strftime("%Y-%m-%d")
            date_to = close.index[-1].strftime("%Y-%m-%d")
            verdict = "LIVE READY" if report.is_live_ready else "NOT LIVE READY"

            # Store in session state so results persist across reruns
            st.session_state["backtest_result"] = {
                "report": report,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(close),
                "capital": capital,
                "date_from": date_from,
                "date_to": date_to,
                "verdict": verdict,
            }

            # Append to backtest log
            _append_log(
                symbol=symbol,
                timeframe=timeframe,
                candles=len(close),
                capital=capital,
                report=report,
                verdict=verdict,
                date_from=date_from,
                date_to=date_to,
            )

        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.session_state.pop("backtest_result", None)

# ── Display results ───────────────────────────────────────────────────────────
if "backtest_result" in st.session_state:
    r = st.session_state["backtest_result"]
    report = r["report"]

    st.divider()
    st.subheader(f"Results — {r['symbol']} {r['timeframe']}  |  {r['date_from']} to {r['date_to']}")

    if report.is_live_ready:
        st.success("LIVE READY — all thresholds met")
    else:
        st.error("NOT LIVE READY — thresholds not met")

    c1, c2, c3, c4, c5 = st.columns(5)

    sharpe_ok = report.sharpe_ratio >= MIN_SHARPE
    pf_ok = report.profit_factor == float("inf") or report.profit_factor >= MIN_PROFIT_FACTOR
    pf_display = "inf" if report.profit_factor == float("inf") else f"{report.profit_factor:.3f}"

    c1.metric(
        "Total Return",
        f"{report.total_return:+.2%}",
    )
    c2.metric(
        "Sharpe Ratio",
        f"{report.sharpe_ratio:.3f}",
        f"Need >= {MIN_SHARPE}",
        delta_color="normal" if sharpe_ok else "inverse",
    )
    c3.metric(
        "Profit Factor",
        pf_display,
        f"Need >= {MIN_PROFIT_FACTOR}",
        delta_color="normal" if pf_ok else "inverse",
    )
    c4.metric("Max Drawdown", f"{report.max_drawdown:.2%}")
    c5.metric("Total Trades", report.total_trades)

    st.caption(
        f"Capital: ${r['capital']:,.0f} USDT  |  "
        f"Candles fetched: {r['candles']:,}  |  "
        f"Result logged to agents/memory/backtest_log.md"
    )
