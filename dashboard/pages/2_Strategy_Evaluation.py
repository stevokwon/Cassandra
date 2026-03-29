"""Page 2 — Strategy Evaluation.

Tabs:
    Calibration      — Sharpe / PF thresholds and live-readiness verdict
    Backtest History — full sortable table from backtest_log.md
    Shadow Pipeline  — PENDING_UPGRADES.md contents
"""
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.calibration import MAX_BRIER, MIN_PROFIT_FACTOR, MIN_SHARPE
from dashboard.utils import latest_backtest_stats, load_pending_upgrades, parse_backtest_log

st.set_page_config(page_title="Strategy Evaluation — Cassandra", layout="wide")

# Auto-refresh every 60 seconds via JS — no external package required
components.html(
    '<script>setTimeout(function(){window.location.reload();}, 60000);</script>',
    height=0,
)

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
