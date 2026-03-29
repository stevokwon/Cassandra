#!/usr/bin/env python3
"""Cassandra — Phase 7 Shadow Pipeline

Fetches historical OHLCV data, runs the current baseline strategy, then
tests all CANDIDATE_VARIANTS. Variants that beat the baseline Sharpe are
written to agents/memory/PENDING_UPGRADES.md for human review.

Usage:
    python scripts/run_shadow_pipeline.py
    python scripts/run_shadow_pipeline.py --symbol ETH/USDT --timeframe 1h --candles 15000
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from agents.orchestrator import run_shadow_pipeline, write_pending_upgrades
from backtest.run_backtest import run_full_backtest
from backtest.vectorbt_engine import load_ohlcv_from_df
from execution.ccxt_client import fetch_ohlcv_bulk


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cassandra Phase 7 Shadow Pipeline")
    parser.add_argument("--symbol",    default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--candles",   default=15000, type=int)
    parser.add_argument("--capital",   default=1000.0, type=float)
    args = parser.parse_args()

    print(f"\n{'═' * 55}")
    print(f"  Cassandra Shadow Pipeline — {args.symbol} {args.timeframe}")
    print(f"  Candles: {args.candles:,}  |  Capital: ${args.capital:,.0f}")
    print(f"{'═' * 55}\n")

    # ── 1. Fetch data ────────────────────────────────────────────────────────
    print("Fetching OHLCV data...")
    df = fetch_ohlcv_bulk(symbol=args.symbol, timeframe=args.timeframe, total_candles=args.candles)
    close = load_ohlcv_from_df(df)
    ts_index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    volume = pd.Series(df["volume"].to_numpy(), index=ts_index)
    print(f"  {len(close)} candles ({close.index[0].date()} → {close.index[-1].date()})\n")

    # ── 2. Baseline ──────────────────────────────────────────────────────────
    print("Running baseline strategy (default params)...")
    baseline_stats = run_full_backtest(close, volume, initial_cash=args.capital)
    baseline_sharpe = baseline_stats.get("sharpe_ratio", math.nan)
    print(f"  Baseline Sharpe: {baseline_sharpe:.3f}  |  Trades: {baseline_stats['total_trades']}\n")

    # ── 3. Shadow pipeline ───────────────────────────────────────────────────
    print(f"Testing {8} strategy variants...")
    winners = run_shadow_pipeline(close, volume, baseline_sharpe=baseline_sharpe)

    # ── 4. Results ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    if not winners:
        print("  No variants beat the baseline — strategy is already optimal.")
        print(f"{'─' * 55}\n")
        return

    print(f"  {len(winners)} variant(s) beat baseline Sharpe of {baseline_sharpe:.3f}:\n")
    for r in winners:
        print(f"  ✓  {r['variant'].describe()}")
        print(f"     Sharpe={r['sharpe_ratio']:.3f}  Return={r['total_return']:+.2%}"
              f"  Trades={r['total_trades']}")

    # ── 5. Write to PENDING_UPGRADES.md ─────────────────────────────────────
    write_pending_upgrades(winners)
    print(f"\n  Winners written → agents/memory/PENDING_UPGRADES.md")
    print(f"  Review and approve manually before changing live strategy.")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    main()
