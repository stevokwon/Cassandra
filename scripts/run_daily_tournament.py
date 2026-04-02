#!/usr/bin/env python3
"""Daily strategy tournament.

Runs all CANDIDATE_VARIANTS against historical OHLCV data, logs a full
ranked results table to agents/memory/PENDING_UPGRADES.md, and prints
the winner of the day.

Usage:
    python scripts/run_daily_tournament.py
    python scripts/run_daily_tournament.py --symbol ETH/USDT --timeframe 4h --candles 10000
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.vectorbt_engine import load_ohlcv_from_df
from execution.ccxt_client import fetch_ohlcv_bulk
from strategy.optimizer import CANDIDATE_VARIANTS, StrategyVariant, backtest_variant

_LOG_PATH = Path(__file__).parent.parent / "agents" / "memory" / "PENDING_UPGRADES.md"


def _run_all(
    close: pd.Series,
    volume: pd.Series,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(CANDIDATE_VARIANTS)
    for i, variant in enumerate(CANDIDATE_VARIANTS, 1):
        result = backtest_variant(variant, close, volume)
        sharpe = result.get("sharpe_ratio", math.nan)
        trades = result.get("total_trades", 0)
        tag = "✓" if not math.isnan(sharpe) and trades >= 3 else "✗"
        print(
            f"  [{i:2d}/{total}] {tag}  {variant.describe()}\n"
            f"         Sharpe={sharpe:+.3f}  "
            f"Return={result['total_return']:+.2%}  "
            f"Trades={trades}"
        )
        results.append(result)
    return results


def _rank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [
        r for r in results
        if not math.isnan(r.get("sharpe_ratio", math.nan))
        and r.get("total_trades", 0) >= 3
    ]
    return sorted(valid, key=lambda r: r["sharpe_ratio"], reverse=True)


def _write_log(
    all_results: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    symbol: str,
    timeframe: str,
    candle_count: int,
    date_from: str,
    date_to: str,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    winner: dict[str, Any] | None = ranked[0] if ranked else None

    lines: list[str] = [
        f"\n---\n\n",
        f"## Daily Tournament — {now}\n\n",
        f"**Dataset:** {symbol} {timeframe} | {candle_count:,} candles | "
        f"{date_from} → {date_to}  \n",
        f"**Variants tested:** {len(all_results)}\n\n",
        "### Full Rankings\n\n",
        "| Rank | Variant | Sharpe | Return | Max DD | Trades |\n",
        "|------|---------|--------|--------|--------|--------|\n",
    ]

    rank_labels = {1: "🥇 1", 2: "🥈 2", 3: "🥉 3"}
    ranked_ids = {id(r): rank_labels.get(i + 1, str(i + 1)) for i, r in enumerate(ranked)}

    for r in ranked:
        v: StrategyVariant = r["variant"]
        rank_str = ranked_ids[id(r)]
        lines.append(
            f"| {rank_str} | {v.describe()} "
            f"| {r['sharpe_ratio']:+.3f} "
            f"| {r['total_return']:+.2%} "
            f"| {r['max_drawdown']:.2%} "
            f"| {r['total_trades']} |\n"
        )

    skipped = [r for r in all_results if r not in ranked]
    for r in skipped:
        v = r["variant"]
        sh = r.get("sharpe_ratio", math.nan)
        sh_str = f"{sh:+.3f}" if not math.isnan(sh) else "NaN"
        lines.append(
            f"| — | {v.describe()} | {sh_str} | {r['total_return']:+.2%} "
            f"| — | {r.get('total_trades', 0)} |\n"
        )

    if winner:
        v = winner["variant"]
        lines += [
            "\n### 🏆 Winner of the Day\n\n",
            f"**{v.describe()}**\n\n",
            f"| Metric | Value |\n",
            f"|--------|-------|\n",
            f"| Sharpe Ratio | {winner['sharpe_ratio']:+.3f} |\n",
            f"| Total Return | {winner['total_return']:+.2%} |\n",
            f"| Max Drawdown | {winner['max_drawdown']:.2%} |\n",
            f"| Trades | {winner['total_trades']} |\n",
            f"\n**Params to apply:**\n\n",
            f"```\n",
            f"rsi_period    = {v.rsi_period}\n",
            f"bb_period     = {v.bb_period}\n",
            f"bb_std        = {v.bb_std}\n",
            f"volume_period = {v.volume_period}\n",
            f"sma_period    = {v.sma_period}\n",
            f"```\n\n",
            f"> Review in `agents/memory/PENDING_UPGRADES.md` and apply manually "
            f"to `strategy/statistical_arb.py` defaults when satisfied.\n",
        ]
    else:
        lines.append("\n> No valid variants this run (all had NaN Sharpe or < 3 trades).\n")

    with _LOG_PATH.open("a") as f:
        f.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily strategy tournament")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--candles", type=int, default=15_000)
    args = parser.parse_args()

    print("═" * 55)
    print(f"  Cassandra Daily Tournament — {args.symbol} {args.timeframe}")
    print(f"  Variants: {len(CANDIDATE_VARIANTS)}  |  Candles: {args.candles:,}")
    print("═" * 55)

    print("\nFetching OHLCV data...")
    df = fetch_ohlcv_bulk(
        symbol=args.symbol,
        timeframe=args.timeframe,
        total_candles=args.candles,
    )
    close = load_ohlcv_from_df(df)
    ts_index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    volume = pd.Series(df["volume"].to_numpy(), index=ts_index)
    date_from = close.index[0].strftime("%Y-%m-%d")
    date_to = close.index[-1].strftime("%Y-%m-%d")
    print(f"  {len(close):,} candles  ({date_from} → {date_to})\n")

    print(f"Running {len(CANDIDATE_VARIANTS)} variants...\n")
    all_results = _run_all(close, volume)

    ranked = _rank(all_results)

    print("\n" + "─" * 55)
    if ranked:
        w = ranked[0]
        v: StrategyVariant = w["variant"]
        print(f"\n  🏆  Winner of the Day")
        print(f"      {v.describe()}")
        print(f"      Sharpe={w['sharpe_ratio']:+.3f}  "
              f"Return={w['total_return']:+.2%}  "
              f"Trades={w['total_trades']}")
    else:
        print("  No valid winner — all variants NaN or < 3 trades.")

    _write_log(all_results, ranked, args.symbol, args.timeframe,
               len(close), date_from, date_to)
    print(f"\n  Results logged → agents/memory/PENDING_UPGRADES.md")
    print("─" * 55)


if __name__ == "__main__":
    main()
