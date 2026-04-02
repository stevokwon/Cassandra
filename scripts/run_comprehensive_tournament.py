#!/usr/bin/env python3
"""Comprehensive daily strategy tournament.

Tests all CANDIDATE_VARIANTS across multiple symbols and timeframes,
scores each variant by its AVERAGE Sharpe across all valid datasets,
and logs a full ranked report to agents/memory/PENDING_UPGRADES.md.

Running on 3 symbols × 2 timeframes means a genuine winner beats the
market consistently — not just one lucky dataset.

Usage:
    python scripts/run_comprehensive_tournament.py
    python scripts/run_comprehensive_tournament.py --candles 5000
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

# Datasets to test every variant on.  Each entry is (symbol, timeframe).
# 1h: fine-grained noise test.  4h: smoother, trend-following friendly.
DATASETS = [
    ("BTC/USDT", "1h"),
    ("BTC/USDT", "4h"),
    ("ETH/USDT", "1h"),
    ("ETH/USDT", "4h"),
    ("SOL/USDT", "1h"),
    ("SOL/USDT", "4h"),
]

MIN_TRADES = 3  # minimum trades per dataset for a result to count


def _fetch(symbol: str, timeframe: str, candles: int) -> tuple[pd.Series, pd.Series, str, str]:
    df = fetch_ohlcv_bulk(symbol=symbol, timeframe=timeframe, total_candles=candles)
    close = load_ohlcv_from_df(df)
    ts_index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    volume = pd.Series(df["volume"].to_numpy(), index=ts_index)
    date_from = close.index[0].strftime("%Y-%m-%d")
    date_to = close.index[-1].strftime("%Y-%m-%d")
    return close, volume, date_from, date_to


def _score(results_by_dataset: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Aggregate per-dataset results into an overall ranking.

    A variant's aggregate Sharpe is the mean over datasets where it had
    >= MIN_TRADES.  Variants that fail on ALL datasets are excluded.
    """
    # Build per-variant accumulator indexed by variant description
    acc: dict[str, dict[str, Any]] = {}

    for dataset_key, results in results_by_dataset.items():
        for r in results:
            v: StrategyVariant = r["variant"]
            key = v.describe()
            if key not in acc:
                acc[key] = {
                    "variant": v,
                    "sharpes": [],
                    "returns": [],
                    "trades_total": 0,
                    "datasets_valid": 0,
                    "datasets_total": 0,
                }
            acc[key]["datasets_total"] += 1

            sharpe = r.get("sharpe_ratio", math.nan)
            trades = r.get("total_trades", 0)
            if not math.isnan(sharpe) and trades >= MIN_TRADES:
                acc[key]["sharpes"].append(sharpe)
                acc[key]["returns"].append(r.get("total_return", 0.0))
                acc[key]["trades_total"] += trades
                acc[key]["datasets_valid"] += 1

    scored = []
    for key, data in acc.items():
        if not data["sharpes"]:
            data["avg_sharpe"] = float("nan")
            data["avg_return"] = float("nan")
        else:
            data["avg_sharpe"] = sum(data["sharpes"]) / len(data["sharpes"])
            data["avg_return"] = sum(data["returns"]) / len(data["returns"])
        scored.append(data)

    valid = [s for s in scored if not math.isnan(s["avg_sharpe"])]
    return sorted(valid, key=lambda x: x["avg_sharpe"], reverse=True)


def _write_log(
    results_by_dataset: dict[str, list[dict[str, Any]]],
    dataset_meta: dict[str, tuple[str, str, int, str, str]],
    overall_ranking: list[dict[str, Any]],
    candles_per_dataset: int,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    winner = overall_ranking[0] if overall_ranking else None

    lines: list[str] = [
        f"\n---\n\n",
        f"## Comprehensive Tournament — {now}\n\n",
        f"**Variants tested:** {len(CANDIDATE_VARIANTS)}  "
        f"| **Datasets:** {len(DATASETS)}  "
        f"| **Candles/dataset:** {candles_per_dataset:,}\n\n",
    ]

    # Per-dataset breakdown
    lines.append("### Per-Dataset Results\n\n")
    for ds_key, results in results_by_dataset.items():
        symbol, tf, n_candles, date_from, date_to = dataset_meta[ds_key]
        lines.append(f"#### {symbol} {tf} ({date_from} → {date_to})\n\n")
        lines.append("| Variant | Sharpe | Return | Trades |\n")
        lines.append("|---------|--------|--------|--------|\n")
        sorted_results = sorted(
            results,
            key=lambda r: r.get("sharpe_ratio", float("-inf"))
            if not math.isnan(r.get("sharpe_ratio", float("nan")))
            else float("-inf"),
            reverse=True,
        )
        for r in sorted_results:
            v = r["variant"]
            sh = r.get("sharpe_ratio", math.nan)
            sh_str = f"{sh:+.3f}" if not math.isnan(sh) else "NaN"
            ret_str = f"{r['total_return']:+.2%}"
            trades = r.get("total_trades", 0)
            flag = "" if trades >= MIN_TRADES else " ⚠️"
            lines.append(f"| {v.describe()} | {sh_str} | {ret_str} | {trades}{flag} |\n")
        lines.append("\n")

    # Overall aggregate ranking
    lines.append("### Overall Ranking (avg Sharpe across valid datasets)\n\n")
    lines.append("| Rank | Variant | Avg Sharpe | Avg Return | Valid/Total | Total Trades |\n")
    lines.append("|------|---------|------------|------------|-------------|-------------|\n")
    rank_labels = {1: "🥇 1", 2: "🥈 2", 3: "🥉 3"}
    for i, entry in enumerate(overall_ranking):
        v = entry["variant"]
        rl = rank_labels.get(i + 1, str(i + 1))
        lines.append(
            f"| {rl} | {v.describe()} "
            f"| {entry['avg_sharpe']:+.3f} "
            f"| {entry['avg_return']:+.2%} "
            f"| {entry['datasets_valid']}/{entry['datasets_total']} "
            f"| {entry['trades_total']} |\n"
        )

    # Winner block
    if winner:
        v = winner["variant"]
        lines += [
            "\n### 🏆 Winner of the Day\n\n",
            f"**{v.describe()}**\n\n",
            f"| Metric | Value |\n",
            f"|--------|-------|\n",
            f"| Avg Sharpe (across datasets) | {winner['avg_sharpe']:+.3f} |\n",
            f"| Avg Return | {winner['avg_return']:+.2%} |\n",
            f"| Valid datasets | {winner['datasets_valid']}/{winner['datasets_total']} |\n",
            f"| Total trades | {winner['trades_total']} |\n",
            f"\n**Params to apply when satisfied:**\n\n",
            f"```\n",
            f"rsi_period     = {v.rsi_period}\n",
            f"rsi_oversold   = {v.rsi_oversold}\n",
            f"rsi_overbought = {v.rsi_overbought}\n",
            f"bb_period      = {v.bb_period}\n",
            f"bb_std         = {v.bb_std}\n",
            f"volume_period  = {v.volume_period}\n",
            f"sma_period     = {v.sma_period}\n",
            f"```\n\n",
            f"> After ~10 consistent wins, update defaults in "
            f"`strategy/statistical_arb.py` to go live.\n",
        ]
    else:
        lines.append("\n> No valid winner — all variants failed on all datasets.\n")

    with _LOG_PATH.open("a") as f:
        f.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive strategy tournament")
    parser.add_argument("--candles", type=int, default=5_000,
                        help="Candles per dataset (default 5000 for speed)")
    args = parser.parse_args()

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("═" * 60)
    print(f"  Cassandra Comprehensive Tournament — {now_str}")
    print(f"  Variants: {len(CANDIDATE_VARIANTS)}  |  "
          f"Datasets: {len(DATASETS)}  |  "
          f"Candles/dataset: {args.candles:,}")
    print("═" * 60)

    # 1. Fetch all datasets up front
    print("\n[1/3] Fetching datasets...")
    datasets: dict[str, tuple[pd.Series, pd.Series]] = {}
    dataset_meta: dict[str, tuple[str, str, int, str, str]] = {}
    for symbol, tf in DATASETS:
        key = f"{symbol} {tf}"
        print(f"  {key}...", end=" ", flush=True)
        close, volume, date_from, date_to = _fetch(symbol, tf, args.candles)
        datasets[key] = (close, volume)
        dataset_meta[key] = (symbol, tf, len(close), date_from, date_to)
        print(f"{len(close):,} candles  ({date_from} → {date_to})")

    # 2. Run all variants on all datasets
    print(f"\n[2/3] Running {len(CANDIDATE_VARIANTS)} variants × {len(DATASETS)} datasets "
          f"= {len(CANDIDATE_VARIANTS) * len(DATASETS)} backtests...\n")

    results_by_dataset: dict[str, list[dict[str, Any]]] = {k: [] for k in datasets}

    for i, variant in enumerate(CANDIDATE_VARIANTS, 1):
        row_sharpes = []
        for ds_key, (close, volume) in datasets.items():
            result = backtest_variant(variant, close, volume)
            results_by_dataset[ds_key].append(result)
            sh = result.get("sharpe_ratio", math.nan)
            trades = result.get("total_trades", 0)
            tag = f"{sh:+.3f}" if (not math.isnan(sh) and trades >= MIN_TRADES) else "  n/a"
            row_sharpes.append(f"{tag:>7}")

        datasets_str = "  ".join(
            f"{sym[0]}/{tf}" for sym, tf in [(s.split("/")[0], t) for s, t in DATASETS]
        )
        sharpes_str = "  ".join(row_sharpes)
        print(f"  [{i:2d}/{len(CANDIDATE_VARIANTS)}] {variant.describe()}")
        print(f"         [{datasets_str}]")
        print(f"         [{sharpes_str}]")

    # 3. Score and rank
    print("\n[3/3] Scoring...")
    overall_ranking = _score(results_by_dataset)

    print("\n" + "─" * 60)
    print("  OVERALL RANKING (avg Sharpe across valid datasets)\n")
    for i, entry in enumerate(overall_ranking[:5], 1):
        v = entry["variant"]
        print(
            f"  {i}. {v.describe()}\n"
            f"     Avg Sharpe={entry['avg_sharpe']:+.3f}  "
            f"Avg Return={entry['avg_return']:+.2%}  "
            f"Valid={entry['datasets_valid']}/{entry['datasets_total']} datasets  "
            f"Trades={entry['trades_total']}"
        )

    if overall_ranking:
        w = overall_ranking[0]
        v = w["variant"]
        print(f"\n  🏆 Winner of the Day: {v.describe()}")
        print(f"     Avg Sharpe={w['avg_sharpe']:+.3f}  "
              f"Avg Return={w['avg_return']:+.2%}")

    _write_log(results_by_dataset, dataset_meta, overall_ranking, args.candles)
    print(f"\n  Full report logged → agents/memory/PENDING_UPGRADES.md")
    print("─" * 60)


if __name__ == "__main__":
    main()
