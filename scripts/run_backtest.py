#!/usr/bin/env python3
"""Cassandra — Phase 5 Backtest Runner

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --symbol ETH/USDT --timeframe 4h --limit 1000

Options:
    --symbol     Trading pair (default: BTC/USDT)
    --timeframe  Candle size  (default: 1h)
    --limit      Number of candles to fetch (default: 500, max: 1000)
    --capital    Starting capital in USDT (default: 1000)
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from backtest.calibration import MIN_PROFIT_FACTOR, MIN_SHARPE, build_report
from backtest.run_backtest import run_full_backtest
from backtest.vectorbt_engine import load_ohlcv_from_df
from execution.ccxt_client import build_exchange, fetch_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cassandra Phase 5 backtest")
    parser.add_argument("--symbol",    default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1h",       help="Candle timeframe")
    parser.add_argument("--limit",     default=500, type=int, help="Number of candles")
    parser.add_argument("--capital",   default=1000.0, type=float, help="Starting USDT")
    args = parser.parse_args()

    print(f"\n{'═' * 50}")
    print(f"  Cassandra Backtest — {args.symbol} {args.timeframe}")
    print(f"  Candles: {args.limit}  |  Capital: ${args.capital:,.0f} USDT")
    print(f"{'═' * 50}\n")

    # ── 1. Fetch OHLCV ───────────────────────────────────────────────────────
    print("Fetching OHLCV data from Binance testnet...")
    try:
        exchange = build_exchange()
        df = fetch_ohlcv(exchange, symbol=args.symbol, timeframe=args.timeframe, limit=args.limit)
    except Exception as exc:
        print(f"  ERROR: Could not fetch data — {exc}")
        sys.exit(1)

    close = load_ohlcv_from_df(df)
    ts_index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    volume = pd.Series(df["volume"].to_numpy(), index=ts_index)

    print(f"  Fetched {len(close)} candles  "
          f"({close.index[0].strftime('%Y-%m-%d')} → {close.index[-1].strftime('%Y-%m-%d')})\n")

    # ── 2. Run backtest ──────────────────────────────────────────────────────
    print("Running statistical swarm backtest...")
    stats = run_full_backtest(close, volume, initial_cash=args.capital)

    # ── 3. Rough P&L split for profit factor ────────────────────────────────
    gross_profit = max(stats["total_return"] * args.capital, 0.0)
    gross_loss   = max(-stats["total_return"] * args.capital, 1.0)  # floor at 1 to avoid inf
    report = build_report(stats, gross_profit=gross_profit, gross_loss=gross_loss)

    # ── 4. Print results ─────────────────────────────────────────────────────
    def _threshold(value: float, minimum: float) -> str:
        return "✓" if value >= minimum else "✗"

    print(f"\n{'─' * 50}")
    print(f"  BACKTEST RESULTS")
    print(f"{'─' * 50}")
    print(f"  Total Return:    {report.total_return:>+.2%}")
    print(f"  Max Drawdown:    {report.max_drawdown:.2%}")
    print(f"  Total Trades:    {report.total_trades}")
    print()
    print(f"  Sharpe Ratio:    {report.sharpe_ratio:>6.3f}   "
          f"{_threshold(report.sharpe_ratio, MIN_SHARPE)}  (need >= {MIN_SHARPE})")
    print(f"  Profit Factor:   {report.profit_factor:>6.3f}   "
          f"{_threshold(report.profit_factor, MIN_PROFIT_FACTOR)}  (need >= {MIN_PROFIT_FACTOR})")
    print()
    verdict = "LIVE READY" if report.is_live_ready else "NOT LIVE READY"
    if report.is_live_ready:
        print("  ✅  LIVE READY — thresholds met")
    else:
        print("  ⛔  NOT LIVE READY — keep backtesting")
    print(f"{'─' * 50}\n")

    # ── 5. Append run to backtest log ────────────────────────────────────────
    _append_log(
        symbol=args.symbol,
        timeframe=args.timeframe,
        candles=len(close),
        capital=args.capital,
        report=report,
        verdict=verdict,
        date_from=close.index[0].strftime("%Y-%m-%d"),
        date_to=close.index[-1].strftime("%Y-%m-%d"),
    )


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
    """Append a one-line entry to agents/memory/backtest_log.md."""
    log_path = Path(__file__).parent.parent / "agents" / "memory" / "backtest_log.md"

    # Create file with header if it doesn't exist
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
    print(f"  Run logged → agents/memory/backtest_log.md\n")


if __name__ == "__main__":
    main()
