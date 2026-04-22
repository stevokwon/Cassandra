#!/usr/bin/env python3
"""Cassandra — Live Trading Runner (one-shot, designed for cron).

Usage:
    python scripts/run_live.py
    python scripts/run_live.py --symbol BTC/USDT --timeframe 1h

Options:
    --symbol     Trading pair (default: BTC/USDT)
    --timeframe  Candle size  (default: 1h)
    --candles    Number of candles to fetch (default: 800)
"""
import argparse
import asyncio
import concurrent.futures
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_POSITION_STATE_PATH = _REPO_ROOT / "agents" / "memory" / "position_state.json"
_TRADE_LOG_PATH = _REPO_ROOT / "agents" / "memory" / "live_trade_log.md"

_LOG_HEADER = (
    "# Live Trade Log\n\n"
    "| Date (UTC) | Symbol | TF | Action | Size USDT | Price | Reasoning |\n"
    "|---|---|---|---|---|---|---|\n"
)


def _load_position_state() -> dict[str, Any]:
    """Load position state from JSON file. Creates default if missing."""
    if not _POSITION_STATE_PATH.exists():
        default: dict[str, Any] = {
            "in_position": False,
            "symbol": "BTC/USDT",
            "side": None,
            "entry_price": None,
            "size_usdt": None,
            "entry_time": None,
        }
        _save_position_state(default)
        return default
    with _POSITION_STATE_PATH.open() as f:
        return json.load(f)


def _save_position_state(state: dict[str, Any]) -> None:
    """Persist position state to JSON file."""
    _POSITION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _POSITION_STATE_PATH.open("w") as f:
        json.dump(state, f, indent=4)


def _append_trade_log(
    timestamp_utc: str,
    symbol: str,
    timeframe: str,
    action: str,
    size_usdt: float | None,
    price: float,
    reasoning: str,
) -> None:
    """Append a row to the markdown trade log."""
    _TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not _TRADE_LOG_PATH.exists():
        _TRADE_LOG_PATH.write_text(_LOG_HEADER)

    size_str = f"{size_usdt:.2f}" if size_usdt else "—"
    # Truncate reasoning to keep table readable
    snippet = reasoning[:80].replace("|", "/") if reasoning else ""

    row = (
        f"| {timestamp_utc} | {symbol} | {timeframe} | {action.upper()} "
        f"| {size_str} | {price:.2f} | {snippet} |\n"
    )
    with _TRADE_LOG_PATH.open("a") as f:
        f.write(row)


def _fetch_headlines() -> list[str]:
    """Run the async scraper inside a fresh event loop (thread-safe)."""
    from agents.tools.news_scraper import scrape_crypto_headlines
    return asyncio.run(scrape_crypto_headlines())


def main() -> None:
    """One-shot live trading run. Designed to be called by cron every hour."""
    # Lazy imports so the module can be imported cleanly without heavy dependencies
    import pandas as pd
    from execution.ccxt_client import (
        build_exchange,
        close_position,
        fetch_balance,
        fetch_ohlcv_bulk,
        place_order,
    )
    from risk_engine.validate_risk import check_drawdown
    from strategy.consensus import TradeDecision, evaluate_consensus
    from strategy.macro_sentiment import MacroSignal, classify_sentiment

    parser = argparse.ArgumentParser(description="Cassandra live trading runner")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument("--candles", type=int, default=800, help="Number of candles (default: 800)")
    args = parser.parse_args()

    symbol: str = args.symbol
    timeframe: str = args.timeframe
    candles: int = args.candles

    now_utc = datetime.now(timezone.utc)
    timestamp_utc = now_utc.strftime("%Y-%m-%d %H:%M")

    logger.info("=== Cassandra live run: %s %s @ %s ===", symbol, timeframe, timestamp_utc)

    # 1. Build exchange (testnet)
    try:
        exchange = build_exchange()
    except ValueError as exc:
        logger.error("Cannot build exchange: %s", exc)
        sys.exit(1)

    # 2. Load position state
    state = _load_position_state()

    # 3. Fetch OHLCV (public endpoint — no API key needed)
    logger.info("Fetching %d candles for %s %s ...", candles, symbol, timeframe)
    try:
        df = fetch_ohlcv_bulk(symbol=symbol, timeframe=timeframe, total_candles=candles)
    except Exception as exc:
        logger.error("OHLCV fetch failed: %s", exc)
        sys.exit(1)

    close: pd.Series = df["close"].reset_index(drop=True)
    volume: pd.Series = df["volume"].reset_index(drop=True)
    last_close: float = float(close.iloc[-1])
    logger.info("Last close price: %.2f", last_close)

    # 4. Fetch balance
    try:
        balance = fetch_balance(exchange)
        logger.info("USDT balance: %.2f", balance)
    except Exception as exc:
        logger.error("Balance fetch failed: %s", exc)
        sys.exit(1)

    # 5. Update peak_equity in state (add field if missing, default = current balance)
    peak_equity: float = float(state.get("peak_equity", 0.0) or 0.0)
    if peak_equity <= 0.0:
        peak_equity = balance
    if balance > peak_equity:
        peak_equity = balance
    state["peak_equity"] = peak_equity

    # 6. Run macro sentiment (async scraper in thread, then classify)
    logger.info("Fetching macro sentiment ...")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            headlines = pool.submit(_fetch_headlines).result(timeout=30)
        macro: MacroSignal = classify_sentiment(headlines)
        logger.info("Macro signal: %s (confidence=%.2f)", macro.signal, macro.confidence)
    except Exception as exc:
        logger.warning("Macro scraper failed (%s) — falling back to neutral", exc)
        macro = MacroSignal(signal="neutral", confidence=0.5, reasoning="Scraper unavailable")

    # 7. Run consensus
    decision: TradeDecision = evaluate_consensus(
        close=close,
        volume=volume,
        macro=macro,
        capital_usdt=balance,
    )
    logger.info(
        "Consensus decision: %s | size=%.2f USDT | EV=%.4f",
        decision.action,
        decision.position_size_usdt,
        decision.expected_value,
    )
    logger.info("Reasoning: %s", decision.reasoning)

    # 8. Drawdown circuit breaker
    if not check_drawdown(balance, peak_equity, max_dd=0.20):
        logger.warning(
            "CIRCUIT BREAKER: drawdown exceeded 20%% (balance=%.2f, peak=%.2f) — skipping trade",
            balance,
            peak_equity,
        )
        _append_trade_log(
            timestamp_utc=timestamp_utc,
            symbol=symbol,
            timeframe=timeframe,
            action="CIRCUIT_BREAKER",
            size_usdt=None,
            price=last_close,
            reasoning=f"Drawdown circuit breaker: balance={balance:.2f}, peak={peak_equity:.2f}",
        )
        _save_position_state(state)
        return

    # 9. Execute
    in_position: bool = bool(state.get("in_position", False))
    action_taken: str = "hold"
    size_logged: float | None = None

    if not in_position and decision.action == "buy":
        logger.info("Placing BUY order: %.2f USDT", decision.position_size_usdt)
        try:
            place_order(exchange, symbol, "buy", decision.position_size_usdt)
            state["in_position"] = True
            state["symbol"] = symbol
            state["side"] = "long"
            state["entry_price"] = last_close
            state["size_usdt"] = decision.position_size_usdt
            state["entry_time"] = now_utc.isoformat()
            action_taken = "buy"
            size_logged = decision.position_size_usdt
            logger.info("BUY order placed successfully.")
        except Exception as exc:
            logger.error("place_order failed: %s — state NOT updated", exc)
            _append_trade_log(
                timestamp_utc=timestamp_utc,
                symbol=symbol,
                timeframe=timeframe,
                action="BUY_ERROR",
                size_usdt=decision.position_size_usdt,
                price=last_close,
                reasoning=f"place_order error: {exc}",
            )
            _save_position_state(state)
            return

    elif in_position and decision.action == "sell":
        logger.info("Closing position for %s", symbol)
        try:
            close_position(exchange, symbol)
            state["in_position"] = False
            state["side"] = None
            state["entry_price"] = None
            state["size_usdt"] = None
            state["entry_time"] = None
            action_taken = "sell"
            logger.info("Position closed successfully.")
        except Exception as exc:
            logger.error("close_position failed: %s — state NOT updated", exc)
            _append_trade_log(
                timestamp_utc=timestamp_utc,
                symbol=symbol,
                timeframe=timeframe,
                action="SELL_ERROR",
                size_usdt=None,
                price=last_close,
                reasoning=f"close_position error: {exc}",
            )
            _save_position_state(state)
            return

    else:
        logger.info("HOLD — %s", decision.reasoning)

    # 10. Always log the run
    _append_trade_log(
        timestamp_utc=timestamp_utc,
        symbol=symbol,
        timeframe=timeframe,
        action=action_taken,
        size_usdt=size_logged,
        price=last_close,
        reasoning=decision.reasoning,
    )

    _save_position_state(state)
    logger.info("Run complete.")


if __name__ == "__main__":
    main()
