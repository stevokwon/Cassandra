"""Unified ccxt exchange client routing to Binance testnet."""
import os
from typing import Any

import ccxt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OHLCV_COLUMNS: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]


def build_exchange() -> ccxt.binance:
    """Construct and return a ccxt.binance instance pointed at the testnet.

    Returns:
        A configured ccxt.binance exchange object.

    Raises:
        ValueError: If required environment variables are missing.
    """
    api_key = os.getenv("BINANCE_TESTNET_API_KEY")
    secret = os.getenv("BINANCE_TESTNET_SECRET")

    if not api_key:
        raise ValueError("BINANCE_TESTNET_API_KEY is not set in environment.")
    if not secret:
        raise ValueError("BINANCE_TESTNET_SECRET is not set in environment.")

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
    })
    exchange.set_sandbox_mode(True)
    return exchange


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch OHLCV candles from the exchange and return as a DataFrame.

    Args:
        exchange: An initialised ccxt Exchange instance.
        symbol: Market symbol, e.g. 'BTC/USDT'.
        timeframe: Candle duration string, e.g. '1h', '4h', '1d'.
        limit: Number of candles to retrieve.

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume].
    """
    raw: list[list[Any]] = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return pd.DataFrame(raw, columns=OHLCV_COLUMNS)


def fetch_ohlcv_bulk(
    symbol: str,
    timeframe: str = "1h",
    total_candles: int = 5000,
) -> pd.DataFrame:
    """Fetch a large number of OHLCV candles from Binance production (public endpoint).

    Uses the public Binance API (no API key required) to access full historical
    data. Trade execution still routes to the testnet — this is for backtesting only.
    Paginates automatically in batches of 1000.

    Args:
        symbol: Market symbol, e.g. 'BTC/USDT'.
        timeframe: Candle duration string, e.g. '1h', '4h'.
        total_candles: Total number of candles to retrieve.

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume],
        sorted oldest-first, deduplicated.
    """
    public = ccxt.binance({"enableRateLimit": True})
    tf_ms = int(public.parse_timeframe(timeframe) * 1000)
    batch = 1000
    all_rows: list[list[Any]] = []

    # Start from now and walk backwards in batches
    until_ms: int | None = None
    while len(all_rows) < total_candles:
        fetch_limit = min(batch, total_candles - len(all_rows))
        params: dict[str, Any] = {}
        if until_ms is not None:
            params["endTime"] = until_ms
        raw = public.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_limit, params=params)
        if not raw:
            break
        all_rows = raw + all_rows
        until_ms = int(raw[0][0]) - tf_ms  # step back one candle before oldest fetched
        if len(raw) < fetch_limit:
            break  # no more history available

    df = pd.DataFrame(all_rows, columns=OHLCV_COLUMNS)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df.tail(total_candles).reset_index(drop=True)


def fetch_balance(exchange: ccxt.Exchange) -> float:
    """Fetch the free USDT balance from the exchange.

    Args:
        exchange: An initialised ccxt Exchange instance.

    Returns:
        Free USDT balance as a float.
    """
    balance_data: dict[str, Any] = exchange.fetch_balance()
    return float(balance_data.get("free", {}).get("USDT", 0.0))
