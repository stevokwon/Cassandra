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


def fetch_balance(exchange: ccxt.Exchange) -> float:
    """Fetch the free USDT balance from the exchange.

    Args:
        exchange: An initialised ccxt Exchange instance.

    Returns:
        Free USDT balance as a float.
    """
    balance_data: dict[str, Any] = exchange.fetch_balance()
    return float(balance_data.get("free", {}).get("USDT", 0.0))
