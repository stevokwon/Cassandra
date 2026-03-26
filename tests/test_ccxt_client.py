"""Tests for the ccxt testnet exchange client."""
import os
import pytest
from unittest.mock import patch, MagicMock
from execution.ccxt_client import build_exchange, fetch_ohlcv, fetch_balance


def test_build_exchange_returns_binance_instance() -> None:
    """build_exchange() must return a ccxt.binance object configured for testnet."""
    with patch.dict(os.environ, {
        "BINANCE_TESTNET_API_KEY": "fake_key",
        "BINANCE_TESTNET_SECRET": "fake_secret",
        "EXECUTION_MODE": "testnet",
    }):
        exchange = build_exchange()
        import ccxt
        assert isinstance(exchange, ccxt.binance)
        assert exchange.urls["api"] == exchange.urls.get("test", exchange.urls["api"])


def test_build_exchange_raises_if_env_missing() -> None:
    """build_exchange() must raise ValueError when credentials are absent."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="BINANCE_TESTNET_API_KEY"):
            build_exchange()


def test_fetch_ohlcv_returns_dataframe() -> None:
    """fetch_ohlcv() must return a DataFrame with OHLCV columns."""
    import pandas as pd
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.return_value = [
        [1700000000000, 30000.0, 31000.0, 29000.0, 30500.0, 100.0],
        [1700003600000, 30500.0, 31500.0, 30000.0, 31000.0, 120.0],
    ]
    df = fetch_ohlcv(mock_exchange, symbol="BTC/USDT", timeframe="1h", limit=2)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_fetch_balance_returns_usdt_float() -> None:
    """fetch_balance() must return the free USDT balance as a float."""
    mock_exchange = MagicMock()
    mock_exchange.fetch_balance.return_value = {
        "free": {"USDT": 1000.0, "BTC": 0.0}
    }
    balance = fetch_balance(mock_exchange)
    assert isinstance(balance, float)
    assert balance == 1000.0
