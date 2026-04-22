"""Tests for the ccxt testnet exchange client."""
import os
import pytest
from unittest.mock import patch, MagicMock
from execution.ccxt_client import (
    build_exchange,
    fetch_ohlcv,
    fetch_balance,
    place_order,
    close_position,
)


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


def test_fetch_balance_returns_zero_when_usdt_missing() -> None:
    """fetch_balance() must return 0.0 when USDT is absent from the balance."""
    mock_exchange = MagicMock()
    mock_exchange.fetch_balance.return_value = {"free": {}}
    balance = fetch_balance(mock_exchange)
    assert balance == 0.0


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------

def test_place_order_buy_calls_create_order_with_correct_args() -> None:
    """place_order(side='buy') must call create_order with 'buy' and correct quantity."""
    mock_exchange = MagicMock()
    mock_exchange.fetch_ticker.return_value = {"last": 50000.0}
    expected_order = {"id": "order-1", "side": "buy"}
    mock_exchange.create_order.return_value = expected_order

    result = place_order(mock_exchange, symbol="BTC/USDT", side="buy", amount_usdt=100.0)

    mock_exchange.fetch_ticker.assert_called_once_with("BTC/USDT")
    mock_exchange.create_order.assert_called_once_with(
        "BTC/USDT", "market", "buy", 100.0 / 50000.0
    )
    assert result == expected_order


def test_place_order_sell_calls_create_order_with_correct_args() -> None:
    """place_order(side='sell') must call create_order with 'sell' and correct quantity."""
    mock_exchange = MagicMock()
    mock_exchange.fetch_ticker.return_value = {"last": 40000.0}
    expected_order = {"id": "order-2", "side": "sell"}
    mock_exchange.create_order.return_value = expected_order

    result = place_order(mock_exchange, symbol="ETH/USDT", side="sell", amount_usdt=200.0)

    mock_exchange.create_order.assert_called_once_with(
        "ETH/USDT", "market", "sell", 200.0 / 40000.0
    )
    assert result == expected_order


def test_place_order_invalid_side_raises_value_error() -> None:
    """place_order() must raise ValueError when side is not 'buy' or 'sell'."""
    mock_exchange = MagicMock()
    with pytest.raises(ValueError, match="Invalid side"):
        place_order(mock_exchange, symbol="BTC/USDT", side="hold", amount_usdt=50.0)


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------

def test_close_position_with_open_long_places_sell_order() -> None:
    """close_position() must sell when a long position (contracts > 0) is open."""
    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.return_value = [
        {"contracts": 0.5, "symbol": "BTC/USDT"}
    ]
    expected_order = {"id": "close-1", "side": "sell"}
    mock_exchange.create_order.return_value = expected_order

    result = close_position(mock_exchange, "BTC/USDT")

    mock_exchange.fetch_positions.assert_called_once_with(["BTC/USDT"])
    mock_exchange.create_order.assert_called_once_with("BTC/USDT", "market", "sell", 0.5)
    assert result == expected_order


def test_close_position_with_open_short_places_buy_order() -> None:
    """close_position() must buy when a short position (contracts < 0) is open."""
    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.return_value = [
        {"contracts": -1.0, "symbol": "BTC/USDT"}
    ]
    expected_order = {"id": "close-2", "side": "buy"}
    mock_exchange.create_order.return_value = expected_order

    result = close_position(mock_exchange, "BTC/USDT")

    mock_exchange.create_order.assert_called_once_with("BTC/USDT", "market", "buy", 1.0)
    assert result == expected_order


def test_close_position_returns_none_when_no_open_position() -> None:
    """close_position() must return None when no position is open."""
    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.return_value = [
        {"contracts": 0, "symbol": "BTC/USDT"}
    ]

    result = close_position(mock_exchange, "BTC/USDT")

    mock_exchange.create_order.assert_not_called()
    assert result is None


def test_close_position_returns_none_when_positions_empty() -> None:
    """close_position() must return None when the positions list is empty."""
    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.return_value = []

    result = close_position(mock_exchange, "BTC/USDT")

    mock_exchange.create_order.assert_not_called()
    assert result is None
