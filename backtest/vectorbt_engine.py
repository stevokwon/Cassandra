"""VectorBT OHLCV ingestion and buy-and-hold simulation engine."""
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt


def load_ohlcv_from_df(df: pd.DataFrame) -> pd.Series:
    """Convert a ccxt-format OHLCV DataFrame into a DatetimeIndex close price Series.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume].
            Timestamps must be Unix milliseconds (int64).

    Returns:
        pd.Series of close prices with a UTC DatetimeIndex.

    Raises:
        KeyError: If 'timestamp' or 'close' columns are missing.
    """
    required = {"timestamp", "close"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing required columns: {missing}")

    index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return pd.Series(df["close"].to_numpy(), index=index, name="close")


def run_buy_and_hold(
    close: pd.Series,
    initial_cash: float = 1_000.0,
    fees: float = 0.001,
) -> dict[str, Any]:
    """Simulate a buy-and-hold strategy and return key performance statistics.

    Buys at the first candle close and holds until the last candle.

    Args:
        close: DatetimeIndex Series of close prices.
        initial_cash: Starting capital in USDT.
        fees: Round-trip fee fraction applied by VectorBT (default 0.1%).

    Returns:
        Dictionary with at minimum 'total_return' (float) and 'sharpe_ratio' (float).
    """
    entries = pd.Series(
        [True] + [False] * (len(close) - 1),
        index=close.index,
    )
    exits = pd.Series(
        [False] * (len(close) - 1) + [True],
        index=close.index,
    )

    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        freq="1h",
    )

    raw_stats = portfolio.stats()

    return {
        "total_return": float(raw_stats.get("Total Return [%]", np.nan)) / 100,
        "sharpe_ratio": float(raw_stats.get("Sharpe Ratio", np.nan)),
        "max_drawdown": float(raw_stats.get("Max Drawdown [%]", np.nan)) / 100,
        "total_trades": int(raw_stats.get("Total Trades", 0)),
    }


def run_strategy_backtest(
    close: pd.Series,
    signals: pd.Series,
    initial_cash: float = 1_000.0,
    fees: float = 0.001,
) -> dict[str, Any]:
    """Run a VectorBT backtest driven by a pre-computed signal Series.

    Enters long on 'bullish' signals, exits on 'bearish' or end-of-series.
    Ignores 'neutral' signals (holds current position).

    Args:
        close: DatetimeIndex Series of close prices.
        signals: Series of Signal literals ('bullish'/'bearish'/'neutral'),
                 same index as close (from rolling_signals()).
        initial_cash: Starting capital in USDT.
        fees: Round-trip fee fraction. Default 0.1%.

    Returns:
        Dictionary with 'total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades'.
    """
    entries: pd.Series = signals == "bullish"
    exits: pd.Series = signals == "bearish"

    # Ensure at least one entry exists to avoid VectorBT errors on empty portfolios
    if not entries.any():
        return {
            "total_return": 0.0,
            "sharpe_ratio": float("nan"),
            "max_drawdown": 0.0,
            "total_trades": 0,
        }

    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        freq="1h",
    )

    raw_stats = portfolio.stats()
    return {
        "total_return": float(raw_stats.get("Total Return [%]", np.nan)) / 100,
        "sharpe_ratio": float(raw_stats.get("Sharpe Ratio", np.nan)),
        "max_drawdown": float(raw_stats.get("Max Drawdown [%]", np.nan)) / 100,
        "total_trades": int(raw_stats.get("Total Trades", 0)),
    }
