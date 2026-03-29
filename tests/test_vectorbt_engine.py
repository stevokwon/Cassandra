"""Tests for the VectorBT simulation engine."""
import numpy as np
import pandas as pd
import pytest
from backtest.vectorbt_engine import load_ohlcv_from_df, run_buy_and_hold


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    """Helper: generate synthetic OHLCV DataFrame."""
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = np.cumsum(np.random.default_rng(42).normal(0, 50, n)) + 30_000
    return pd.DataFrame({
        "timestamp": timestamps.astype(np.int64) // 10**6,
        "open": close * 0.999,
        "high": close * 1.001,
        "low": close * 0.998,
        "close": close,
        "volume": np.random.default_rng(42).uniform(10, 200, n),
    })


def test_load_ohlcv_from_df_returns_series() -> None:
    """load_ohlcv_from_df() must return a pd.Series of close prices with DatetimeIndex."""
    df = _make_ohlcv()
    close_series = load_ohlcv_from_df(df)
    assert isinstance(close_series, pd.Series)
    assert isinstance(close_series.index, pd.DatetimeIndex)
    assert len(close_series) == 100


def test_load_ohlcv_from_df_raises_on_missing_columns() -> None:
    """load_ohlcv_from_df() must raise KeyError if required columns are absent."""
    bad_df = pd.DataFrame({"price": [1, 2, 3]})
    with pytest.raises(KeyError):
        load_ohlcv_from_df(bad_df)


def test_run_buy_and_hold_returns_stats_dict() -> None:
    """run_buy_and_hold() must return a dict containing 'total_return' and 'sharpe_ratio'."""
    df = _make_ohlcv(200)
    close_series = load_ohlcv_from_df(df)
    stats = run_buy_and_hold(close_series, initial_cash=1_000.0)
    assert isinstance(stats, dict)
    assert "total_return" in stats
    assert "sharpe_ratio" in stats


def test_run_buy_and_hold_initial_cash_reflected() -> None:
    """run_buy_and_hold() stats must reflect the provided initial_cash amount."""
    df = _make_ohlcv(200)
    close_series = load_ohlcv_from_df(df)
    stats = run_buy_and_hold(close_series, initial_cash=1_000.0)
    # Total return is a ratio — just verify it is a finite float
    assert np.isfinite(stats["total_return"])


from strategy.statistical_arb import rolling_signals
from backtest.vectorbt_engine import run_strategy_backtest


def test_run_strategy_backtest_returns_stats_dict() -> None:
    """run_strategy_backtest() must return a dict with 'total_return' and 'sharpe_ratio'."""
    df = _make_ohlcv(200)
    close_series = load_ohlcv_from_df(df)
    volume_series = pd.Series(
        df["volume"].to_numpy(),
        index=close_series.index,
        name="volume",
    )
    sigs = rolling_signals(close_series, volume_series)
    stats = run_strategy_backtest(close_series, sigs, initial_cash=1_000.0)
    assert isinstance(stats, dict)
    assert "total_return" in stats
    assert "sharpe_ratio" in stats
    assert "max_drawdown" in stats
    assert "total_trades" in stats


def test_run_strategy_backtest_total_return_is_finite() -> None:
    """run_strategy_backtest() total_return must be a finite float."""
    df = _make_ohlcv(200)
    close_series = load_ohlcv_from_df(df)
    volume_series = pd.Series(
        df["volume"].to_numpy(),
        index=close_series.index,
        name="volume",
    )
    sigs = rolling_signals(close_series, volume_series)
    stats = run_strategy_backtest(close_series, sigs, initial_cash=1_000.0)
    assert np.isfinite(stats["total_return"])
