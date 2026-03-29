"""TDD tests for the Phase 5 full-pipeline backtest runner."""
import pandas as pd
import pytest

from backtest.run_backtest import run_full_backtest


def _make_series(n: int = 60, start: float = 100.0, slope: float = 0.5) -> tuple[pd.Series, pd.Series]:
    """Synthetic OHLCV series with a gentle upward trend (60 hourly candles)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series([start + i * slope for i in range(n)], index=idx, dtype=float)
    volume = pd.Series([1_000.0] * n, index=idx, dtype=float)
    return close, volume


def test_run_full_backtest_returns_dict() -> None:
    """run_full_backtest() must return a dict."""
    close, volume = _make_series()
    result = run_full_backtest(close, volume)
    assert isinstance(result, dict)


def test_run_full_backtest_has_required_keys() -> None:
    """Result dict must contain total_return, sharpe_ratio, max_drawdown, total_trades."""
    close, volume = _make_series()
    result = run_full_backtest(close, volume)
    for key in ("total_return", "sharpe_ratio", "max_drawdown", "total_trades"):
        assert key in result, f"Missing key: {key}"


def test_run_full_backtest_total_trades_is_non_negative() -> None:
    """total_trades must never be negative."""
    close, volume = _make_series()
    result = run_full_backtest(close, volume)
    assert result["total_trades"] >= 0


def test_run_full_backtest_max_drawdown_is_non_negative() -> None:
    """max_drawdown is a positive fraction from vectorbt (e.g. 0.05 = 5% drawdown) or zero."""
    close, volume = _make_series()
    result = run_full_backtest(close, volume)
    assert result["max_drawdown"] >= 0.0
