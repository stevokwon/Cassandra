"""TDD test suite for strategy/statistical_arb.py — Phase 2 Statistical Swarm."""
import numpy as np
import pandas as pd
import pytest
from strategy.statistical_arb import compute_rsi


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_close(values: list[float]) -> pd.Series:
    """Wrap a list of floats into a DatetimeIndex Series."""
    index = pd.date_range("2024-01-01", periods=len(values), freq="1h")
    return pd.Series(values, index=index, name="close")


# ── RSI ───────────────────────────────────────────────────────────────────────

def test_rsi_returns_series_same_length() -> None:
    """compute_rsi() must return a Series of the same length as the input."""
    close = _make_close([float(i) for i in range(1, 51)])
    rsi = compute_rsi(close, period=14)
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(close)


def test_rsi_values_bounded_0_to_100() -> None:
    """All non-NaN RSI values must be in the range [0, 100]."""
    rng = np.random.default_rng(0)
    close = _make_close((30_000 + np.cumsum(rng.normal(0, 200, 100))).tolist())
    rsi = compute_rsi(close, period=14)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_first_period_minus_one_values_are_nan() -> None:
    """The first (period - 1) values of RSI must be NaN (insufficient history)."""
    close = _make_close([float(i) for i in range(1, 51)])
    period = 14
    rsi = compute_rsi(close, period=period)
    assert rsi.iloc[: period - 1].isna().all()


def test_rsi_constant_price_returns_nan_or_50() -> None:
    """A flat price series has zero gains and losses — RSI should be NaN or 50."""
    close = _make_close([100.0] * 30)
    rsi = compute_rsi(close, period=14)
    valid = rsi.dropna()
    # All valid values must be 50 (no gain, no loss) or the series is all-NaN
    assert valid.empty or (valid == 50.0).all()


# ── Bollinger Bands ───────────────────────────────────────────────────────────

from strategy.statistical_arb import compute_bollinger_bands


def test_bollinger_bands_returns_dataframe_with_three_columns() -> None:
    """compute_bollinger_bands() must return a DataFrame with columns [upper, middle, lower]."""
    close = _make_close([float(30_000 + i) for i in range(50)])
    bb = compute_bollinger_bands(close, period=20)
    assert isinstance(bb, pd.DataFrame)
    assert list(bb.columns) == ["upper", "middle", "lower"]
    assert len(bb) == len(close)


def test_bollinger_bands_upper_ge_middle_ge_lower() -> None:
    """upper >= middle >= lower must hold for all non-NaN rows."""
    rng = np.random.default_rng(1)
    close = _make_close((30_000 + np.cumsum(rng.normal(0, 100, 60))).tolist())
    bb = compute_bollinger_bands(close, period=20).dropna()
    assert (bb["upper"] >= bb["middle"]).all()
    assert (bb["middle"] >= bb["lower"]).all()


def test_bollinger_bands_middle_is_rolling_mean() -> None:
    """The 'middle' band must equal the rolling mean of the close prices."""
    close = _make_close([float(i) for i in range(1, 51)])
    period = 20
    bb = compute_bollinger_bands(close, period=period)
    expected_middle = close.rolling(window=period).mean()
    pd.testing.assert_series_equal(bb["middle"], expected_middle, check_names=False)


def test_bollinger_bands_first_period_minus_one_rows_are_nan() -> None:
    """The first (period - 1) rows must be NaN (insufficient history)."""
    close = _make_close([float(i) for i in range(1, 51)])
    period = 20
    bb = compute_bollinger_bands(close, period=period)
    assert bb.iloc[: period - 1].isna().all().all()
