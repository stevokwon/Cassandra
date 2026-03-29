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


from strategy.statistical_arb import compute_volume_zscore


# ── Volume Z-Score ────────────────────────────────────────────────────────────

def _make_volume(values: list[float]) -> pd.Series:
    """Wrap a list of floats into a DatetimeIndex volume Series."""
    index = pd.date_range("2024-01-01", periods=len(values), freq="1h")
    return pd.Series(values, index=index, name="volume")


def test_volume_zscore_returns_series_same_length() -> None:
    """compute_volume_zscore() must return a Series of the same length as the input."""
    volume = _make_volume([float(i + 1) for i in range(50)])
    z = compute_volume_zscore(volume, period=20)
    assert isinstance(z, pd.Series)
    assert len(z) == len(volume)


def test_volume_zscore_first_period_minus_one_values_are_nan() -> None:
    """The first (period - 1) values must be NaN."""
    volume = _make_volume([float(i + 1) for i in range(50)])
    period = 20
    z = compute_volume_zscore(volume, period=period)
    assert z.iloc[: period - 1].isna().all()


def test_volume_zscore_constant_volume_is_nan_or_zero() -> None:
    """A flat volume series has zero std — z-score should be NaN or 0."""
    volume = _make_volume([100.0] * 40)
    z = compute_volume_zscore(volume, period=20)
    valid = z.dropna()
    assert valid.empty or (valid == 0.0).all()


def test_volume_zscore_spike_gives_high_positive_value() -> None:
    """A large volume spike must produce a Z-score well above 2.0."""
    base = [100.0] * 25
    spike = base + [10_000.0]  # one massive outlier at position 25
    volume = _make_volume(spike)
    z = compute_volume_zscore(volume, period=20)
    assert z.iloc[-1] > 2.0


from strategy.statistical_arb import generate_signal, rolling_signals


# ── Signal Aggregation ────────────────────────────────────────────────────────

def _make_series(values: list[float], name: str = "close") -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1h")
    return pd.Series(values, index=index, name=name)


def test_generate_signal_returns_valid_literal() -> None:
    """generate_signal() must return one of 'bullish', 'bearish', 'neutral'."""
    rng = np.random.default_rng(7)
    close = _make_series((30_000 + np.cumsum(rng.normal(0, 100, 60))).tolist())
    volume = _make_series(rng.uniform(50, 200, 60).tolist(), name="volume")
    sig = generate_signal(close, volume)
    assert sig in ("bullish", "bearish", "neutral")


def test_generate_signal_neutral_on_flat_data() -> None:
    """Flat price and volume should produce a neutral signal (no directional edge)."""
    close = _make_series([100.0] * 60)
    volume = _make_series([100.0] * 60, name="volume")
    sig = generate_signal(close, volume)
    assert sig == "neutral"


def test_generate_signal_bullish_on_oversold_conditions() -> None:
    """Simulate oversold conditions: a sudden crash should push RSI < 30 and price < lower BB."""
    # Stable period followed by a sudden sharp drop on the last candle.
    # The rolling std is small (stable history), so a large drop breaks below the lower BB.
    # RSI also drops into oversold territory (< 30) → majority bullish.
    close_values = [100.0] * 58 + [100.0, 60.0]  # sudden 40% crash on last candle
    close = _make_series(close_values)
    volume = _make_series([100.0] * 60, name="volume")
    sig = generate_signal(close, volume)
    # RSI < 30 (bullish) + price < lower BB (bullish) → majority bullish
    assert sig == "bullish"


def test_rolling_signals_returns_series_of_correct_length() -> None:
    """rolling_signals() must return a Series with the same length as the input."""
    rng = np.random.default_rng(8)
    close = _make_series((30_000 + np.cumsum(rng.normal(0, 100, 80))).tolist())
    volume = _make_series(rng.uniform(50, 200, 80).tolist(), name="volume")
    sigs = rolling_signals(close, volume)
    assert isinstance(sigs, pd.Series)
    assert len(sigs) == len(close)


def test_rolling_signals_values_are_valid_literals() -> None:
    """Every value in rolling_signals() must be a valid Signal literal."""
    rng = np.random.default_rng(9)
    close = _make_series((30_000 + np.cumsum(rng.normal(0, 100, 80))).tolist())
    volume = _make_series(rng.uniform(50, 200, 80).tolist(), name="volume")
    sigs = rolling_signals(close, volume)
    assert set(sigs.unique()).issubset({"bullish", "bearish", "neutral"})
