"""Statistical Swarm — Phase 2 deterministic time-series indicators.

All functions are pure: same inputs always produce same outputs.
No side effects, no I/O, no randomness.
"""
from typing import Literal

import numpy as np
import pandas as pd

Signal = Literal["bullish", "bearish", "neutral"]

# ── RSI ───────────────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 21) -> pd.Series:
    """Compute the Relative Strength Index (RSI) for a close price Series.

    Uses Wilder's smoothed moving average (EMA with alpha=1/period).

    Args:
        close: Time-ordered close price Series with a DatetimeIndex.
        period: Lookback window. Default 14.

    Returns:
        pd.Series of RSI values (0–100). First (period-1) values are NaN.
    """
    delta: pd.Series = close.diff()
    gain: pd.Series = delta.clip(lower=0.0)
    loss: pd.Series = (-delta).clip(lower=0.0)

    avg_gain: pd.Series = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss: pd.Series = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Avoid division by zero: where avg_loss == 0, RSI is 100 (or 50 if avg_gain also 0)
    rs: pd.Series = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi: pd.Series = 100.0 - (100.0 / (1.0 + rs))

    # Flat price: both avg_gain and avg_loss are 0 → RSI = 50
    rsi = rsi.where(avg_loss != 0.0, other=50.0)

    return rsi.rename("rsi")


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands (upper, middle, lower) for a close price Series.

    Args:
        close: Time-ordered close price Series with a DatetimeIndex.
        period: Rolling window for the moving average and standard deviation.
        std_dev: Number of standard deviations for the band width. Default 2.0.

    Returns:
        DataFrame with columns ['upper', 'middle', 'lower'].
        First (period - 1) rows are NaN.
    """
    middle: pd.Series = close.rolling(window=period).mean()
    rolling_std: pd.Series = close.rolling(window=period).std(ddof=1)
    upper: pd.Series = middle + std_dev * rolling_std
    lower: pd.Series = middle - std_dev * rolling_std
    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})


# ── Volume Z-Score ────────────────────────────────────────────────────────────

def compute_volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    """Compute the rolling Z-score of trading volume.

    A Z-score > 2.0 indicates a statistically significant volume spike
    (more than 2 standard deviations above the rolling mean).

    Args:
        volume: Time-ordered volume Series with a DatetimeIndex.
        period: Rolling window for mean and standard deviation. Default 20.

    Returns:
        pd.Series of Z-scores. First (period - 1) values are NaN.
        Returns 0.0 where rolling std is zero (constant volume).
    """
    rolling_mean: pd.Series = volume.rolling(window=period).mean()
    rolling_std: pd.Series = volume.rolling(window=period).std(ddof=1)

    safe_std: pd.Series = rolling_std.replace(0.0, np.nan)  # NaN where std is zero
    z: pd.Series = (volume - rolling_mean) / safe_std
    z = z.fillna(0.0)           # constant-volume windows → z = 0
    z = z.where(rolling_mean.notna())  # warmup rows stay NaN
    return z.rename("volume_zscore")


# ── Signal Aggregation ────────────────────────────────────────────────────────

def _rsi_signal(rsi_value: float) -> Signal:
    """Map a single RSI value to a Signal literal."""
    if rsi_value < 30.0:
        return "bullish"
    if rsi_value > 70.0:
        return "bearish"
    return "neutral"


def _bb_signal(close_value: float, upper: float, lower: float) -> Signal:
    """Map a close price relative to Bollinger Bands to a Signal literal."""
    if close_value < lower:
        return "bullish"
    if close_value > upper:
        return "bearish"
    return "neutral"


def _volume_signal(z_value: float, close_now: float, close_prev: float) -> Signal:
    """Map a volume Z-score and price direction to a Signal literal."""
    if z_value > 2.0:
        if close_now > close_prev:
            return "bullish"
        if close_now < close_prev:
            return "bearish"
    return "neutral"


def generate_signal(close: pd.Series, volume: pd.Series) -> Signal:
    """Generate a consensus Signal from the last available candle.

    Applies majority vote across RSI, Bollinger Bands, and Volume Z-Score.
    Requires at least 20 candles of history (the longest lookback window).

    Args:
        close: Time-ordered close price Series (DatetimeIndex).
        volume: Time-ordered volume Series (DatetimeIndex, same length as close).

    Returns:
        'bullish', 'bearish', or 'neutral'.
    """
    rsi = compute_rsi(close)
    bb = compute_bollinger_bands(close)
    z = compute_volume_zscore(volume)

    # Use the last fully-computed value for each indicator
    rsi_clean = rsi.dropna()
    bb_clean = bb.dropna()
    z_clean = z.dropna()

    # Not enough history yet — return neutral
    if bb_clean.empty:
        return "neutral"

    last_rsi = rsi_clean.iloc[-1] if not rsi_clean.empty else 50.0
    last_bb = bb_clean.iloc[-1]
    last_z = z_clean.iloc[-1] if not z_clean.empty else 0.0
    last_close = close.iloc[-1]
    prev_close = close.iloc[-2] if len(close) >= 2 else last_close

    votes: list[Signal] = [
        _rsi_signal(last_rsi),
        _bb_signal(last_close, last_bb["upper"], last_bb["lower"]),
        _volume_signal(last_z, last_close, prev_close),
    ]

    bullish_count = votes.count("bullish")
    bearish_count = votes.count("bearish")

    if bullish_count >= 2:
        return "bullish"
    if bearish_count >= 2:
        return "bearish"
    return "neutral"


def rolling_signals(
    close: pd.Series,
    volume: pd.Series,
    warmup: int = 20,
) -> pd.Series:
    """Apply generate_signal() across a rolling window of the full Series.

    The first (warmup - 1) positions are set to 'neutral' (insufficient history).

    Args:
        close: Time-ordered close price Series.
        volume: Time-ordered volume Series (same length as close).
        warmup: Minimum candles needed before a real signal is generated. Default 20.

    Returns:
        pd.Series of Signal literals ('bullish'/'bearish'/'neutral'),
        same length and index as close.
    """
    signals: list[Signal] = []
    for i in range(len(close)):
        if i < warmup - 1:
            signals.append("neutral")
        else:
            signals.append(
                generate_signal(close.iloc[: i + 1], volume.iloc[: i + 1])
            )
    return pd.Series(signals, index=close.index, name="signal")
