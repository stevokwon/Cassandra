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


# ── ATR Approximation (close-only, for position sizing) ──────────────────────

def compute_atr(close: pd.Series, period: int = 14) -> pd.Series:
    """Approximate Average True Range from close prices only.

    Uses the rolling mean of absolute candle-to-candle price changes as a
    proxy for ATR. Used by the Position Sizer to scale positions inversely
    to current volatility.

    Args:
        close: Time-ordered close price Series with a DatetimeIndex.
        period: Lookback window. Default 14.

    Returns:
        pd.Series of approximate ATR values (same units as price).
        First (period-1) values are NaN.
    """
    return close.diff().abs().rolling(window=period).mean().rename("atr")


# ── Market Regime Classifier (Exposure Coach / Macro Regime Detector) ────────

def classify_market_regime(close: pd.Series) -> str:
    """Classify the macro market regime using a dual SMA filter.

    Bull  = price > SMA(50) AND SMA(50) > SMA(200)  — confirmed uptrend
    Bear  = price < SMA(50) AND SMA(50) < SMA(200)  — confirmed downtrend
    Sideways = everything else (mixed / insufficient data)

    Used by the consensus engine to scale position size:
      bull → full Kelly, sideways → half Kelly, bear → no new longs.

    Args:
        close: Time-ordered close price Series (min ~200 candles for full signal).

    Returns:
        'bull', 'bear', or 'sideways'.
    """
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    sma50_valid = sma50.dropna()
    sma200_valid = sma200.dropna()

    if sma50_valid.empty or sma200_valid.empty:
        return "sideways"

    last_close = float(close.iloc[-1])
    last_sma50 = float(sma50_valid.iloc[-1])
    last_sma200 = float(sma200_valid.iloc[-1])

    if last_close > last_sma50 and last_sma50 > last_sma200:
        return "bull"
    if last_close < last_sma50 and last_sma50 < last_sma200:
        return "bear"
    return "sideways"


# ── Simple Moving Average (Trend Filter) ─────────────────────────────────────

def compute_sma(close: pd.Series, period: int = 200) -> pd.Series:
    """Compute a Simple Moving Average used as a macro trend-direction filter.

    Only used as a regime gate — bullish signals are suppressed when price is
    below the SMA, and bearish signals are suppressed when price is above it.

    Args:
        close: Time-ordered close price Series with a DatetimeIndex.
        period: Lookback window. Default 200.

    Returns:
        pd.Series of SMA values. First (period-1) values are NaN.
    """
    return close.rolling(window=period).mean().rename("sma")


# ── Signal Aggregation ────────────────────────────────────────────────────────

def _rsi_signal(rsi_value: float) -> Signal:
    """Map a single RSI value to a Signal literal.

    Thresholds relaxed to 35/65 (from 30/70) so the regime filter has
    enough signal flow to work with — confirmed by tournament 2026-04-07.
    """
    if rsi_value < 35.0:
        return "bullish"
    if rsi_value > 65.0:
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


def generate_signal(
    close: pd.Series,
    volume: pd.Series,
    sma_period: int = 200,
) -> Signal:
    """Generate a consensus Signal from the last available candle.

    Applies majority vote across RSI, Bollinger Bands, and Volume Z-Score,
    then gates the result through an SMA trend filter: bullish signals are
    only emitted when price is above the SMA (uptrend) and bearish signals
    only when price is below it (downtrend). This prevents mean-reversion
    entries against the macro trend.

    Args:
        close: Time-ordered close price Series (DatetimeIndex).
        volume: Time-ordered volume Series (DatetimeIndex, same length as close).
        sma_period: Period for the SMA trend filter. Set to 0 to disable.

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

    signal: Signal = "neutral"
    if bullish_count >= 2:
        signal = "bullish"
    elif bearish_count >= 2:
        signal = "bearish"

    # SMA trend gate: only trade in the direction of the macro trend.
    # Suppresses bullish signals below SMA (downtrend) and bearish signals
    # above SMA (uptrend) to avoid catching falling knives or shorting rallies.
    if sma_period > 0 and signal != "neutral":
        sma = compute_sma(close, sma_period)
        sma_clean = sma.dropna()
        if not sma_clean.empty:
            last_sma = float(sma_clean.iloc[-1])
            if signal == "bullish" and last_close < last_sma:
                return "neutral"
            if signal == "bearish" and last_close > last_sma:
                return "neutral"

    return signal


def rolling_signals(
    close: pd.Series,
    volume: pd.Series,
    warmup: int = 20,
    sma_period: int = 200,
) -> pd.Series:
    """Apply generate_signal() across a rolling window of the full Series.

    The first max(warmup, sma_period) - 1 positions are 'neutral' to ensure
    both indicator warm-up and a valid SMA reading before any signal fires.

    Args:
        close: Time-ordered close price Series.
        volume: Time-ordered volume Series (same length as close).
        warmup: Minimum candles for indicator warm-up. Default 20.
        sma_period: Passed to generate_signal() for the trend gate. Default 200.

    Returns:
        pd.Series of Signal literals ('bullish'/'bearish'/'neutral'),
        same length and index as close.
    """
    effective_warmup = max(warmup, sma_period) if sma_period > 0 else warmup
    signals: list[Signal] = []
    for i in range(len(close)):
        if i < effective_warmup - 1:
            signals.append("neutral")
        else:
            signals.append(
                generate_signal(close.iloc[: i + 1], volume.iloc[: i + 1], sma_period)
            )
    return pd.Series(signals, index=close.index, name="signal")
