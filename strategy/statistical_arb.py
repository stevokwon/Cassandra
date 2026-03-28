"""Statistical Swarm — Phase 2 deterministic time-series indicators.

All functions are pure: same inputs always produce same outputs.
No side effects, no I/O, no randomness.
"""
from typing import Literal

import numpy as np
import pandas as pd

Signal = Literal["bullish", "bearish", "neutral"]

# ── RSI ───────────────────────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
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
