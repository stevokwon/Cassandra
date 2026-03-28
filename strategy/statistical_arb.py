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
