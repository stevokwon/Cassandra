"""Phase 5 full-pipeline backtest runner.

Wires together Phase 2 rolling signals and the VectorBT engine into a
single callable. Use this for out-of-sample calibration runs.
"""
from typing import Any

import pandas as pd

from backtest.vectorbt_engine import run_strategy_backtest
from strategy.statistical_arb import rolling_signals


def run_full_backtest(
    close: pd.Series,
    volume: pd.Series,
    initial_cash: float = 1_000.0,
    fees: float = 0.001,
) -> dict[str, Any]:
    """Run the full Phase 2 statistical strategy through the VectorBT engine.

    Generates rolling signals from the close/volume series, then feeds them
    into the VectorBT backtest. Use synthetic or historical OHLCV data.

    Args:
        close: DatetimeIndex Series of close prices (at least 20 candles).
        volume: DatetimeIndex Series of volume (same index as close).
        initial_cash: Starting capital in USDT. Default 1 000.
        fees: Round-trip fee fraction. Default 0.1%.

    Returns:
        Dict with 'total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades'.
    """
    signals = rolling_signals(close, volume)
    return run_strategy_backtest(close, signals, initial_cash, fees)
