"""Phase 7 Shadow Pipeline — parameterised strategy variant optimizer.

Tests alternative RSI / Bollinger Band / Volume Z-Score configurations
against historical OHLCV data and returns backtest stats for each.
All I/O is pure — no file writes here; orchestrator.py handles persistence.
"""
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from backtest.vectorbt_engine import run_strategy_backtest
from strategy.statistical_arb import (
    _bb_signal,
    _rsi_signal,
    _volume_signal,
    compute_bollinger_bands,
    compute_rsi,
    compute_volume_zscore,
)

Signal = Literal["bullish", "bearish", "neutral"]


@dataclass
class StrategyVariant:
    """A candidate set of indicator parameters for the statistical swarm.

    Attributes:
        rsi_period: RSI lookback window.
        bb_period: Bollinger Band rolling window.
        bb_std: Number of standard deviations for band width.
        volume_period: Volume Z-score rolling window.
    """

    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    volume_period: int = 20

    def describe(self) -> str:
        """Human-readable one-line description for PENDING_UPGRADES.md."""
        return (
            f"RSI({self.rsi_period}) + BB({self.bb_period}, σ={self.bb_std}) "
            f"+ VolZ({self.volume_period})"
        )


# Catalogue of parameter combinations to test in the shadow pipeline.
CANDIDATE_VARIANTS: list[StrategyVariant] = [
    StrategyVariant(rsi_period=7,  bb_period=15, bb_std=1.5, volume_period=15),
    StrategyVariant(rsi_period=10, bb_period=20, bb_std=2.0, volume_period=15),
    StrategyVariant(rsi_period=14, bb_period=20, bb_std=2.0, volume_period=20),  # default
    StrategyVariant(rsi_period=14, bb_period=15, bb_std=1.5, volume_period=20),
    StrategyVariant(rsi_period=21, bb_period=20, bb_std=2.0, volume_period=20),
    StrategyVariant(rsi_period=21, bb_period=25, bb_std=2.5, volume_period=25),
    StrategyVariant(rsi_period=10, bb_period=25, bb_std=2.5, volume_period=20),
    StrategyVariant(rsi_period=14, bb_period=25, bb_std=2.0, volume_period=25),
]


def generate_signals_with_params(
    close: pd.Series,
    volume: pd.Series,
    variant: StrategyVariant,
    warmup: int | None = None,
) -> pd.Series:
    """Generate rolling signals using a custom StrategyVariant configuration."""
    _warmup = warmup if warmup is not None else variant.bb_period
    rsi = compute_rsi(close, period=variant.rsi_period)
    bb = compute_bollinger_bands(close, period=variant.bb_period, std_dev=variant.bb_std)
    z = compute_volume_zscore(volume, period=variant.volume_period)

    signals: list[Signal] = []
    for i in range(len(close)):
        if i < _warmup - 1:
            signals.append("neutral")
            continue

        rsi_clean = rsi.iloc[: i + 1].dropna()
        bb_clean = bb.iloc[: i + 1].dropna()
        z_clean = z.iloc[: i + 1].dropna()

        if bb_clean.empty:
            signals.append("neutral")
            continue

        last_rsi = float(rsi_clean.iloc[-1]) if not rsi_clean.empty else 50.0
        last_bb = bb_clean.iloc[-1]
        last_z = float(z_clean.iloc[-1]) if not z_clean.empty else 0.0
        last_close = float(close.iloc[i])
        prev_close = float(close.iloc[i - 1]) if i > 0 else last_close

        votes: list[Signal] = [
            _rsi_signal(last_rsi),
            _bb_signal(last_close, float(last_bb["upper"]), float(last_bb["lower"])),
            _volume_signal(last_z, last_close, prev_close),
        ]

        if votes.count("bullish") >= 2:
            signals.append("bullish")
        elif votes.count("bearish") >= 2:
            signals.append("bearish")
        else:
            signals.append("neutral")

    return pd.Series(signals, index=close.index, name="signal")


def backtest_variant(
    variant: StrategyVariant,
    close: pd.Series,
    volume: pd.Series,
    initial_cash: float = 1_000.0,
    fees: float = 0.001,
) -> dict[str, Any]:
    """Backtest a StrategyVariant and return stats + the variant itself."""
    signals = generate_signals_with_params(close, volume, variant)
    stats = run_strategy_backtest(close, signals, initial_cash, fees)
    return {**stats, "variant": variant}
