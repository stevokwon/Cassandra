"""TDD tests for the Phase 4 consensus engine."""
from unittest.mock import patch

import pandas as pd
import pytest

from strategy.consensus import TradeDecision, evaluate_consensus
from strategy.macro_sentiment import MacroSignal


def _make_series(n: int = 30, price: float = 100.0) -> tuple[pd.Series, pd.Series]:
    """Minimal close+volume Series for testing (content doesn't matter — generate_signal is mocked)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series([price] * n, index=idx, dtype=float)
    volume = pd.Series([1000.0] * n, index=idx, dtype=float)
    return close, volume


def test_consensus_returns_trade_decision_instance() -> None:
    """evaluate_consensus() must always return a TradeDecision dataclass."""
    close, volume = _make_series()
    macro = MacroSignal(signal="bullish", confidence=0.7, reasoning="test")
    with patch("strategy.consensus.generate_signal", return_value="bullish"):
        result = evaluate_consensus(close, volume, macro, capital_usdt=1000.0)
    assert isinstance(result, TradeDecision)


def test_consensus_both_bullish_returns_buy() -> None:
    """Both swarms bullish + positive EV → action='buy'."""
    close, volume = _make_series()
    macro = MacroSignal(signal="bullish", confidence=0.7, reasoning="test")
    with patch("strategy.consensus.generate_signal", return_value="bullish"):
        result = evaluate_consensus(close, volume, macro, capital_usdt=1000.0)
    assert result.action == "buy"


def test_consensus_both_bearish_returns_sell() -> None:
    """Both swarms bearish + positive EV → action='sell'."""
    close, volume = _make_series()
    macro = MacroSignal(signal="bearish", confidence=0.7, reasoning="test")
    with patch("strategy.consensus.generate_signal", return_value="bearish"):
        result = evaluate_consensus(close, volume, macro, capital_usdt=1000.0)
    assert result.action == "sell"


def test_consensus_disagree_returns_hold() -> None:
    """Phase 2 bullish but Phase 3 bearish → no consensus → hold."""
    close, volume = _make_series()
    macro = MacroSignal(signal="bearish", confidence=0.7, reasoning="test")
    with patch("strategy.consensus.generate_signal", return_value="bullish"):
        result = evaluate_consensus(close, volume, macro, capital_usdt=1000.0)
    assert result.action == "hold"


def test_consensus_stat_neutral_returns_hold() -> None:
    """Phase 2 neutral → no directional signal → hold regardless of macro."""
    close, volume = _make_series()
    macro = MacroSignal(signal="bullish", confidence=0.7, reasoning="test")
    with patch("strategy.consensus.generate_signal", return_value="neutral"):
        result = evaluate_consensus(close, volume, macro, capital_usdt=1000.0)
    assert result.action == "hold"


def test_consensus_low_confidence_ev_gate_fails_returns_hold() -> None:
    """p=0.4, b=1.0 → EV = 0.4 - 0.6 = -0.2 → EV gate fails → hold."""
    close, volume = _make_series()
    macro = MacroSignal(signal="bullish", confidence=0.4, reasoning="test")
    with patch("strategy.consensus.generate_signal", return_value="bullish"):
        result = evaluate_consensus(close, volume, macro, capital_usdt=1000.0)
    assert result.action == "hold"


def test_consensus_buy_has_positive_position_size() -> None:
    """Valid buy decision must have position_size_usdt > 0."""
    close, volume = _make_series()
    macro = MacroSignal(signal="bullish", confidence=0.7, reasoning="test")
    with patch("strategy.consensus.generate_signal", return_value="bullish"):
        result = evaluate_consensus(close, volume, macro, capital_usdt=1000.0)
    assert result.position_size_usdt > 0.0


def test_consensus_hold_has_zero_position_size() -> None:
    """Hold decisions must have position_size_usdt == 0.0."""
    close, volume = _make_series()
    macro = MacroSignal(signal="bearish", confidence=0.7, reasoning="test")
    with patch("strategy.consensus.generate_signal", return_value="bullish"):
        result = evaluate_consensus(close, volume, macro, capital_usdt=1000.0)
    assert result.position_size_usdt == 0.0
