"""TDD tests for the Phase 7 strategy optimizer."""
from unittest.mock import patch

import pandas as pd
import pytest

from strategy.optimizer import (
    CANDIDATE_VARIANTS,
    StrategyVariant,
    backtest_variant,
    generate_signals_with_params,
)


def _make_series(n: int = 60) -> tuple[pd.Series, pd.Series]:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series([100.0 + i * 0.3 for i in range(n)], index=idx, dtype=float)
    volume = pd.Series([1_000.0] * n, index=idx, dtype=float)
    return close, volume


def test_strategy_variant_describe_returns_string() -> None:
    """StrategyVariant.describe() must return a non-empty string."""
    v = StrategyVariant(rsi_period=14, bb_period=20, bb_std=2.0, volume_period=20)
    assert isinstance(v.describe(), str)
    assert len(v.describe()) > 0


def test_generate_signals_with_params_returns_series() -> None:
    """generate_signals_with_params() must return a pd.Series of signal strings."""
    close, volume = _make_series()
    variant = StrategyVariant(rsi_period=14, bb_period=20, bb_std=2.0, volume_period=20)
    result = generate_signals_with_params(close, volume, variant)
    assert isinstance(result, pd.Series)
    assert set(result.unique()).issubset({"bullish", "bearish", "neutral"})


def test_backtest_variant_returns_dict_with_sharpe() -> None:
    """backtest_variant() must return a dict containing 'sharpe_ratio'."""
    close, volume = _make_series()
    variant = StrategyVariant(rsi_period=14, bb_period=20, bb_std=2.0, volume_period=20)
    result = backtest_variant(variant, close, volume)
    assert isinstance(result, dict)
    assert "sharpe_ratio" in result


def test_backtest_variant_includes_variant_in_result() -> None:
    """backtest_variant() result must include the variant under key 'variant'."""
    close, volume = _make_series()
    variant = StrategyVariant(rsi_period=7, bb_period=15, bb_std=1.5, volume_period=15)
    result = backtest_variant(variant, close, volume)
    assert "variant" in result
    assert result["variant"] is variant


def test_candidate_variants_is_nonempty_list() -> None:
    """CANDIDATE_VARIANTS must be a non-empty list of StrategyVariant instances."""
    assert isinstance(CANDIDATE_VARIANTS, list)
    assert len(CANDIDATE_VARIANTS) >= 4
    assert all(isinstance(v, StrategyVariant) for v in CANDIDATE_VARIANTS)
