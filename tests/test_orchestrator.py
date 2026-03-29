"""TDD tests for the Phase 7 shadow pipeline orchestrator."""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from agents.orchestrator import run_shadow_pipeline, write_pending_upgrades
from strategy.optimizer import StrategyVariant


def _make_series(n: int = 60) -> tuple[pd.Series, pd.Series]:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series([100.0 + i * 0.3 for i in range(n)], index=idx, dtype=float)
    volume = pd.Series([1_000.0] * n, index=idx, dtype=float)
    return close, volume


def _mock_backtest(variant, close, volume, **kwargs):
    """Return a fake backtest result with sharpe = rsi_period / 10.0."""
    return {
        "sharpe_ratio": variant.rsi_period / 10.0,
        "total_return": 0.1,
        "max_drawdown": 0.05,
        "total_trades": 5,
        "variant": variant,
    }


def test_run_shadow_pipeline_returns_list() -> None:
    """run_shadow_pipeline() must return a list."""
    close, volume = _make_series()
    with patch("agents.orchestrator.backtest_variant", side_effect=_mock_backtest):
        result = run_shadow_pipeline(close, volume, baseline_sharpe=0.0)
    assert isinstance(result, list)


def test_run_shadow_pipeline_filters_by_baseline_sharpe() -> None:
    """Only variants with sharpe > baseline_sharpe are returned."""
    close, volume = _make_series()
    with patch("agents.orchestrator.backtest_variant", side_effect=_mock_backtest):
        # baseline_sharpe=1.5 → only variants with rsi_period > 15 pass (sharpe > 1.5)
        result = run_shadow_pipeline(close, volume, baseline_sharpe=1.5)
    assert all(r["sharpe_ratio"] > 1.5 for r in result)


def test_run_shadow_pipeline_returns_empty_when_none_beat_baseline() -> None:
    """Returns empty list when no variant beats a very high baseline."""
    close, volume = _make_series()
    with patch("agents.orchestrator.backtest_variant", side_effect=_mock_backtest):
        result = run_shadow_pipeline(close, volume, baseline_sharpe=99.0)
    assert result == []


def test_write_pending_upgrades_appends_entries(tmp_path: Path) -> None:
    """write_pending_upgrades() must append at least one entry to the file."""
    log_file = tmp_path / "PENDING_UPGRADES.md"
    log_file.write_text("# Pending Upgrades\n\n")

    results = [
        {
            "sharpe_ratio": 2.5,
            "total_return": 0.3,
            "max_drawdown": 0.08,
            "total_trades": 12,
            "variant": StrategyVariant(21, 25, 2.5, 25),
        }
    ]
    write_pending_upgrades(results, path=log_file)
    content = log_file.read_text()
    assert "RSI(21)" in content
    assert "2.5" in content  # sharpe ratio appears


def test_write_pending_upgrades_no_op_when_empty(tmp_path: Path) -> None:
    """write_pending_upgrades() must not modify the file when results is empty."""
    log_file = tmp_path / "PENDING_UPGRADES.md"
    original = "# Pending Upgrades\n\n_No entries._\n"
    log_file.write_text(original)
    write_pending_upgrades([], path=log_file)
    assert log_file.read_text() == original
