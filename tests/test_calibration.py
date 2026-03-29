"""TDD tests for the Phase 5 calibration report."""
import pytest

from backtest.calibration import CalibrationReport, build_report

# ── Fixtures ──────────────────────────────────────────────────────────────────

_GOOD_STATS = {
    "total_return": 0.35,
    "sharpe_ratio": 1.5,
    "max_drawdown": -0.08,
    "total_trades": 12,
}

_POOR_STATS = {
    "total_return": 0.05,
    "sharpe_ratio": 0.4,
    "max_drawdown": -0.25,
    "total_trades": 3,
}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build_report_returns_calibration_report() -> None:
    """build_report() must return a CalibrationReport dataclass instance."""
    result = build_report(_GOOD_STATS)
    assert isinstance(result, CalibrationReport)


def test_is_live_ready_true_when_thresholds_met() -> None:
    """Sharpe>=1.0 and profit_factor>=1.2 -> is_live_ready=True."""
    result = build_report(
        _GOOD_STATS,
        gross_profit=350.0,
        gross_loss=200.0,
    )
    assert result.is_live_ready is True


def test_is_live_ready_false_when_sharpe_too_low() -> None:
    """Sharpe < 1.0 -> is_live_ready=False regardless of other metrics."""
    result = build_report(
        _POOR_STATS,
        gross_profit=50.0,
        gross_loss=40.0,
    )
    assert result.is_live_ready is False


def test_brier_score_included_when_predictions_provided() -> None:
    """brier_score field is populated when probabilities+outcomes are given."""
    result = build_report(
        _GOOD_STATS,
        gross_profit=350.0,
        gross_loss=200.0,
        probabilities=[0.8, 0.6, 0.7],
        outcomes=[1, 1, 0],
    )
    assert result.brier_score is not None
    assert 0.0 <= result.brier_score <= 1.0


def test_brier_score_none_when_no_predictions_provided() -> None:
    """brier_score field is None when no predictions are passed."""
    result = build_report(_GOOD_STATS, gross_profit=350.0, gross_loss=200.0)
    assert result.brier_score is None
