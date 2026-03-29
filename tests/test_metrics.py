"""TDD tests for calibration metrics (Brier Score, Profit Factor)."""
import math

import pytest

from risk_engine.metrics import brier_score, profit_factor


def test_brier_score_perfect_predictions() -> None:
    """Perfect prediction: p=1.0 for outcome=1 → BS = (1-1)^2 = 0.0."""
    assert brier_score([1.0], [1]) == 0.0


def test_brier_score_worst_case() -> None:
    """Worst prediction: p=1.0 for outcome=0 → BS = (1-0)^2 = 1.0."""
    assert brier_score([1.0], [0]) == 1.0


def test_profit_factor_normal() -> None:
    """gross_profit=200, gross_loss=100 → 2.0."""
    assert profit_factor(gross_profit=200.0, gross_loss=100.0) == 2.0


def test_profit_factor_no_losses() -> None:
    """gross_loss=0 → math.inf (no losing trades)."""
    assert profit_factor(gross_profit=500.0, gross_loss=0.0) == math.inf
