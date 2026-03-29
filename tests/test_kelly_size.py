"""TDD tests for Fractional Kelly position-sizing math."""
import pytest

from risk_engine.kelly_size import expected_value, kelly_fraction, position_size_usdt


def test_expected_value_positive() -> None:
    """EV = p*b - (1-p). p=0.6, b=1.0 → 0.6 - 0.4 = 0.2."""
    assert abs(expected_value(p=0.6, b=1.0) - 0.2) < 1e-9


def test_expected_value_negative() -> None:
    """p=0.3, b=1.0 → 0.3 - 0.7 = -0.4."""
    assert abs(expected_value(p=0.3, b=1.0) - (-0.4)) < 1e-9


def test_expected_value_zero_at_breakeven() -> None:
    """p=0.5, b=1.0 → 0.5 - 0.5 = 0.0."""
    assert abs(expected_value(p=0.5, b=1.0)) < 1e-9


def test_kelly_fraction_with_edge() -> None:
    """p=0.6, b=1.0: f* = (0.6-0.4)/1.0 = 0.2; fractional=0.25 → 0.05."""
    result = kelly_fraction(p=0.6, b=1.0, fractional=0.25)
    assert abs(result - 0.05) < 1e-9


def test_kelly_fraction_no_edge_returns_zero() -> None:
    """p=0.4, b=1.0: f* = (0.4-0.6)/1.0 = -0.2 ≤ 0 → returns 0.0."""
    result = kelly_fraction(p=0.4, b=1.0)
    assert result == 0.0


def test_position_size_usdt_scales_with_capital() -> None:
    """capital=1000, p=0.6, b=1.0, fractional=0.25 → 1000 * 0.05 = 50.0."""
    result = position_size_usdt(capital=1000.0, p=0.6, b=1.0, fractional=0.25)
    assert abs(result - 50.0) < 1e-6
