"""TDD tests for trade validation and drawdown circuit breaker."""
import pytest

from risk_engine.validate_risk import check_drawdown, validate_trade


def test_validate_trade_passes_positive_ev() -> None:
    """EV = 0.1 > 0.0 (min_ev default) → True."""
    assert validate_trade(ev=0.1) is True


def test_validate_trade_fails_zero_ev() -> None:
    """EV = 0.0 is NOT > 0.0 → False (strictly positive required)."""
    assert validate_trade(ev=0.0) is False


def test_validate_trade_fails_negative_ev() -> None:
    """EV = -0.2 → False."""
    assert validate_trade(ev=-0.2) is False


def test_check_drawdown_within_limit() -> None:
    """equity=850, peak=1000 → drawdown=15% < 20% max → True (trading allowed)."""
    assert check_drawdown(current_equity=850.0, peak_equity=1000.0, max_dd=0.20) is True


def test_check_drawdown_exceeds_limit() -> None:
    """equity=790, peak=1000 → drawdown=21% > 20% max → False (circuit breaker)."""
    assert check_drawdown(current_equity=790.0, peak_equity=1000.0, max_dd=0.20) is False
