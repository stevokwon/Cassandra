"""Hard risk guardrails for trade validation and drawdown protection.

All functions are pure: same inputs always produce same outputs.
These are the safety gates — never bypass them.
"""


def validate_trade(ev: float, min_ev: float = 0.0) -> bool:
    """Check whether a trade's Expected Value clears the minimum threshold.

    Args:
        ev: Expected value of the trade (from expected_value()).
        min_ev: Minimum EV required. Default 0.0 (must be strictly positive).

    Returns:
        True if ev > min_ev, False otherwise.
    """
    return ev > min_ev


def check_drawdown(
    current_equity: float,
    peak_equity: float,
    max_dd: float = 0.20,
) -> bool:
    """Check whether current drawdown is within the maximum allowed limit.

    A circuit breaker: returns False when drawdown would halt trading.

    Args:
        current_equity: Current portfolio value in USDT.
        peak_equity: Highest recorded portfolio value in USDT.
        max_dd: Maximum drawdown fraction before circuit breaker trips. Default 0.20 (20%).

    Returns:
        True if drawdown is within limit (trading allowed).
        False if drawdown exceeds max_dd (circuit breaker tripped).
    """
    if peak_equity <= 0.0:
        return True  # No peak recorded yet — allow trading
    drawdown = (peak_equity - current_equity) / peak_equity
    return drawdown < max_dd
