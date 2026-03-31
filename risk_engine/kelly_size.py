"""Deterministic Fractional Kelly position-sizing math.

All functions are pure: same inputs always produce same outputs.
No side effects, no I/O, no randomness.

EV formula:       EV = p * b - (1 - p)
Kelly formula:    f* = (p * b - q) / b   where q = 1 - p
Fractional Kelly: f_frac = f* * fractional  (default 0.25 = quarter Kelly)
"""


def expected_value(p: float, b: float) -> float:
    """Compute the expected value of a trade.

    Args:
        p: Probability of a winning trade (0.0–1.0).
        b: Payout ratio (profit / amount risked). E.g. 1.0 = 1:1 risk/reward.

    Returns:
        Expected value. Positive means edge. Negative means house advantage.
    """
    return p * b - (1.0 - p)


def kelly_fraction(p: float, b: float, fractional: float = 0.15) -> float:
    """Compute the Fractional Kelly fraction of capital to risk.

    Returns 0.0 if the full Kelly f* <= 0 (no edge or negative edge).

    Args:
        p: Probability of a winning trade (0.0–1.0).
        b: Payout ratio.
        fractional: Scaling factor. Default 0.15 (reduced from 0.25 for drawdown control).

    Returns:
        Fraction of capital to risk, in [0.0, 1.0].
    """
    q = 1.0 - p
    full_kelly = (p * b - q) / b
    if full_kelly <= 0.0:
        return 0.0
    return full_kelly * fractional


def position_size_usdt(
    capital: float,
    p: float,
    b: float,
    fractional: float = 0.15,
) -> float:
    """Compute the USDT position size using Fractional Kelly.

    Args:
        capital: Available capital in USDT.
        p: Probability of winning.
        b: Payout ratio.
        fractional: Kelly scaling factor. Default 0.15 (reduced from 0.25 for drawdown control).

    Returns:
        Position size in USDT. Zero if no edge.
    """
    return capital * kelly_fraction(p, b, fractional)
