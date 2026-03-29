"""Calibration metrics for post-hoc model evaluation.

Used in Phase 5 (backtesting) to measure probability calibration and profitability.
"""
import math


def brier_score(probabilities: list[float], outcomes: list[int]) -> float:
    """Compute the Brier Score for a set of probabilistic predictions.

    Brier Score = mean((p_i - o_i)^2)
    Perfect calibration → 0.0. Worst case → 1.0.

    Args:
        probabilities: List of predicted probabilities (0.0–1.0).
        outcomes: List of actual binary outcomes (0 or 1), same length.

    Returns:
        Mean squared error of probability forecasts.

    Raises:
        ValueError: If inputs are empty or of different lengths.
    """
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be non-empty and equal length")
    n = len(probabilities)
    return sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / n


def profit_factor(gross_profit: float, gross_loss: float) -> float:
    """Compute the Profit Factor (gross profit / gross loss).

    A value > 1.0 indicates a profitable strategy.
    Returns math.inf if gross_loss is 0 (no losing trades).

    Args:
        gross_profit: Sum of all winning trade profits (positive float).
        gross_loss: Sum of all losing trade losses (positive float, i.e. abs value).

    Returns:
        Profit factor ratio. math.inf if no losing trades.
    """
    if gross_loss == 0.0:
        return math.inf
    return gross_profit / gross_loss
