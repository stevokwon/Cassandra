"""Phase 5 calibration report — go/no-go gate for live trading.

build_report() assembles backtest stats and optional LLM probability
calibration into a CalibrationReport. The is_live_ready flag is the
single source of truth before switching to EXECUTION_MODE=live.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from risk_engine.metrics import brier_score as _brier_score
from risk_engine.metrics import profit_factor as _profit_factor

# ── Minimum thresholds for live-trading readiness ────────────────────────────
MIN_SHARPE: float = 1.0
MIN_PROFIT_FACTOR: float = 1.2
MAX_BRIER: float = 0.25


@dataclass
class CalibrationReport:
    """Summary of out-of-sample backtest performance and LLM calibration.

    Attributes:
        sharpe_ratio: Annualised Sharpe Ratio from the backtest.
        profit_factor: Gross profit / gross losses ratio.
        max_drawdown: Maximum peak-to-trough drawdown fraction.
        total_return: Total strategy return (e.g. 0.35 = +35%).
        total_trades: Number of round-trip trades executed.
        brier_score: Mean squared probability error (None if not provided).
        is_live_ready: True when all applicable thresholds are met.
    """

    sharpe_ratio: float
    profit_factor: float
    max_drawdown: float
    total_return: float
    total_trades: int
    brier_score: float | None
    is_live_ready: bool


def build_report(
    backtest_stats: dict[str, Any],
    gross_profit: float = 0.0,
    gross_loss: float = 0.0,
    probabilities: list[float] | None = None,
    outcomes: list[int] | None = None,
) -> CalibrationReport:
    """Assemble a CalibrationReport from backtest stats and optional LLM predictions.

    Args:
        backtest_stats: Dict from run_full_backtest() or run_strategy_backtest().
            Must contain 'sharpe_ratio', 'max_drawdown', 'total_return', 'total_trades'.
        gross_profit: Sum of all winning trade profits (USDT). Used for profit_factor.
        gross_loss: Sum of all losing trade losses (USDT, positive value).
        probabilities: LLM confidence values for each prediction (0.0-1.0).
        outcomes: Actual binary outcomes (1=correct direction, 0=wrong).

    Returns:
        CalibrationReport with is_live_ready=True only when all thresholds pass.
    """
    sharpe = float(backtest_stats.get("sharpe_ratio", math.nan))
    pf = _profit_factor(gross_profit, gross_loss)
    bs: float | None = None

    if probabilities and outcomes and len(probabilities) == len(outcomes):
        bs = _brier_score(probabilities, outcomes)

    sharpe_ok = not math.isnan(sharpe) and sharpe >= MIN_SHARPE
    pf_ok = (pf != math.inf and pf >= MIN_PROFIT_FACTOR) or pf == math.inf
    brier_ok = bs is None or bs <= MAX_BRIER

    live_ready = sharpe_ok and pf_ok and brier_ok

    return CalibrationReport(
        sharpe_ratio=sharpe,
        profit_factor=pf,
        max_drawdown=float(backtest_stats.get("max_drawdown", math.nan)),
        total_return=float(backtest_stats.get("total_return", math.nan)),
        total_trades=int(backtest_stats.get("total_trades", 0)),
        brier_score=bs,
        is_live_ready=live_ready,
    )
