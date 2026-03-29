"""Phase 4 Consensus Engine — combines Phase 2 Statistical Swarm + Phase 3 Macro Swarm.

A trade is only generated when both conditions are met:
  1. Both swarms agree on direction (bullish/bearish — neither can be neutral).
  2. Expected Value is strictly positive (EV > 0).

Position sizing uses Fractional Kelly (default fractional=0.25).
"""
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from risk_engine.kelly_size import expected_value, kelly_fraction, position_size_usdt
from risk_engine.validate_risk import validate_trade
from strategy.macro_sentiment import MacroSignal
from strategy.statistical_arb import Signal, generate_signal

Action = Literal["buy", "sell", "hold"]

_DEFAULT_PAYOUT_RATIO = 1.0  # 1:1 risk/reward


@dataclass
class TradeDecision:
    """Output of the consensus engine.

    Attributes:
        action: 'buy', 'sell', or 'hold'.
        position_size_usdt: USDT amount to trade. 0.0 when action='hold'.
        expected_value: EV of the trade. 0.0 when action='hold'.
        kelly_fraction: Kelly fraction used. 0.0 when action='hold'.
        reasoning: Human-readable explanation for the Phase 6 audit log.
    """

    action: Action
    position_size_usdt: float
    expected_value: float
    kelly_fraction: float
    reasoning: str


def evaluate_consensus(
    close: pd.Series,
    volume: pd.Series,
    macro: MacroSignal,
    capital_usdt: float,
    payout_ratio: float = _DEFAULT_PAYOUT_RATIO,
) -> TradeDecision:
    """Evaluate whether Phase 2 and Phase 3 signals agree enough to trade.

    Gate 1: Both swarms must agree on direction (same non-neutral signal).
    Gate 2: Expected Value must be strictly positive.

    Args:
        close: Time-ordered close price Series (Phase 2 input).
        volume: Time-ordered volume Series (Phase 2 input).
        macro: MacroSignal from Phase 3 (signal + confidence probability).
        capital_usdt: Current available capital in USDT.
        payout_ratio: Expected payout per unit risked. Default 1.0 (1:1).

    Returns:
        TradeDecision with action, position_size_usdt, ev, kelly_fraction, reasoning.
    """
    stat_signal: Signal = generate_signal(close, volume)

    # Gate 1: both swarms must agree on a non-neutral direction
    if stat_signal == "neutral" or macro.signal == "neutral" or stat_signal != macro.signal:
        return TradeDecision(
            action="hold",
            position_size_usdt=0.0,
            expected_value=0.0,
            kelly_fraction=0.0,
            reasoning=f"Swarm disagreement: stat={stat_signal}, macro={macro.signal}",
        )

    # Gate 2: EV must be strictly positive
    p = macro.confidence
    b = payout_ratio
    ev = expected_value(p, b)

    if not validate_trade(ev):
        return TradeDecision(
            action="hold",
            position_size_usdt=0.0,
            expected_value=ev,
            kelly_fraction=0.0,
            reasoning=f"EV gate failed: EV={ev:.4f} (p={p:.2f}, b={b:.2f})",
        )

    # Both gates passed — size the position
    kf = kelly_fraction(p, b)
    pos = position_size_usdt(capital_usdt, p, b)
    action: Action = "buy" if stat_signal == "bullish" else "sell"

    return TradeDecision(
        action=action,
        position_size_usdt=pos,
        expected_value=ev,
        kelly_fraction=kf,
        reasoning=(
            f"Consensus {action}: stat={stat_signal}, macro={macro.signal} "
            f"(p={p:.2f}, EV={ev:.4f}, Kelly={kf:.4f}, size={pos:.2f} USDT) — "
            f"{macro.reasoning}"
        ),
    )
