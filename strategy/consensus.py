"""Phase 4 Consensus Engine — combines Phase 2 Statistical Swarm + Phase 3 Macro Swarm.

A trade is only generated when both conditions are met:
  1. Both swarms agree on direction (bullish/bearish — neither can be neutral).
  2. Expected Value is strictly positive (EV > 0).

Position sizing applies three layered caps (Position Sizer + Exposure Coach skills):
  - Fractional Kelly base size (0.15 fractional)
  - Regime scale: bull=1.0, sideways=0.5, bear=0 (no new longs)
  - ATR-based risk cap: max risk = 1% of capital per trade
  - Portfolio cap: hard floor of 6% of capital regardless of other factors
"""
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from risk_engine.kelly_size import expected_value, kelly_fraction
from risk_engine.validate_risk import validate_trade
from strategy.macro_sentiment import MacroSignal
from strategy.statistical_arb import (
    Signal,
    classify_market_regime,
    compute_atr,
    generate_signal,
)

Action = Literal["buy", "sell", "hold"]

_DEFAULT_PAYOUT_RATIO = 1.0   # 1:1 risk/reward
_MAX_PORTFOLIO_RISK = 0.06    # 6% hard cap on any single position
_RISK_PER_TRADE = 0.01        # 1% of capital risked per trade (ATR method)
_REGIME_SCALE = {"bull": 1.0, "sideways": 0.5, "bear": 0.0}


@dataclass
class TradeDecision:
    """Output of the consensus engine.

    Attributes:
        action: 'buy', 'sell', or 'hold'.
        position_size_usdt: USDT amount to trade. 0.0 when action='hold'.
        expected_value: EV of the trade. 0.0 when action='hold'.
        kelly_fraction: Kelly fraction used. 0.0 when action='hold'.
        reasoning: Human-readable explanation for the Phase 6 audit log.
        regime: Market regime at time of decision ('bull', 'bear', 'sideways').
    """

    action: Action
    position_size_usdt: float
    expected_value: float
    kelly_fraction: float
    reasoning: str
    regime: str = "sideways"


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

    if not validate_trade(ev, min_ev=0.05):
        return TradeDecision(
            action="hold",
            position_size_usdt=0.0,
            expected_value=ev,
            kelly_fraction=0.0,
            reasoning=f"EV gate failed: EV={ev:.4f} (p={p:.2f}, b={b:.2f})",
        )

    # Both gates passed — determine action direction
    action: Action = "buy" if stat_signal == "bullish" else "sell"

    # Gate 3: regime filter — no new long entries in confirmed bear market
    regime = classify_market_regime(close)
    regime_scale = _REGIME_SCALE[regime]
    if action == "buy" and regime_scale == 0.0:
        return TradeDecision(
            action="hold",
            position_size_usdt=0.0,
            expected_value=ev,
            kelly_fraction=0.0,
            reasoning=f"Bear regime — long entry suppressed (SMA50 < SMA200)",
            regime=regime,
        )

    # Position sizing: smallest of Kelly × regime_scale, ATR cap, portfolio cap
    kf = kelly_fraction(p, b)
    kelly_pos = capital_usdt * kf * regime_scale

    # ATR-based cap: risk no more than 1% of capital per trade
    atr = compute_atr(close)
    atr_valid = atr.dropna()
    if not atr_valid.empty and float(atr_valid.iloc[-1]) > 0:
        atr_val = float(atr_valid.iloc[-1])
        atr_pos = (capital_usdt * _RISK_PER_TRADE) / atr_val * float(close.iloc[-1])
    else:
        atr_pos = kelly_pos  # flat/test data — skip ATR cap

    # Hard portfolio cap: never risk more than 6% of capital
    portfolio_cap = capital_usdt * _MAX_PORTFOLIO_RISK

    pos = min(kelly_pos, atr_pos, portfolio_cap)

    return TradeDecision(
        action=action,
        position_size_usdt=pos,
        expected_value=ev,
        kelly_fraction=kf,
        reasoning=(
            f"Consensus {action} [{regime} regime, scale={regime_scale:.1f}]: "
            f"stat={stat_signal}, macro={macro.signal} "
            f"(p={p:.2f}, EV={ev:.4f}, Kelly={kf:.4f}, "
            f"Kelly_pos={kelly_pos:.2f}, ATR_pos={atr_pos:.2f}, "
            f"final={pos:.2f} USDT) — {macro.reasoning}"
        ),
        regime=regime,
    )
