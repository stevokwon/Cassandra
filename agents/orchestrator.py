"""Phase 7 Shadow Pipeline Orchestrator.

Runs all CANDIDATE_VARIANTS against historical OHLCV data, filters those
that beat the baseline Sharpe, and appends winners to PENDING_UPGRADES.md.

IMPORTANT: Nothing here auto-merges or modifies live strategy code.
Human approval via PENDING_UPGRADES.md is mandatory.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strategy.optimizer import CANDIDATE_VARIANTS, StrategyVariant, backtest_variant

import pandas as pd

_DEFAULT_UPGRADES_PATH = Path(__file__).parent / "memory" / "PENDING_UPGRADES.md"


def run_shadow_pipeline(
    close: pd.Series,
    volume: pd.Series,
    baseline_sharpe: float = 0.0,
    min_trades: int = 3,
) -> list[dict[str, Any]]:
    """Test all CANDIDATE_VARIANTS and return those that beat the baseline Sharpe.

    Args:
        close: Historical close price Series for backtesting.
        volume: Historical volume Series.
        baseline_sharpe: Current strategy's Sharpe Ratio. Only variants that
            exceed this threshold are returned.
        min_trades: Minimum trades a variant must make to be considered valid.

    Returns:
        List of backtest result dicts (with 'variant' key) sorted by
        sharpe_ratio descending. Empty list if nothing beats the baseline.
    """
    winners: list[dict[str, Any]] = []

    for variant in CANDIDATE_VARIANTS:
        result = backtest_variant(variant, close, volume)
        sharpe = result.get("sharpe_ratio", math.nan)
        trades = result.get("total_trades", 0)

        if math.isnan(sharpe):
            continue
        if trades < min_trades:
            continue
        if sharpe > baseline_sharpe:
            winners.append(result)

    return sorted(winners, key=lambda r: r["sharpe_ratio"], reverse=True)


def write_pending_upgrades(
    results: list[dict[str, Any]],
    path: Path = _DEFAULT_UPGRADES_PATH,
) -> None:
    """Append winning strategy variants to PENDING_UPGRADES.md for human review.

    Does nothing when results is empty (preserves existing file content).

    Args:
        results: List of backtest result dicts from run_shadow_pipeline().
        path: Path to the PENDING_UPGRADES.md file.
    """
    if not results:
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [f"\n## Shadow Pipeline Run — {now}\n\n"]

    for r in results:
        variant: StrategyVariant = r["variant"]
        lines.append(f"### {variant.describe()}\n\n")
        lines.append(f"- **Sharpe Ratio:** {r['sharpe_ratio']:.3f}\n")
        lines.append(f"- **Total Return:** {r['total_return']:+.2%}\n")
        lines.append(f"- **Max Drawdown:** {r['max_drawdown']:.2%}\n")
        lines.append(f"- **Total Trades:** {r['total_trades']}\n")
        lines.append(
            f"- **Params:** rsi_period={variant.rsi_period}, "
            f"bb_period={variant.bb_period}, bb_std={variant.bb_std}, "
            f"volume_period={variant.volume_period}\n"
        )
        lines.append(
            "\n> Warning: Human approval required before applying. "
            "Update `strategy/statistical_arb.py` defaults manually after review.\n\n"
        )

    with path.open("a") as f:
        f.writelines(lines)
