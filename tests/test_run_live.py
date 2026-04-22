"""Tests for scripts/run_live.py — BUY path, HOLD path, circuit breaker."""
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Lightweight stand-ins (no external deps) ──────────────────────────────────

@dataclass
class _MacroSignal:
    """Duck-typed stand-in for strategy.macro_sentiment.MacroSignal."""
    signal: str
    confidence: float
    reasoning: str


@dataclass
class _TradeDecision:
    """Duck-typed stand-in for strategy.consensus.TradeDecision."""
    action: str
    position_size_usdt: float
    expected_value: float
    kelly_fraction: float
    reasoning: str
    regime: str = "bull"


def _make_ohlcv_df(n: int = 50, price: float = 50_000.0) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame for patching fetch_ohlcv_bulk."""
    return pd.DataFrame(
        {
            "timestamp": range(n),
            "open": [price] * n,
            "high": [price * 1.01] * n,
            "low": [price * 0.99] * n,
            "close": [price] * n,
            "volume": [100.0] * n,
        }
    )


_NEUTRAL_MACRO = _MacroSignal(signal="neutral", confidence=0.5, reasoning="test neutral")


def _make_decision(action: str = "buy", size_usdt: float = 50.0) -> _TradeDecision:
    return _TradeDecision(
        action=action,
        position_size_usdt=size_usdt,
        expected_value=0.1 if action != "hold" else 0.0,
        kelly_fraction=0.05,
        reasoning=f"Test decision: {action}",
        regime="bull",
    )


def _make_mock_modules(
    mock_exchange: MagicMock | None = None,
    mock_balance: float = 1_000.0,
    mock_ohlcv: pd.DataFrame | None = None,
    mock_decision: _TradeDecision | None = None,
    mock_place_order: MagicMock | None = None,
    mock_close_position: MagicMock | None = None,
) -> dict[str, types.ModuleType]:
    """Build a dict of fake sys.modules entries for all heavy dependencies."""
    if mock_exchange is None:
        mock_exchange = MagicMock()
    if mock_ohlcv is None:
        mock_ohlcv = _make_ohlcv_df()
    if mock_decision is None:
        mock_decision = _make_decision("hold")
    if mock_place_order is None:
        mock_place_order = MagicMock(return_value={"id": "order-mock"})
    if mock_close_position is None:
        mock_close_position = MagicMock(return_value=None)

    # execution.ccxt_client
    ccxt_mod = types.ModuleType("execution.ccxt_client")
    ccxt_mod.build_exchange = MagicMock(return_value=mock_exchange)  # type: ignore[attr-defined]
    ccxt_mod.fetch_ohlcv_bulk = MagicMock(return_value=mock_ohlcv)  # type: ignore[attr-defined]
    ccxt_mod.fetch_balance = MagicMock(return_value=mock_balance)  # type: ignore[attr-defined]
    ccxt_mod.place_order = mock_place_order  # type: ignore[attr-defined]
    ccxt_mod.close_position = mock_close_position  # type: ignore[attr-defined]

    # strategy.macro_sentiment
    macro_mod = types.ModuleType("strategy.macro_sentiment")
    macro_mod.MacroSignal = _MacroSignal  # type: ignore[attr-defined]
    macro_mod.classify_sentiment = MagicMock(return_value=_NEUTRAL_MACRO)  # type: ignore[attr-defined]

    # strategy.consensus
    consensus_mod = types.ModuleType("strategy.consensus")
    consensus_mod.TradeDecision = _TradeDecision  # type: ignore[attr-defined]
    consensus_mod.evaluate_consensus = MagicMock(return_value=mock_decision)  # type: ignore[attr-defined]

    # risk_engine.validate_risk
    risk_mod = types.ModuleType("risk_engine.validate_risk")
    from risk_engine.validate_risk import check_drawdown  # real implementation
    risk_mod.check_drawdown = check_drawdown  # type: ignore[attr-defined]

    return {
        "execution.ccxt_client": ccxt_mod,
        "strategy.macro_sentiment": macro_mod,
        "strategy.consensus": consensus_mod,
        "risk_engine.validate_risk": risk_mod,
    }


def _run_main(
    state_file: Path,
    log_file: Path,
    mock_modules: dict[str, types.ModuleType],
) -> None:
    """Helper: inject mock sys.modules and call main() with tmp paths."""
    import scripts.run_live as run_live

    # Patch _fetch_headlines to avoid playwright dependency
    with patch.dict(sys.modules, mock_modules):
        with patch.object(run_live, "_fetch_headlines", return_value=[]):
            original_state = run_live._POSITION_STATE_PATH
            original_log = run_live._TRADE_LOG_PATH
            run_live._POSITION_STATE_PATH = state_file
            run_live._TRADE_LOG_PATH = log_file
            try:
                with patch("sys.argv", ["run_live.py"]):
                    run_live.main()
            finally:
                run_live._POSITION_STATE_PATH = original_state
                run_live._TRADE_LOG_PATH = original_log


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_import_no_side_effects() -> None:
    """Importing scripts.run_live must not execute any trading logic."""
    import scripts.run_live  # noqa: F401 — just verifying clean import


def test_buy_path_places_order_and_updates_state(tmp_path: Path) -> None:
    """When not in position and decision=buy, place_order is called and state is updated."""
    state_file = tmp_path / "position_state.json"
    log_file = tmp_path / "live_trade_log.md"

    state_file.write_text(json.dumps({
        "in_position": False,
        "symbol": "BTC/USDT",
        "side": None,
        "entry_price": None,
        "size_usdt": None,
        "entry_time": None,
    }))

    mock_place_order = MagicMock(return_value={"id": "order-123"})
    mock_close_position = MagicMock()

    mocks = _make_mock_modules(
        mock_decision=_make_decision("buy", 50.0),
        mock_place_order=mock_place_order,
        mock_close_position=mock_close_position,
    )

    _run_main(state_file=state_file, log_file=log_file, mock_modules=mocks)

    # place_order must have been called once with side="buy"
    mock_place_order.assert_called_once()
    assert mock_place_order.call_args[0][2] == "buy"

    # close_position must NOT have been called
    mock_close_position.assert_not_called()

    # State must be updated to in_position=True
    updated = json.loads(state_file.read_text())
    assert updated["in_position"] is True
    assert updated["side"] == "long"
    assert updated["size_usdt"] == 50.0
    assert updated["entry_price"] is not None

    # Log must contain a BUY row
    assert "BUY" in log_file.read_text()


def test_hold_path_does_not_place_order(tmp_path: Path) -> None:
    """When consensus returns hold, place_order is never called."""
    state_file = tmp_path / "position_state.json"
    log_file = tmp_path / "live_trade_log.md"

    state_file.write_text(json.dumps({
        "in_position": False,
        "symbol": "BTC/USDT",
        "side": None,
        "entry_price": None,
        "size_usdt": None,
        "entry_time": None,
    }))

    mock_place_order = MagicMock()
    mock_close_position = MagicMock()

    mocks = _make_mock_modules(
        mock_decision=_make_decision("hold", 0.0),
        mock_place_order=mock_place_order,
        mock_close_position=mock_close_position,
    )

    _run_main(state_file=state_file, log_file=log_file, mock_modules=mocks)

    mock_place_order.assert_not_called()
    mock_close_position.assert_not_called()

    updated = json.loads(state_file.read_text())
    assert updated["in_position"] is False

    # Log must still contain a HOLD row (always log)
    assert "HOLD" in log_file.read_text()


def test_circuit_breaker_blocks_trade(tmp_path: Path) -> None:
    """When drawdown exceeds 20%, place_order must not be called."""
    state_file = tmp_path / "position_state.json"
    log_file = tmp_path / "live_trade_log.md"

    # peak_equity=1000, balance=750 → 25% drawdown → circuit breaker trips
    state_file.write_text(json.dumps({
        "in_position": False,
        "symbol": "BTC/USDT",
        "side": None,
        "entry_price": None,
        "size_usdt": None,
        "entry_time": None,
        "peak_equity": 1_000.0,
    }))

    mock_place_order = MagicMock()
    mock_close_position = MagicMock()

    mocks = _make_mock_modules(
        mock_balance=750.0,  # 25% below peak → circuit breaker trips
        mock_decision=_make_decision("buy", 50.0),
        mock_place_order=mock_place_order,
        mock_close_position=mock_close_position,
    )

    _run_main(state_file=state_file, log_file=log_file, mock_modules=mocks)

    mock_place_order.assert_not_called()
    mock_close_position.assert_not_called()

    # Log must contain circuit breaker entry
    assert "CIRCUIT_BREAKER" in log_file.read_text()
