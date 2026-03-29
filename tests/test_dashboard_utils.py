"""TDD tests for dashboard shared utilities."""
from pathlib import Path

import pandas as pd
import pytest


def test_parse_backtest_log_returns_dataframe(tmp_path, monkeypatch) -> None:
    """parse_backtest_log() returns a DataFrame from a valid log file."""
    import dashboard.utils as utils

    log = tmp_path / "backtest_log.md"
    log.write_text(
        "# Backtest Log\n\n"
        "| Date (UTC) | Symbol | TF | Candles | Period | Return | Sharpe | PF | DD | Trades | Verdict |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| 2026-03-01 10:00 | BTC/USDT | 1h | 500 | 2024-01-01->2024-03-01 | +5.00% | 1.200 | 1.30 | 3.00% | 45 | LIVE READY |\n"
    )
    monkeypatch.setattr(utils, "_BACKTEST_LOG", log)
    df = utils.parse_backtest_log()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_parse_backtest_log_returns_empty_when_file_missing(tmp_path, monkeypatch) -> None:
    """parse_backtest_log() returns empty DataFrame when file does not exist."""
    import dashboard.utils as utils

    monkeypatch.setattr(utils, "_BACKTEST_LOG", tmp_path / "nonexistent.md")
    df = utils.parse_backtest_log()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_parse_backtest_log_newest_first(tmp_path, monkeypatch) -> None:
    """parse_backtest_log() returns rows newest-first."""
    import dashboard.utils as utils

    log = tmp_path / "backtest_log.md"
    log.write_text(
        "# Backtest Log\n\n"
        "| Date (UTC) | Symbol | TF | Candles | Period | Return | Sharpe | PF | DD | Trades | Verdict |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| 2026-01-01 10:00 | BTC/USDT | 1h | 500 | 2024-01-01->2024-03-01 | +5.00% | 1.200 | 1.30 | 3.00% | 45 | LIVE READY |\n"
        "| 2026-02-01 10:00 | ETH/USDT | 4h | 500 | 2024-03-01->2024-06-01 | +3.00% | 0.900 | 1.10 | 5.00% | 30 | NOT LIVE READY |\n"
    )
    monkeypatch.setattr(utils, "_BACKTEST_LOG", log)
    df = utils.parse_backtest_log()
    assert "2026-02-01" in df.iloc[0]["Date (UTC)"]


def test_latest_backtest_stats_returns_dict(tmp_path, monkeypatch) -> None:
    """latest_backtest_stats() returns a non-empty dict from a populated log."""
    import dashboard.utils as utils

    log = tmp_path / "backtest_log.md"
    log.write_text(
        "# Backtest Log\n\n"
        "| Date (UTC) | Symbol | TF | Candles | Period | Return | Sharpe | PF | DD | Trades | Verdict |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| 2026-03-01 10:00 | BTC/USDT | 1h | 500 | 2024-01-01->2024-03-01 | +5.00% | 1.200 | 1.30 | 3.00% | 45 | LIVE READY |\n"
    )
    monkeypatch.setattr(utils, "_BACKTEST_LOG", log)
    result = utils.latest_backtest_stats()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_latest_backtest_stats_empty_when_no_log(tmp_path, monkeypatch) -> None:
    """latest_backtest_stats() returns empty dict when log file is absent."""
    import dashboard.utils as utils

    monkeypatch.setattr(utils, "_BACKTEST_LOG", tmp_path / "nonexistent.md")
    assert utils.latest_backtest_stats() == {}


def test_load_pending_upgrades_returns_message_when_file_missing(tmp_path, monkeypatch) -> None:
    """load_pending_upgrades() returns empty-state message when file is absent."""
    import dashboard.utils as utils

    monkeypatch.setattr(utils, "_PENDING_UPGRADES", tmp_path / "nonexistent.md")
    result = utils.load_pending_upgrades()
    assert "No variants" in result
