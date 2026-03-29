"""Shared data-loading utilities for the Cassandra dashboard.

`load_ohlcv` is wrapped with @st.cache_data(ttl=60) so results are cached
for 60 seconds and refreshed automatically on each Streamlit rerun.
File-reading helpers are plain functions — fast enough to not need caching.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.ccxt_client import fetch_ohlcv_bulk

_BACKTEST_LOG = Path(__file__).parent.parent / "agents" / "memory" / "backtest_log.md"
_PENDING_UPGRADES = Path(__file__).parent.parent / "agents" / "memory" / "PENDING_UPGRADES.md"

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
CANDLES = 500


@st.cache_data(ttl=60)
def load_ohlcv() -> tuple[pd.Series, pd.Series]:
    """Fetch 500 BTC/USDT 1h candles from the Binance public API.

    Cached for 60 seconds. No API key required.

    Returns:
        Tuple of (close, volume) as DatetimeIndex pd.Series.
    """
    df = fetch_ohlcv_bulk(symbol=SYMBOL, timeframe=TIMEFRAME, total_candles=CANDLES)
    ts_index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    close = pd.Series(df["close"].to_numpy(), index=ts_index, dtype=float)
    volume = pd.Series(df["volume"].to_numpy(), index=ts_index, dtype=float)
    return close, volume


def parse_backtest_log() -> pd.DataFrame:
    """Parse agents/memory/backtest_log.md into a DataFrame, newest-first.

    Returns:
        DataFrame with columns matching the log table header.
        Empty DataFrame if file does not exist or has no data rows.
    """
    if not _BACKTEST_LOG.exists():
        return pd.DataFrame()

    lines = [
        ln.strip()
        for ln in _BACKTEST_LOG.read_text().splitlines()
        if ln.strip().startswith("|")
    ]

    if len(lines) < 3:
        return pd.DataFrame()

    def _parse_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    headers = _parse_row(lines[0])
    data_rows = [_parse_row(ln) for ln in lines[2:] if "---" not in ln]

    if not data_rows:
        return pd.DataFrame()

    df = pd.DataFrame(data_rows, columns=headers)
    return df.iloc[::-1].reset_index(drop=True)


def latest_backtest_stats() -> dict:
    """Return the most recent backtest log row as a dict.

    Returns:
        Dict keyed by column names, or empty dict if log is absent/empty.
    """
    df = parse_backtest_log()
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def load_pending_upgrades() -> str:
    """Read PENDING_UPGRADES.md as a string.

    Returns:
        File contents, or an empty-state message if the file is absent or
        contains no entries beyond the header.
    """
    _EMPTY = "No variants have beaten the baseline yet."

    if not _PENDING_UPGRADES.exists():
        return _EMPTY

    text = _PENDING_UPGRADES.read_text().strip()
    non_header = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not non_header:
        return _EMPTY

    return text
