"""WebSocket OHLCV stream listener — stub for Phase 1.

Full async implementation deferred to Phase 2 Statistical Swarm.
"""
from collections.abc import Callable
from typing import Any


def start_stream(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    on_candle: Callable[[dict[str, Any]], None],
) -> None:
    """Start a WebSocket candle stream. Phase 1 stub — raises NotImplementedError.

    Args:
        exchange_id: ccxt exchange id string, e.g. 'binance'.
        symbol: Market symbol, e.g. 'BTC/USDT'.
        timeframe: Candle timeframe, e.g. '1h'.
        on_candle: Callback invoked with each new candle dict.

    Raises:
        NotImplementedError: Until Phase 2 implementation.
    """
    raise NotImplementedError("WebSocket stream is implemented in Phase 2.")
