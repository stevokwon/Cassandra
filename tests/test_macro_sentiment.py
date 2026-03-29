"""TDD tests for the Anthropic LLM macro sentiment classifier."""
import json
from unittest.mock import MagicMock, patch

import pytest

from strategy.macro_sentiment import MacroSignal, classify_sentiment


# ── Fixtures ──────────────────────────────────────────────────────────────────

BULLISH_HEADLINES = [
    "Bitcoin surges past $70,000 as institutional buying accelerates",
    "ETF inflows hit record $1B single-day high",
    "Fed signals rate cuts ahead, risk assets rally",
]

BEARISH_HEADLINES = [
    "Bitcoin crashes 15% amid regulatory crackdown fears",
    "Major exchange halts withdrawals citing liquidity issues",
    "SEC files lawsuit against top crypto firms",
]

NEUTRAL_HEADLINES = [
    "Bitcoin trades sideways as market awaits CPI data",
    "Ethereum developers announce routine upgrade timeline",
]


def _make_mock_response(signal: str, confidence: float, reasoning: str) -> MagicMock:
    """Build a mock Anthropic API response with JSON content."""
    payload = json.dumps({
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
    })
    mock_content = MagicMock()
    mock_content.text = payload
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    return mock_response


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_classify_sentiment_returns_macro_signal() -> None:
    """classify_sentiment() must return a MacroSignal dataclass."""
    mock_response = _make_mock_response("bullish", 0.78, "Strong institutional inflow.")
    with patch("strategy.macro_sentiment.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_response
        result = classify_sentiment(BULLISH_HEADLINES)
    assert isinstance(result, MacroSignal)


def test_classify_sentiment_bullish_headlines() -> None:
    """Bullish headlines must produce a MacroSignal with signal='bullish'."""
    mock_response = _make_mock_response("bullish", 0.82, "Institutional demand.")
    with patch("strategy.macro_sentiment.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_response
        result = classify_sentiment(BULLISH_HEADLINES)
    assert result.signal == "bullish"
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reasoning, str) and len(result.reasoning) > 0


def test_classify_sentiment_bearish_headlines() -> None:
    """Bearish headlines must produce a MacroSignal with signal='bearish'."""
    mock_response = _make_mock_response("bearish", 0.75, "Regulatory risk elevated.")
    with patch("strategy.macro_sentiment.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_response
        result = classify_sentiment(BEARISH_HEADLINES)
    assert result.signal == "bearish"


def test_classify_sentiment_confidence_is_bounded() -> None:
    """MacroSignal.confidence must always be in [0.0, 1.0]."""
    mock_response = _make_mock_response("neutral", 0.5, "Mixed signals.")
    with patch("strategy.macro_sentiment.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_response
        result = classify_sentiment(NEUTRAL_HEADLINES)
    assert 0.0 <= result.confidence <= 1.0


def test_classify_sentiment_empty_headlines_returns_neutral() -> None:
    """No headlines → MacroSignal with signal='neutral' and low confidence."""
    result = classify_sentiment([])
    assert result.signal == "neutral"
    assert result.confidence == 0.5


def test_classify_sentiment_invalid_llm_json_returns_neutral() -> None:
    """Malformed LLM JSON response must fall back to neutral signal."""
    mock_content = MagicMock()
    mock_content.text = "This is not valid JSON"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    with patch("strategy.macro_sentiment.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_response
        result = classify_sentiment(BULLISH_HEADLINES)
    assert result.signal == "neutral"
    assert result.confidence == 0.5
