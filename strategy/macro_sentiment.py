"""Macro Swarm — Phase 3 LLM-based news sentiment classifier.

Sends scraped crypto headlines to Claude and parses the structured
JSON response into a MacroSignal dataclass.
"""
import json
import os
from dataclasses import dataclass
from typing import Literal

import anthropic
from dotenv import load_dotenv

load_dotenv()

Signal = Literal["bullish", "bearish", "neutral"]

_VALID_SIGNALS: frozenset[str] = frozenset({"bullish", "bearish", "neutral"})

_SYSTEM_PROMPT = """You are a professional crypto market analyst.
You will be given a list of recent crypto and macro-economic news headlines.
Analyze the overall market sentiment and respond with ONLY a valid JSON object.

Response format (no markdown, no extra text):
{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explanation>"
}

Rules:
- signal must be exactly one of: bullish, bearish, neutral
- confidence must be between 0.0 and 1.0
- reasoning must be a single concise sentence
- Return ONLY the JSON object, nothing else"""


@dataclass
class MacroSignal:
    """Structured output from the LLM sentiment classifier.

    Attributes:
        signal: Directional signal — 'bullish', 'bearish', or 'neutral'.
        confidence: Probability estimate in [0.0, 1.0]. Feeds Phase 4 EV/Kelly math.
        reasoning: Single-sentence LLM explanation, stored in the Phase 6 audit log.
    """

    signal: Signal
    confidence: float
    reasoning: str


def classify_sentiment(
    headlines: list[str],
    model: str = "claude-haiku-4-5-20251001",
) -> MacroSignal:
    """Classify the macro/crypto sentiment of a list of news headlines.

    Sends headlines to Claude and parses the structured JSON response.
    Falls back to neutral (confidence=0.5) on empty input or API/parse errors.

    Args:
        headlines: List of news headline strings from the scraper.
        model: Anthropic model ID to use. Defaults to claude-haiku for speed/cost.

    Returns:
        MacroSignal with signal, confidence, and reasoning.
    """
    if not headlines:
        return MacroSignal(signal="neutral", confidence=0.5, reasoning="No headlines provided.")

    headlines_text = "\n".join(f"- {h}" for h in headlines)
    user_message = f"Analyze the following crypto/macro headlines:\n\n{headlines_text}"

    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_text: str = response.content[0].text.strip()
        parsed: dict = json.loads(raw_text)

        signal_raw = parsed.get("signal", "neutral")
        signal: Signal = signal_raw if signal_raw in _VALID_SIGNALS else "neutral"  # type: ignore[assignment]
        confidence: float = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        reasoning: str = str(parsed.get("reasoning", "LLM classification."))

        return MacroSignal(signal=signal, confidence=confidence, reasoning=reasoning)

    except (json.JSONDecodeError, KeyError, IndexError, Exception):
        return MacroSignal(
            signal="neutral",
            confidence=0.5,
            reasoning="Classification unavailable — defaulting to neutral.",
        )
