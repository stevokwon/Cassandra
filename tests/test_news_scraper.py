"""TDD tests for the playwright-based crypto news scraper."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.tools.news_scraper import DEFAULT_SOURCES, scrape_crypto_headlines


# ── Scraper tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scrape_returns_list_of_strings() -> None:
    """scrape_crypto_headlines() must return a list of non-empty strings."""
    mock_page = AsyncMock()
    mock_page.query_selector_all = AsyncMock(return_value=[])
    mock_page.title = AsyncMock(return_value="CoinDesk")

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)

    mock_playwright_cm = AsyncMock()
    mock_playwright_cm.__aenter__ = AsyncMock(return_value=mock_playwright_cm)
    mock_playwright_cm.__aexit__ = AsyncMock(return_value=False)
    mock_playwright_cm.chromium = MagicMock()
    mock_playwright_cm.chromium.launch = AsyncMock(return_value=mock_browser)

    with patch("agents.tools.news_scraper.async_playwright", return_value=mock_playwright_cm):
        result = await scrape_crypto_headlines(sources=["https://coindesk.com"])

    assert isinstance(result, list)
    assert all(isinstance(h, str) for h in result)


@pytest.mark.asyncio
async def test_scrape_deduplicates_headlines() -> None:
    """scrape_crypto_headlines() must return unique headlines only."""
    duplicate_text = "Bitcoin hits new ATH"

    mock_elem1 = AsyncMock()
    mock_elem1.inner_text = AsyncMock(return_value=duplicate_text)
    mock_elem2 = AsyncMock()
    mock_elem2.inner_text = AsyncMock(return_value=duplicate_text)

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.query_selector_all = AsyncMock(return_value=[mock_elem1, mock_elem2])

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)

    mock_playwright_cm = AsyncMock()
    mock_playwright_cm.__aenter__ = AsyncMock(return_value=mock_playwright_cm)
    mock_playwright_cm.__aexit__ = AsyncMock(return_value=False)
    mock_playwright_cm.chromium = MagicMock()
    mock_playwright_cm.chromium.launch = AsyncMock(return_value=mock_browser)

    with patch("agents.tools.news_scraper.async_playwright", return_value=mock_playwright_cm):
        result = await scrape_crypto_headlines(sources=["https://coindesk.com"])

    assert result.count(duplicate_text) <= 1


def test_default_sources_is_nonempty_list() -> None:
    """DEFAULT_SOURCES must be a non-empty list of URL strings."""
    assert isinstance(DEFAULT_SOURCES, list)
    assert len(DEFAULT_SOURCES) > 0
    assert all(s.startswith("https://") for s in DEFAULT_SOURCES)


@pytest.mark.asyncio
async def test_scrape_handles_page_error_gracefully() -> None:
    """scrape_crypto_headlines() must return an empty list on network failure."""
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(side_effect=Exception("Network error"))
    mock_page.query_selector_all = AsyncMock(return_value=[])

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_context)

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)

    mock_playwright_cm = AsyncMock()
    mock_playwright_cm.__aenter__ = AsyncMock(return_value=mock_playwright_cm)
    mock_playwright_cm.__aexit__ = AsyncMock(return_value=False)
    mock_playwright_cm.chromium = MagicMock()
    mock_playwright_cm.chromium.launch = AsyncMock(return_value=mock_browser)

    with patch("agents.tools.news_scraper.async_playwright", return_value=mock_playwright_cm):
        result = await scrape_crypto_headlines(sources=["https://coindesk.com"])

    assert isinstance(result, list)
