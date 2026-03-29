"""Playwright-based async crypto news headline scraper.

Fetches raw headline text from crypto news sources via headless Chromium.
All network I/O is isolated here; strategy/macro_sentiment.py does classification.
"""
import asyncio
from typing import Any

from playwright.async_api import async_playwright

DEFAULT_SOURCES: list[str] = [
    "https://coindesk.com",
    "https://cointelegraph.com",
    "https://cryptopanic.com",
]

# CSS selectors that typically contain article headlines across news sites
_HEADLINE_SELECTORS: list[str] = [
    "h1",
    "h2",
    "h3",
    "article h1",
    "article h2",
    "[data-module='headline']",
    ".headline",
    ".article-title",
]


async def _scrape_page_headlines(page: Any, url: str) -> list[str]:
    """Navigate to a URL and extract all headline text from the DOM.

    Args:
        page: A playwright Page object.
        url: The URL to scrape.

    Returns:
        List of headline strings found on the page. Empty list on error.
    """
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        headlines: list[str] = []
        for selector in _HEADLINE_SELECTORS:
            elements = await page.query_selector_all(selector)
            for elem in elements:
                text: str = await elem.inner_text()
                text = text.strip()
                if text and len(text) > 10:
                    headlines.append(text)
        return headlines
    except Exception:
        return []


async def scrape_crypto_headlines(
    sources: list[str] | None = None,
    max_headlines: int = 50,
) -> list[str]:
    """Scrape crypto news headlines from multiple sources via headless Chromium.

    Args:
        sources: List of URLs to scrape. Defaults to DEFAULT_SOURCES.
        max_headlines: Maximum number of unique headlines to return.

    Returns:
        Deduplicated list of headline strings, up to max_headlines items.
    """
    target_sources = sources if sources is not None else DEFAULT_SOURCES
    all_headlines: list[str] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        for url in target_sources:
            page_headlines = await _scrape_page_headlines(page, url)
            all_headlines.extend(page_headlines)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for h in all_headlines:
        if h not in seen:
            seen.add(h)
            unique.append(h)

    return unique[:max_headlines]


if __name__ == "__main__":
    headlines = asyncio.run(scrape_crypto_headlines())
    for i, h in enumerate(headlines, 1):
        print(f"{i:2d}. {h}")
