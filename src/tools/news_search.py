"""
News search — Tavily (primary) with NewsAPI fallback.
"""
from __future__ import annotations

import os
from typing import Any


def _is_news_enabled() -> bool:
    flag = os.getenv("ENABLE_NEWS", "false").lower() == "true"
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_newsapi = bool(os.getenv("NEWSAPI_KEY"))
    return flag or has_tavily or has_newsapi


def search_news(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Returns a list of article dicts: title, url, source, published_at."""
    if not _is_news_enabled():
        return []

    try:
        return _tavily_search(query, max_results)
    except Exception:
        return _newsapi_search(query, max_results)


def _tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=max_results, search_depth="advanced")
    articles = []
    for r in response.get("results", []):
        articles.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "source": r.get("source", ""),
            "published_at": r.get("published_date", ""),
        })
    return articles


def _newsapi_search(query: str, max_results: int) -> list[dict]:
    from newsapi import NewsApiClient
    client = NewsApiClient(api_key=os.environ["NEWSAPI_KEY"])
    response = client.get_everything(q=query, sort_by="publishedAt", page_size=max_results, language="en")
    articles = []
    for art in response.get("articles", []):
        articles.append({
            "title": art.get("title", ""),
            "url": art.get("url", ""),
            "source": art.get("source", {}).get("name", ""),
            "published_at": art.get("publishedAt", ""),
        })
    return articles


if __name__ == "__main__":
    results = search_news("Portugal World Cup 2026")
    for r in results:
        print(r)
