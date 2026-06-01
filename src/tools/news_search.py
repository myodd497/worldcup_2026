"""
Web search utility with provider fallback chain:
1) Serper (Google API)
2) Tavily
3) NewsAPI
4) Google News RSS (free, no key)
"""
from __future__ import annotations

import os
from urllib.parse import quote_plus
from typing import Any
import xml.etree.ElementTree as ET

import httpx


def _is_news_enabled() -> bool:
    # Keep search on by default for robust fallback behavior.
    flag = os.getenv("ENABLE_NEWS", "true").lower()
    if flag in {"false", "0", "no", "off"}:
        return False
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_newsapi = bool(os.getenv("NEWSAPI_KEY"))
    has_serper = bool(os.getenv("SERPER_API_KEY"))
    return (flag == "true") or has_tavily or has_newsapi or has_serper


def search_news(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Returns a list of web/article dicts: title, url, source, published_at."""
    if not _is_news_enabled():
        return []

    has_serper = bool(os.getenv("SERPER_API_KEY"))
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_newsapi = bool(os.getenv("NEWSAPI_KEY"))

    if has_serper:
        try:
            results = _serper_search(query, max_results)
            if results:
                return results
        except Exception:
            pass

    if has_tavily:
        try:
            results = _tavily_search(query, max_results)
            if results:
                return results
        except Exception:
            pass

    if has_newsapi:
        try:
            results = _newsapi_search(query, max_results)
            if results:
                return results
        except Exception:
            pass

    try:
        return _google_news_rss_search(query, max_results)
    except Exception:
        return []


def _serper_search(query: str, max_results: int) -> list[dict]:
    api_key = os.environ["SERPER_API_KEY"]
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": max(1, min(int(max_results), 10)),
        "gl": "us",
        "hl": "en",
    }
    with httpx.Client(timeout=12) as client:
        resp = client.post("https://google.serper.dev/search", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    articles = []
    for item in data.get("organic", []):
        articles.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "source": item.get("source", "Google"),
            "published_at": item.get("date", ""),
        })
    return articles


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


def _google_news_rss_search(query: str, max_results: int) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    with httpx.Client(timeout=12, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        xml_text = resp.text

    root = ET.fromstring(xml_text)
    items = root.findall("./channel/item")

    results: list[dict] = []
    for item in items[: max(1, int(max_results))]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source = (item.findtext("source") or "Google News").strip()
        published = (item.findtext("pubDate") or "").strip()
        if not title or not link:
            continue
        results.append(
            {
                "title": title,
                "url": link,
                "source": source,
                "published_at": published,
            }
        )
    return results


if __name__ == "__main__":
    results = search_news("Portugal World Cup 2026")
    for r in results:
        print(r)
