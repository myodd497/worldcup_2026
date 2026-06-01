"""
Web search utility with provider fallback chain:
1) DuckDuckGo HTML (free, no key)
2) Serper (Google API)
3) Tavily
4) NewsAPI
5) Google News RSS (free, no key)
"""
from __future__ import annotations

import html
import os
import re
from urllib.parse import quote_plus
from urllib.parse import parse_qs
from urllib.parse import unquote
from urllib.parse import urlparse
from typing import Any
import xml.etree.ElementTree as ET

import httpx


def _is_news_enabled() -> bool:
    # Keep search on by default for robust fallback behavior.
    # Even if ENABLE_NEWS is disabled, we keep free RSS fallback available
    # so the assistant can still produce a web-backed answer.
    flag = os.getenv("ENABLE_NEWS", "true").lower()
    if flag in {"false", "0", "no", "off"}:
        return True
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_newsapi = bool(os.getenv("NEWSAPI_KEY"))
    has_serper = bool(os.getenv("SERPER_API_KEY"))
    return (flag == "true") or has_tavily or has_newsapi or has_serper


def search_news(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Returns a list of web/article dicts: title, url, source, published_at, snippet."""
    if not _is_news_enabled():
        return []

    has_serper = bool(os.getenv("SERPER_API_KEY"))
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_newsapi = bool(os.getenv("NEWSAPI_KEY"))

    try:
        results = _duckduckgo_html_search(query, max_results)
        if results:
            return results
    except Exception:
        pass

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


def _duckduckgo_html_search(query: str, max_results: int) -> list[dict]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    with httpx.Client(timeout=12, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        html_text = resp.text

    results: list[dict] = []
    seen_urls: set[str] = set()
    anchor_matches = list(
        re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>[\s\S]*?)</a>',
            html_text,
            flags=re.IGNORECASE,
        )
    )

    for idx, match in enumerate(anchor_matches):
        raw_title = match.group("title")
        raw_href = match.group("href").strip()
        start = match.end()
        end = anchor_matches[idx + 1].start() if idx + 1 < len(anchor_matches) else min(len(html_text), start + 5000)
        segment = html_text[start:end]

        snippet_match = re.search(
            r'class="result__snippet"[^>]*>([\s\S]*?)</a>|class="result__snippet"[^>]*>([\s\S]*?)</div>',
            segment,
            flags=re.IGNORECASE,
        )

        if not raw_title or not raw_href:
            continue

        raw_snippet = ""
        if snippet_match:
            raw_snippet = (snippet_match.group(1) or snippet_match.group(2) or "").strip()

        title = re.sub(r"<[^>]+>", " ", raw_title)
        title = html.unescape(title)
        title = re.sub(r"\s+", " ", title).strip()

        snippet = re.sub(r"<[^>]+>", " ", raw_snippet)
        snippet = html.unescape(snippet)
        snippet = re.sub(r"\s+", " ", snippet).strip()

        if raw_href.startswith("//"):
            raw_href = "https:" + raw_href

        parsed = urlparse(raw_href)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            qs = parse_qs(parsed.query)
            decoded_url = unquote((qs.get("uddg") or [""])[0]).strip()
            final_url = decoded_url or raw_href
        else:
            final_url = raw_href

        if not final_url or final_url in seen_urls:
            continue
        seen_urls.add(final_url)

        source = urlparse(final_url).netloc or "DuckDuckGo"
        source = source.replace("www.", "")

        results.append(
            {
                "title": title,
                "url": final_url,
                "source": source,
                "published_at": "",
                "snippet": snippet,
            }
        )
        if len(results) >= max(1, int(max_results)):
            break

    return results


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
            "snippet": item.get("snippet", ""),
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
            "snippet": r.get("content", "") or r.get("snippet", ""),
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
            "snippet": art.get("description", "") or art.get("content", ""),
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
        description = (item.findtext("description") or "").strip()
        if not title or not link:
            continue
        results.append(
            {
                "title": title,
                "url": link,
                "source": source,
                "published_at": published,
                "snippet": description,
            }
        )
    return results


if __name__ == "__main__":
    results = search_news("Portugal World Cup 2026")
    for r in results:
        print(r)
