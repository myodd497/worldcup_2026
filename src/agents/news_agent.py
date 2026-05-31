"""
News Agent — fetches recent news about a match, team, or player.
Uses Tavily search + NewsAPI as fallback.
"""
from __future__ import annotations

from src.tools.news_search import search_news


def run_structured(query: str) -> dict:
    articles = search_news(query, max_results=5)
    if not articles:
        return {
            "answer": f"No recent news found for: {query}",
            "confidence_score": 0.35,
            "confidence_reason": "No relevant articles returned from search providers.",
            "metadata": {"article_count": 0},
        }

    lines = [f"*Latest news for: {query}*\n"]
    for i, art in enumerate(articles, 1):
        lines.append(f"{i}. [{art['title']}]({art['url']})\n   _{art['source']}_ — {art['published_at']}")
    return {
        "answer": "\n".join(lines),
        "confidence_score": min(0.95, 0.55 + 0.08 * len(articles)),
        "confidence_reason": "Confidence rises with number of corroborating articles.",
        "metadata": {"article_count": len(articles)},
    }


def run(query: str) -> str:
    return run_structured(query)["answer"]


if __name__ == "__main__":
    print(run("Portugal vs Morocco World Cup 2026"))
