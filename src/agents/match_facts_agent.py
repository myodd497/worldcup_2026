"""
Match Facts Agent — returns lineups, venue, weather, standings and referee
for a given match query.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.tools.api_football import get_fixtures_cache_first
from src.tools.news_search import search_news
from src.tools.weather import get_venue_weather


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _fallback_answer(query: str, reason: str) -> dict:
    web_hits = search_news(f"{query} football", max_results=3)
    if web_hits:
        lines = ["I couldn't confirm this from my internal fixture data, but I found relevant web sources:"]
        for i, item in enumerate(web_hits, 1):
            lines.append(f"{i}. [{item['title']}]({item['url']})")
        return {
            "answer": "\n".join(lines),
            "confidence_score": 0.45,
            "confidence_reason": f"{reason} Used web search fallback.",
            "metadata": {"has_match": False, "data_source": "web_search", "count": len(web_hits)},
        }

    # Final fallback: still answer conversationally so users never get a dead-end.
    llm_prompt = (
        "You are a football assistant. Give a concise and direct answer to the user question. "
        "If the question is about when the FIFA men's World Cup 2026 starts, answer: June 11, 2026. "
        "If uncertain, state that this is a best-effort answer.\n\n"
        f"User question: {query}"
    )
    llm_answer = _llm.invoke(llm_prompt).content.strip()
    return {
        "answer": llm_answer,
        "confidence_score": 0.35,
        "confidence_reason": f"{reason} Web providers returned no results; used LLM best-effort fallback.",
        "metadata": {"has_match": False, "data_source": "llm_fallback"},
    }


def run_structured(query: str) -> dict:
    fixtures, source = get_fixtures_cache_first(query=query, limit=5)
    if not fixtures:
        return _fallback_answer(
            query=query,
            reason="No matching fixture in BigQuery cache and API returned no matching results.",
        )

    q = query.lower()
    wants_list = any(term in q for term in ("list", "matches", "results", "fixtures", "next", "upcoming", "schedule"))

    if wants_list:
        lines = ["*Matches found:*", f"📦 Source: {source}"]
        for row in fixtures:
            home = row.get("home_team", "Unknown")
            away = row.get("away_team", "Unknown")
            season = row.get("season", "?")
            date = str(row.get("date", ""))[:10]
            hg = row.get("home_goals")
            ag = row.get("away_goals")
            score = "TBD" if hg is None or ag is None else f"{hg}-{ag}"
            lines.append(f"- {date} ({season}) | {home} vs {away} | {score}")

        confidence_score = 0.9 if source == "bigquery" else 0.7
        confidence_reason = (
            "Data served from BigQuery cache."
            if source == "bigquery"
            else "Data fetched from API and stored in BigQuery cache."
        )
        return {
            "answer": "\n".join(lines),
            "confidence_score": confidence_score,
            "confidence_reason": confidence_reason,
            "metadata": {"has_match": True, "data_source": source, "count": len(fixtures)},
        }

    match = fixtures[0]
    weather = get_venue_weather(str(match.get("venue_city", "")))

    home = str(match.get("home_team", "Unknown"))
    away = str(match.get("away_team", "Unknown"))
    lines = [
        f"*{home} vs {away}*",
        f"📦 Source: {source}",
        f"📅 {match.get('date', 'TBD')} | 🏟 {match.get('venue', 'TBD')}, {match.get('venue_city', 'TBD')}",
        f"🌤 Weather: {weather['description']}, {weather['temp_c']:.0f}°C",
        f"👨‍⚖️ Referee: {match.get('referee', 'TBD')}",
        "",
        f"*Result:* {match.get('home_goals', 'TBD')} - {match.get('away_goals', 'TBD')}",
    ]
    score = 0.9 if source == "bigquery" else 0.7
    reason = "Data served from BigQuery cache." if source == "bigquery" else "Data fetched from API and stored in BigQuery cache."

    return {
        "answer": "\n".join(lines),
        "confidence_score": score,
        "confidence_reason": reason,
        "metadata": {"has_match": True, "data_source": source},
    }


def run(query: str) -> str:
    return run_structured(query)["answer"]


if __name__ == "__main__":
    print(run("Portugal"))
