"""
Match Facts Agent — returns lineups, venue, weather, standings and referee
for a given match query.
"""
from __future__ import annotations

from src.tools.api_football import get_fixtures_cache_first
from src.tools.weather import get_venue_weather


def run_structured(query: str) -> dict:
    fixtures, source = get_fixtures_cache_first(query=query, limit=5)
    if not fixtures:
        return {
            "answer": f"Could not find match data for: {query}",
            "confidence_score": 0.3,
            "confidence_reason": "No matching fixture in BigQuery cache and API returned no matching results.",
            "metadata": {"has_match": False, "data_source": "none"},
        }

    wants_list = "list" in query.lower() or "matches" in query.lower() or "results" in query.lower()

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
