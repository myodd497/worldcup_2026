"""
Match Facts Agent — returns lineups, venue, weather, standings and referee
for a given match query.
"""
from __future__ import annotations

from src.tools.api_football import get_next_match
from src.tools.weather import get_venue_weather


def run_structured(query: str) -> dict:
    match = get_next_match(query)
    if not match:
        return {
            "answer": f"Could not find an upcoming match for: {query}",
            "confidence_score": 0.3,
            "confidence_reason": "No matching fixture in available schedule.",
            "metadata": {"has_match": False},
        }

    weather = get_venue_weather(match["venue_city"])

    home = match["home_team"]
    away = match["away_team"]
    lines = [
        f"*{home} vs {away}*",
        f"📅 {match['date']} | 🏟 {match['venue']}, {match['venue_city']}",
        f"🌤 Weather: {weather['description']}, {weather['temp_c']:.0f}°C",
        f"👨‍⚖️ Referee: {match.get('referee', 'TBD')}",
        "",
        f"*{home} lineup:* {', '.join(match.get('home_lineup', ['TBD']))}",
        f"*{away} lineup:* {', '.join(match.get('away_lineup', ['TBD']))}",
    ]
    lineup_known = bool(match.get("home_lineup")) and bool(match.get("away_lineup"))
    score = 0.9 if lineup_known else 0.7
    reason = "Confirmed lineups available." if lineup_known else "Fixture and venue verified, lineups not confirmed yet."

    return {
        "answer": "\n".join(lines),
        "confidence_score": score,
        "confidence_reason": reason,
        "metadata": {"has_match": True, "lineup_known": lineup_known},
    }


def run(query: str) -> str:
    return run_structured(query)["answer"]


if __name__ == "__main__":
    print(run("Portugal"))
