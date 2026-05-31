"""
API-Football wrapper — fetches fixtures, lineups, standings.
Docs: https://www.api-football.com/documentation-v3
"""
from __future__ import annotations

import os
import httpx

_BASE = "https://api-football-v1.p.rapidapi.com/v3"
_HEADERS = {
    "X-RapidAPI-Key": os.environ.get("API_FOOTBALL_KEY", ""),
    "X-RapidAPI-Host": os.environ.get("API_FOOTBALL_HOST", "api-football-v1.p.rapidapi.com"),
}

# World Cup 2026 league ID (confirm once tournament starts)
WC_LEAGUE_ID = 1


def _get(endpoint: str, params: dict) -> dict:
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{_BASE}/{endpoint}", headers=_HEADERS, params=params)
        resp.raise_for_status()
        return resp.json()


def get_next_match(query: str) -> dict | None:
    """
    Searches for the next fixture matching the query (team name).
    Returns a normalised dict or None if not found.
    """
    data = _get("fixtures", {"league": WC_LEAGUE_ID, "season": 2026, "next": 10})
    fixtures = data.get("response", [])

    query_lower = query.lower()
    for fix in fixtures:
        home = fix["teams"]["home"]["name"]
        away = fix["teams"]["away"]["name"]
        if query_lower in home.lower() or query_lower in away.lower():
            return _normalise_fixture(fix)
    return None


def _normalise_fixture(fix: dict) -> dict:
    return {
        "fixture_id": fix["fixture"]["id"],
        "date": fix["fixture"]["date"],
        "venue": fix["fixture"]["venue"]["name"],
        "venue_city": fix["fixture"]["venue"]["city"],
        "referee": fix["fixture"].get("referee", "TBD"),
        "home_team": fix["teams"]["home"]["name"],
        "away_team": fix["teams"]["away"]["name"],
        "home_lineup": [],   # populated separately via /lineups endpoint
        "away_lineup": [],
    }


def get_lineups(fixture_id: int) -> dict:
    data = _get("fixtures/lineups", {"fixture": fixture_id})
    lineups = data.get("response", [])
    result = {}
    for team in lineups:
        name = team["team"]["name"]
        starters = [p["player"]["name"] for p in team.get("startXI", [])]
        result[name] = starters
    return result


if __name__ == "__main__":
    match = get_next_match("Portugal")
    print(match)
