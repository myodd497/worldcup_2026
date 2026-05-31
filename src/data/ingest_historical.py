"""
Historical data ingestion — fetches past World Cup results and team stats
from API-Football and loads them into BigQuery for use in model training.
"""
from __future__ import annotations

import os
import httpx
import pandas as pd
from src.tools.bigquery_tools import upload_dataframe

_BASE = "https://v3.football.api-sports.io"
_HEADERS = {
    "x-apisports-key": os.environ.get("API_FOOTBALL_KEY", ""),
}

# Past World Cup league IDs: 2018=1, 2022=1 (same league, different seasons)
SEASONS = [2018, 2022,2026]
WC_LEAGUE_ID = 1
BQ_TABLE = "fixtures_historical"


def _get(endpoint: str, params: dict) -> dict:
    with httpx.Client(timeout=15) as client:
        resp = client.get(f"{_BASE}/{endpoint}", headers=_HEADERS, params=params)
        resp.raise_for_status()
        return resp.json()


def _normalise_fixture(fix: dict, season: int) -> dict:
    return {
        "fixture_id": fix["fixture"]["id"],
        "season": season,
        "date": fix["fixture"]["date"],
        "venue": fix["fixture"]["venue"]["name"],
        "venue_city": fix["fixture"]["venue"]["city"],
        "referee": fix["fixture"].get("referee", ""),
        "status": fix["fixture"]["status"]["short"],
        "home_team": fix["teams"]["home"]["name"],
        "away_team": fix["teams"]["away"]["name"],
        "home_goals": fix["goals"].get("home"),
        "away_goals": fix["goals"].get("away"),
    }


def ingest_season(season: int) -> int:
    data = _get("fixtures", {"league": WC_LEAGUE_ID, "season": season})
    fixtures = data.get("response", [])
    if not fixtures:
        return 0
    rows = [_normalise_fixture(f, season) for f in fixtures]
    df = pd.DataFrame(rows)
    upload_dataframe(df, BQ_TABLE)
    return len(rows)


def run_ingestion() -> None:
    total = 0
    for season in SEASONS:
        count = ingest_season(season)
        print(f"Season {season}: {count} fixtures uploaded to BigQuery.")
        total += count
    print(f"Total: {total} fixtures.")


if __name__ == "__main__":
    run_ingestion()
