"""
API-Football wrapper — fetches fixtures, lineups, standings.
Docs: https://www.api-football.com/documentation-v3
"""
from __future__ import annotations

import datetime as dt
import os
import httpx
import logging
import re
import unicodedata

import pandas as pd

from src.tools.bigquery_tools import run_query, upload_dataframe

_BASE = "https://v3.football.api-sports.io"
_HEADERS = {
    "x-apisports-key": os.environ.get("API_FOOTBALL_KEY", ""),
}

logger = logging.getLogger(__name__)

# World Cup 2026 league ID (confirm once tournament starts)
WC_LEAGUE_ID = 1
BQ_TABLE = "fixtures_historical"
_QUERY_TOKEN_LIMIT = 6

_STOPWORDS = {
    "what", "when", "where", "which", "who", "with", "without", "about", "from", "that",
    "this", "these", "those", "list", "show", "tell", "give", "me", "the", "and", "or",
    "for", "of", "to", "in", "on", "is", "are", "was", "were", "will", "be", "can",
    "could", "please", "result", "results", "match", "matches", "world", "cup", "next",
}


def _get(endpoint: str, params: dict) -> dict:
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{_BASE}/{endpoint}", headers=_HEADERS, params=params)
        resp.raise_for_status()
        return resp.json()


def _extract_season(query: str) -> int | None:
    m = re.search(r"\b(20\d{2})\b", query)
    if not m:
        return None
    season = int(m.group(1))
    return season if 2010 <= season <= 2030 else None


def _query_tokens(query: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", query.lower())
    tokens: list[str] = []
    for word in words:
        if word in _STOPWORDS or len(word) < 4:
            continue
        if word not in tokens:
            tokens.append(word)
        if len(tokens) >= _QUERY_TOKEN_LIMIT:
            break
    return tokens


def _normalise_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _extract_matchup(query: str) -> tuple[str, str] | None:
    # Supports "Team A vs Team B" and "Team A v Team B"
    m = re.search(r"([A-Za-zÀ-ÿ'\- ]+?)\s+v(?:s)?\.?\s+([A-Za-zÀ-ÿ'\- ]+)", query, re.IGNORECASE)
    if not m:
        return None
    team_a = _normalise_text(m.group(1))
    team_b = _normalise_text(m.group(2))
    if not team_a or not team_b:
        return None
    return team_a, team_b


def _is_today_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in ("today", "tonight", "this evening", "this afternoon", "this morning"))


def _row_matches_matchup(row: dict, matchup: tuple[str, str]) -> bool:
    a, b = matchup
    home = _normalise_text(str(row.get("home_team", "")))
    away = _normalise_text(str(row.get("away_team", "")))
    return (a in home and b in away) or (a in away and b in home)


def _table_ref() -> str:
    project = os.environ["BIGQUERY_PROJECT_ID"]
    dataset = os.environ["BIGQUERY_DATASET_ID"]
    return f"`{project}.{dataset}.{BQ_TABLE}`"


def _filter_clauses(tokens: list[str]) -> str:
    if not tokens:
        return "1=1"
    parts = []
    for tok in tokens:
        safe = tok.replace("'", "")
        parts.append(
            f"(LOWER(home_team) LIKE '%{safe}%' OR LOWER(away_team) LIKE '%{safe}%')"
        )
    return " OR ".join(parts)


def _load_fixtures_from_bq(query: str, season: int | None, limit: int) -> list[dict]:
    tokens = _query_tokens(query)
    where_team = _filter_clauses(tokens)
    season_clause = f"AND season = {season}" if season is not None else ""
    sql = f"""
    SELECT fixture_id, season, date, venue, venue_city, referee, status,
           home_team, away_team, home_goals, away_goals
    FROM {_table_ref()}
    WHERE ({where_team})
      {season_clause}
    ORDER BY date DESC
    LIMIT {int(limit)}
    """
    try:
        df = run_query(sql)
    except Exception as exc:  # pragma: no cover - defensive for runtime env issues
        logger.warning("BigQuery lookup failed: %s", exc)
        return []
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _api_fetch_fixtures_for_date(date_value: dt.date) -> list[dict]:
    params = {
        "date": date_value.isoformat(),
        "timezone": "UTC",
    }
    try:
        data = _get("fixtures", params)
    except httpx.HTTPError as exc:
        logger.warning("API-Football request failed in date fixture fetch: %s", exc)
        return []
    response = data.get("response", [])
    return [_normalise_fixture_with_season(fix) for fix in response]


def _normalise_fixture_with_season(fix: dict) -> dict:
    return {
        "fixture_id": fix["fixture"]["id"],
        "season": fix["league"]["season"],
        "date": fix["fixture"]["date"],
        "venue": fix["fixture"]["venue"]["name"],
        "venue_city": fix["fixture"]["venue"]["city"],
        "referee": fix["fixture"].get("referee", "TBD"),
        "status": fix["fixture"]["status"].get("short", "TBD"),
        "home_team": fix["teams"]["home"]["name"],
        "away_team": fix["teams"]["away"]["name"],
        "home_goals": fix["goals"].get("home"),
        "away_goals": fix["goals"].get("away"),
    }


def _store_fixtures_in_bq(rows: list[dict]) -> None:
    if not rows:
        return
    try:
        upload_dataframe(pd.DataFrame(rows), BQ_TABLE)
    except Exception as exc:  # pragma: no cover - defensive for runtime env issues
        logger.warning("BigQuery write-back failed: %s", exc)


def _api_fetch_fixtures(query: str, season: int | None) -> list[dict]:
    params = {"league": WC_LEAGUE_ID}
    if season is not None:
        params["season"] = season
    else:
        params["season"] = dt.datetime.utcnow().year
        params["next"] = 20
    try:
        data = _get("fixtures", params)
    except httpx.HTTPError as exc:
        logger.warning("API-Football request failed in fixture fetch: %s", exc)
        return []
    response = data.get("response", [])
    return [_normalise_fixture_with_season(fix) for fix in response]


def _filter_rows_by_query(
    rows: list[dict],
    query: str,
    limit: int,
    matchup: tuple[str, str] | None = None,
) -> list[dict]:
    if matchup:
        matched = [row for row in rows if _row_matches_matchup(row, matchup)]
        if matched:
            return matched[:limit]

    tokens = _query_tokens(query)
    if not tokens:
        return rows[:limit]

    ranked: list[tuple[int, dict]] = []
    for row in rows:
        home = str(row.get("home_team", "")).lower()
        away = str(row.get("away_team", "")).lower()
        score = sum(1 for tok in tokens if tok in home or tok in away)
        if score > 0:
            ranked.append((score, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[:limit]]


def get_fixtures_cache_first(query: str, limit: int = 5) -> tuple[list[dict], str]:
    """
    Cache-first fixtures lookup:
    1) Read from BigQuery
    2) If cache miss, fetch from API-Football
    3) Persist API result into BigQuery for future requests
    """
    season = _extract_season(query)
    matchup = _extract_matchup(query)

    # Live-date intent (e.g. "today") should query all competitions, not only World Cup.
    if _is_today_query(query):
        today_rows = _api_fetch_fixtures_for_date(dt.datetime.utcnow().date())
        live_filtered = _filter_rows_by_query(today_rows, query=query, limit=limit, matchup=matchup)
        if live_filtered:
            return live_filtered, "api"

    bq_limit = max(limit, 50) if matchup else limit
    cached = _load_fixtures_from_bq(query=query, season=season, limit=bq_limit)
    cached_filtered = _filter_rows_by_query(cached, query=query, limit=limit, matchup=matchup)
    if cached_filtered:
        return cached_filtered, "bigquery"

    api_rows = _api_fetch_fixtures(query=query, season=season)
    if not api_rows:
        return [], "none"

    _store_fixtures_in_bq(api_rows)
    filtered = _filter_rows_by_query(api_rows, query=query, limit=limit, matchup=matchup)
    return filtered, "api"


def get_next_match(query: str) -> dict | None:
    """
    Searches for the next fixture matching the query (team name).
    Returns a normalised dict or None if not found.
    """
    fixtures, _source = get_fixtures_cache_first(query=query, limit=1)
    if not fixtures:
        return None
    row = fixtures[0]
    return {
        "fixture_id": row.get("fixture_id"),
        "date": row.get("date"),
        "venue": row.get("venue"),
        "venue_city": row.get("venue_city"),
        "referee": row.get("referee", "TBD"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "home_lineup": [],
        "away_lineup": [],
        "status": row.get("status", "TBD"),
        "season": row.get("season"),
        "home_goals": row.get("home_goals"),
        "away_goals": row.get("away_goals"),
    }


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
    try:
        data = _get("fixtures/lineups", {"fixture": fixture_id})
    except httpx.HTTPError as exc:
        logger.warning("API-Football request failed in get_lineups for fixture %s: %s", fixture_id, exc)
        return {}
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