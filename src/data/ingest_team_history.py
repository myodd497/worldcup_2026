"""
Team match history ingestion for all 48 WC 2026 national teams.

Fetches every match result for each team across ALL competitions
(World Cup, qualifiers, UEFA/CONMEBOL/CAF/AFC competitions, friendlies, etc.)
from 2020 to the current season and loads them into BigQuery.

BQ table: team_match_history
  - One row per (team × fixture) pair — team-centric view for easy querying
    e.g. "Portugal's last 10 matches", "Brazil's away record since 2022"
  - Deduplication: skips (team_id, fixture_id) pairs already present
  - WRITE_APPEND: safe to re-run at any time

Usage:
    set -a && source .env && set +a
    poetry run python -m src.data.ingest_team_history
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import httpx
import pandas as pd
from google.cloud import bigquery

from src.tools.bigquery_tools import run_query, upload_dataframe_with_schema, _table_ref

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE = "https://v3.football.api-sports.io"
SEASONS = list(range(2020, 2027))   # 2020 → 2026 inclusive
_FINAL_STATUSES = {"FT", "AET", "PEN", "AWD", "WO"}
_INGESTED_AT = datetime.now(timezone.utc)


def _headers() -> dict[str, str]:
    return {"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")}


def _get(endpoint: str, params: dict) -> dict:
    with httpx.Client(timeout=20) as client:
        resp = client.get(f"{_BASE}/{endpoint}", headers=_headers(), params=params)
        resp.raise_for_status()
        return resp.json()


def _safe_int(val) -> int | None:
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

TEAM_MATCH_HISTORY_SCHEMA: list[bigquery.SchemaField] = [
    # --- Team perspective ---
    bigquery.SchemaField("team_id",              "INT64",  description="API-Football ID of the team this row represents"),
    bigquery.SchemaField("team_name",            "STRING", description="Name of the team this row represents"),
    # --- Fixture identity ---
    bigquery.SchemaField("fixture_id",           "INT64",  description="API-Football fixture ID (unique per match)"),
    bigquery.SchemaField("match_date",           "DATE",   description="Date the match was played (UTC)"),
    bigquery.SchemaField("match_datetime",       "TIMESTAMP", description="Full UTC date-time the match started"),
    bigquery.SchemaField("season",               "INT64",  description="Season year (e.g. 2022 for the 2022/23 season)"),
    bigquery.SchemaField("status",               "STRING", description="Match status short code: FT, AET, PEN, NS, CANC …"),
    # --- Competition ---
    bigquery.SchemaField("competition_id",       "INT64",  description="API-Football league/competition ID"),
    bigquery.SchemaField("competition_name",     "STRING", description="Competition name e.g. 'World Cup', 'UEFA Nations League'"),
    bigquery.SchemaField("competition_country",  "STRING", description="Country or confederation the competition belongs to"),
    bigquery.SchemaField("competition_round",    "STRING", description="Round or matchday e.g. 'Group Stage - 1', 'Final'"),
    # --- Participants ---
    bigquery.SchemaField("home_team_id",         "INT64",  description="API-Football ID of the home team"),
    bigquery.SchemaField("home_team_name",       "STRING", description="Name of the home team"),
    bigquery.SchemaField("away_team_id",         "INT64",  description="API-Football ID of the away team"),
    bigquery.SchemaField("away_team_name",       "STRING", description="Name of the away team"),
    # --- Team-centric perspective ---
    bigquery.SchemaField("was_home",             "BOOL",   description="True if team_id played as the home side"),
    bigquery.SchemaField("opponent_id",          "INT64",  description="API-Football ID of the opposing team"),
    bigquery.SchemaField("opponent_name",        "STRING", description="Name of the opposing team"),
    bigquery.SchemaField("goals_scored",         "INT64",  description="Goals scored by team_id in this match"),
    bigquery.SchemaField("goals_conceded",       "INT64",  description="Goals conceded by team_id in this match"),
    bigquery.SchemaField("result",               "STRING", description="Match result from team_id's perspective: W / D / L (null if not yet played)"),
    # --- Raw scores ---
    bigquery.SchemaField("home_goals",           "INT64",  description="Full-time goals scored by the home team"),
    bigquery.SchemaField("away_goals",           "INT64",  description="Full-time goals scored by the away team"),
    bigquery.SchemaField("home_goals_ht",        "INT64",  description="Half-time goals by home team"),
    bigquery.SchemaField("away_goals_ht",        "INT64",  description="Half-time goals by away team"),
    bigquery.SchemaField("home_goals_et",        "INT64",  description="Extra-time goals by home team (nullable)"),
    bigquery.SchemaField("away_goals_et",        "INT64",  description="Extra-time goals by away team (nullable)"),
    bigquery.SchemaField("home_goals_pen",       "INT64",  description="Penalty shootout goals by home team (nullable)"),
    bigquery.SchemaField("away_goals_pen",       "INT64",  description="Penalty shootout goals by away team (nullable)"),
    # --- Venue ---
    bigquery.SchemaField("venue_name",           "STRING", description="Stadium name"),
    bigquery.SchemaField("venue_city",           "STRING", description="City where the match was played"),
    bigquery.SchemaField("referee",              "STRING", description="Referee name as provided by API"),
    # --- Metadata ---
    bigquery.SchemaField("ingested_at",          "TIMESTAMP", description="UTC timestamp this row was written to BQ"),
    bigquery.SchemaField("data_source",          "STRING",    description="Origin of data: api-football-v3"),
]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _compute_result(goals_scored: int | None, goals_conceded: int | None) -> str | None:
    if goals_scored is None or goals_conceded is None:
        return None
    if goals_scored > goals_conceded:
        return "W"
    if goals_scored == goals_conceded:
        return "D"
    return "L"


def _normalise_fixture(fix: dict, team_id: int, team_name: str) -> dict:
    fixture    = fix.get("fixture") or {}
    league     = fix.get("league")  or {}
    teams      = fix.get("teams")   or {}
    goals      = fix.get("goals")   or {}
    score      = fix.get("score")   or {}
    venue      = fixture.get("venue") or {}

    home = teams.get("home") or {}
    away = teams.get("away") or {}

    home_id = _safe_int(home.get("id"))
    was_home = home_id == team_id

    home_goals_ft = _safe_int(goals.get("home"))
    away_goals_ft = _safe_int(goals.get("away"))

    goals_scored   = home_goals_ft if was_home else away_goals_ft
    goals_conceded = away_goals_ft if was_home else home_goals_ft
    result = _compute_result(goals_scored, goals_conceded)

    opponent_id   = _safe_int(away.get("id"))   if was_home else _safe_int(home.get("id"))
    opponent_name = (away.get("name") or "")    if was_home else (home.get("name") or "")

    raw_date = fixture.get("date", "")
    try:
        dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        match_date     = dt.date()
        match_datetime = dt
    except (ValueError, AttributeError):
        match_date     = None
        match_datetime = None

    ht  = (score.get("halftime")  or {})
    et  = (score.get("extratime") or {})
    pen = (score.get("penalty")   or {})

    return {
        "team_id":             team_id,
        "team_name":           team_name,
        "fixture_id":          _safe_int(fixture.get("id")),
        "match_date":          match_date,
        "match_datetime":      match_datetime,
        "season":              _safe_int(league.get("season")),
        "status":              fixture.get("status", {}).get("short", "") or "",
        "competition_id":      _safe_int(league.get("id")),
        "competition_name":    league.get("name", "") or "",
        "competition_country": league.get("country", "") or "",
        "competition_round":   league.get("round", "") or "",
        "home_team_id":        _safe_int(home.get("id")),
        "home_team_name":      home.get("name", "") or "",
        "away_team_id":        _safe_int(away.get("id")),
        "away_team_name":      away.get("name", "") or "",
        "was_home":            was_home,
        "opponent_id":         opponent_id,
        "opponent_name":       opponent_name,
        "goals_scored":        goals_scored,
        "goals_conceded":      goals_conceded,
        "result":              result,
        "home_goals":          home_goals_ft,
        "away_goals":          away_goals_ft,
        "home_goals_ht":       _safe_int(ht.get("home")),
        "away_goals_ht":       _safe_int(ht.get("away")),
        "home_goals_et":       _safe_int(et.get("home")),
        "away_goals_et":       _safe_int(et.get("away")),
        "home_goals_pen":      _safe_int(pen.get("home")),
        "away_goals_pen":      _safe_int(pen.get("away")),
        "venue_name":          venue.get("name", "") or "",
        "venue_city":          venue.get("city", "") or "",
        "referee":             fixture.get("referee", "") or "",
        "ingested_at":         _INGESTED_AT,
        "data_source":         "api-football-v3",
    }


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _existing_pairs(table: str) -> set[tuple[int, int]]:
    """Returns set of (team_id, fixture_id) already in BQ to prevent re-ingestion."""
    try:
        df = run_query(
            f"SELECT DISTINCT team_id, fixture_id FROM `{_table_ref(table)}`"
        )
        return {(int(r.team_id), int(r.fixture_id)) for r in df.itertuples()}
    except Exception:
        return set()


def _load_teams_from_bq() -> list[tuple[int, str]]:
    """Returns (team_id, team_name) for all 48 WC 2026 teams from BQ."""
    proj = os.environ["BIGQUERY_PROJECT_ID"]
    ds   = os.environ["BIGQUERY_DATASET_ID"]
    df = run_query(
        f"SELECT DISTINCT team_id, team_name FROM `{proj}.{ds}.team_stats` "
        f"WHERE season = 2026 ORDER BY team_name"
    )
    return [(int(r.team_id), r.team_name) for r in df.itertuples()]


def ingest_team_match_history(seasons: list[int] = SEASONS) -> int:
    teams   = _load_teams_from_bq()
    existing = _existing_pairs("team_match_history")
    print(f"  Loaded {len(teams)} teams. Already have {len(existing)} (team, fixture) pairs in BQ.")

    rows: list[dict] = []
    total_api_calls = 0
    skipped_existing = 0

    for team_id, team_name in teams:
        for season in seasons:
            data = _get("fixtures", {"team": team_id, "season": season})
            total_api_calls += 1

            new_for_team = 0
            for fix in data.get("response", []):
                fid = fix.get("fixture", {}).get("id")
                if (team_id, fid) in existing:
                    skipped_existing += 1
                    continue
                rows.append(_normalise_fixture(fix, team_id, team_name))
                existing.add((team_id, fid))  # prevent duplicates within this run
                new_for_team += 1

            if new_for_team > 0:
                print(f"    {team_name} {season}: +{new_for_team} fixtures")

            time.sleep(0.12)  # ~8 req/s — well within API-Football limits

    if not rows:
        print(f"team_match_history: nothing new (skipped {skipped_existing} existing rows).")
        return 0

    df = pd.DataFrame(rows)

    nullable_ints = [
        "team_id", "fixture_id", "season", "competition_id",
        "home_team_id", "away_team_id", "opponent_id",
        "goals_scored", "goals_conceded",
        "home_goals", "away_goals",
        "home_goals_ht", "away_goals_ht",
        "home_goals_et", "away_goals_et",
        "home_goals_pen", "away_goals_pen",
    ]
    for col in nullable_ints:
        df[col] = pd.array(df[col], dtype=pd.Int64Dtype())

    count = upload_dataframe_with_schema(
        df, "team_match_history", TEAM_MATCH_HISTORY_SCHEMA, "WRITE_APPEND"
    )
    print(f"\nteam_match_history: {count} rows appended "
          f"({total_api_calls} API calls, {skipped_existing} existing rows skipped).")
    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_team_history_ingestion() -> None:
    print("=" * 60)
    print("  WC 2026 National Teams — Full Match History")
    print(f"  Teams: all 48 WC 2026 qualifiers")
    print(f"  Seasons: {SEASONS}")
    print(f"  Competitions: all (qualifiers, friendlies, cups …)")
    print("=" * 60)

    n = ingest_team_match_history(SEASONS)

    print()
    print("=" * 60)
    print(f"  Total rows written: {n}")
    print("=" * 60)


if __name__ == "__main__":
    run_team_history_ingestion()
