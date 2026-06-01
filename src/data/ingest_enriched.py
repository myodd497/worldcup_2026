"""
Priority-1 enrichment ingestion for World Cup 2026 platform.

Ingests three high-value data sets from API-Football v3 into BigQuery:

  standings       — group-stage standings (W/D/L/GF/GA/PTS per team per season)
                    WRITE_TRUNCATE: always replaced with latest snapshot.

  fixture_events  — goals, cards, substitutions, VAR decisions per fixture
                    WRITE_APPEND with dedup: skips fixture_ids already present.

  team_stats      — season-level aggregate stats per team (form, goals, cards …)
                    WRITE_TRUNCATE: always replaced with latest snapshot.

All tables include the metadata columns:
  ingested_at   TIMESTAMP  — UTC time this row was written
  data_source   STRING     — "api-football-v3"
  league_id     INT64      — 1 (World Cup)
  season        INT64      — 2018 / 2022 / 2026

Usage:
    set -a && source .env && set +a
    poetry run python -m src.data.ingest_enriched
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import httpx
import pandas as pd
from google.cloud import bigquery

from src.tools.bigquery_tools import run_query, upload_dataframe_with_schema, _table_ref
from src.tools.api_usage_tracker import record_api_call

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_BASE = "https://v3.football.api-sports.io"
WC_LEAGUE_ID = 1
SEASONS = [2018, 2022, 2026]
_INGESTED_AT = datetime.now(timezone.utc)


def _headers() -> dict[str, str]:
    return {"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")}


def _get(endpoint: str, params: dict) -> dict:
    with httpx.Client(timeout=20) as client:
        resp = client.get(f"{_BASE}/{endpoint}", headers=_headers(), params=params)
        resp.raise_for_status()
        record_api_call(endpoint=endpoint, response_headers=dict(resp.headers))
        return resp.json()


def _safe_int(val) -> int | None:
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# TABLE 1: standings
# ---------------------------------------------------------------------------

STANDINGS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("season",      "INT64",  description="World Cup season year"),
    bigquery.SchemaField("league_id",   "INT64",  description="API-Football league ID (1 = World Cup)"),
    bigquery.SchemaField("group_name",  "STRING", description="Group label e.g. 'Group A'"),
    bigquery.SchemaField("rank",        "INT64",  description="Position within the group (1 = top)"),
    bigquery.SchemaField("team_id",     "INT64",  description="API-Football team ID"),
    bigquery.SchemaField("team_name",   "STRING", description="Full team name"),
    bigquery.SchemaField("points",      "INT64",  description="Points accumulated in the group stage"),
    bigquery.SchemaField("played",      "INT64",  description="Matches played"),
    bigquery.SchemaField("wins",        "INT64",  description="Matches won"),
    bigquery.SchemaField("draws",       "INT64",  description="Matches drawn"),
    bigquery.SchemaField("losses",      "INT64",  description="Matches lost"),
    bigquery.SchemaField("goals_for",   "INT64",  description="Goals scored"),
    bigquery.SchemaField("goals_against","INT64", description="Goals conceded"),
    bigquery.SchemaField("goal_diff",   "INT64",  description="Goal difference (goals_for - goals_against)"),
    bigquery.SchemaField("form",        "STRING", description="Last-N results string e.g. 'WWDLW' (null until matches played)"),
    bigquery.SchemaField("status",      "STRING", description="Promotion/same/relegation status from API"),
    bigquery.SchemaField("description", "STRING", description="Human-readable status e.g. 'Playoffs'"),
    bigquery.SchemaField("snapshot_date","DATE",  description="Date this standings snapshot was captured"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", description="UTC timestamp this row was written to BQ"),
    bigquery.SchemaField("data_source", "STRING", description="Origin of data: api-football-v3"),
]


def _normalise_standing(entry: dict, group: str, season: int) -> dict:
    all_s = entry.get("all", {})
    goals = all_s.get("goals", {})
    return {
        "season":       season,
        "league_id":    WC_LEAGUE_ID,
        "group_name":   group,
        "rank":         _safe_int(entry.get("rank")),
        "team_id":      _safe_int((entry.get("team") or {}).get("id")),
        "team_name":    (entry.get("team") or {}).get("name", "") or "",
        "points":       _safe_int(entry.get("points")),
        "played":       _safe_int(all_s.get("played")),
        "wins":         _safe_int(all_s.get("win")),
        "draws":        _safe_int(all_s.get("draw")),
        "losses":       _safe_int(all_s.get("lose")),
        "goals_for":    _safe_int(goals.get("for")),
        "goals_against":_safe_int(goals.get("against")),
        "goal_diff":    _safe_int(entry.get("goalsDiff")),
        "form":         entry.get("form") or "",
        "status":       entry.get("status") or "",
        "description":  entry.get("description") or "",
        "snapshot_date": _INGESTED_AT.date(),
        "ingested_at":  _INGESTED_AT,
        "data_source":  "api-football-v3",
    }


def ingest_standings(seasons: list[int] = SEASONS) -> int:
    rows: list[dict] = []
    for season in seasons:
        data = _get("standings", {"league": WC_LEAGUE_ID, "season": season})
        for item in data.get("response", []):
            for group in item.get("league", {}).get("standings", []):
                for entry in group:
                    rows.append(_normalise_standing(entry, entry.get("group", "Unknown"), season))

    if not rows:
        print("standings: no data returned from API.")
        return 0

    df = pd.DataFrame(rows)
    nullable_ints = ["rank", "team_id", "points", "played", "wins", "draws",
                     "losses", "goals_for", "goals_against", "goal_diff"]
    for col in nullable_ints:
        df[col] = pd.array(df[col], dtype=pd.Int64Dtype())

    count = upload_dataframe_with_schema(df, "standings", STANDINGS_SCHEMA, "WRITE_TRUNCATE")
    print(f"standings: {count} rows written (WRITE_TRUNCATE) — seasons {seasons}")
    return count


# ---------------------------------------------------------------------------
# TABLE 2: fixture_events
# ---------------------------------------------------------------------------

FIXTURE_EVENTS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("fixture_id",     "INT64",  description="API-Football fixture ID"),
    bigquery.SchemaField("season",         "INT64",  description="World Cup season year"),
    bigquery.SchemaField("league_id",      "INT64",  description="API-Football league ID (1 = World Cup)"),
    bigquery.SchemaField("time_elapsed",   "INT64",  description="Minute when the event occurred"),
    bigquery.SchemaField("time_extra",     "INT64",  description="Extra/stoppage-time minutes (nullable)"),
    bigquery.SchemaField("team_id",        "INT64",  description="ID of the team that triggered the event"),
    bigquery.SchemaField("team_name",      "STRING", description="Name of the team that triggered the event"),
    bigquery.SchemaField("player_id",      "INT64",  description="Primary player involved (goal scorer, carded player, subbed-off player)"),
    bigquery.SchemaField("player_name",    "STRING", description="Name of primary player"),
    bigquery.SchemaField("assist_id",      "INT64",  description="Assist provider or subbed-in player ID (nullable)"),
    bigquery.SchemaField("assist_name",    "STRING", description="Assist provider or subbed-in player name"),
    bigquery.SchemaField("event_type",     "STRING", description="Category: Goal | Card | subst | Var"),
    bigquery.SchemaField("event_detail",   "STRING", description="Sub-type: Normal Goal | Own Goal | Penalty | Yellow Card | Red Card | Substitution 1 | Goal cancelled …"),
    bigquery.SchemaField("event_comments", "STRING", description="Optional free-text commentary from API"),
    bigquery.SchemaField("ingested_at",    "TIMESTAMP", description="UTC timestamp this row was written to BQ"),
    bigquery.SchemaField("data_source",    "STRING", description="Origin of data: api-football-v3"),
]


def _normalise_event(evt: dict, fixture_id: int, season: int) -> dict:
    time   = evt.get("time")   or {}
    team   = evt.get("team")   or {}
    player = evt.get("player") or {}
    assist = evt.get("assist") or {}
    return {
        "fixture_id":     fixture_id,
        "season":         season,
        "league_id":      WC_LEAGUE_ID,
        "time_elapsed":   _safe_int(time.get("elapsed")),
        "time_extra":     _safe_int(time.get("extra")),
        "team_id":        _safe_int(team.get("id")),
        "team_name":      team.get("name") or "",
        "player_id":      _safe_int(player.get("id")),
        "player_name":    player.get("name") or "",
        "assist_id":      _safe_int(assist.get("id")),
        "assist_name":    assist.get("name") or "",
        "event_type":     evt.get("type")    or "",
        "event_detail":   evt.get("detail")  or "",
        "event_comments": evt.get("comments") or "",
        "ingested_at":    _INGESTED_AT,
        "data_source":    "api-football-v3",
    }


def _existing_fixture_ids(table: str) -> set[int]:
    """Returns set of fixture_ids already loaded in BQ to prevent duplicate rows."""
    try:
        df = run_query(f"SELECT DISTINCT fixture_id FROM `{_table_ref(table)}`")
        return set(df["fixture_id"].tolist())
    except Exception:
        return set()


# Statuses that mean the match is fully completed (no more events expected)
_FINAL_STATUSES = {"FT", "AET", "PEN", "AWD", "WO"}


def ingest_fixture_events(seasons: list[int] = SEASONS) -> int:
    existing = _existing_fixture_ids("fixture_events")
    rows: list[dict] = []
    skipped = 0
    fetched = 0

    for season in seasons:
        fixtures_resp = _get("fixtures", {"league": WC_LEAGUE_ID, "season": season})
        for fix in fixtures_resp.get("response", []):
            fid    = fix["fixture"]["id"]
            status = fix["fixture"]["status"]["short"]

            if fid in existing:
                skipped += 1
                continue
            if status not in _FINAL_STATUSES:
                continue  # match not yet played — skip

            events_resp = _get("fixtures/events", {"fixture": fid})
            fetched += 1
            for evt in events_resp.get("response", []):
                rows.append(_normalise_event(evt, fid, season))
            time.sleep(0.1)  # respect API rate limit

    if not rows:
        print(f"fixture_events: nothing new to ingest (skipped {skipped} already-loaded fixtures).")
        return 0

    df = pd.DataFrame(rows)
    nullable_ints = ["fixture_id", "season", "league_id", "time_elapsed", "time_extra",
                     "team_id", "player_id", "assist_id"]
    for col in nullable_ints:
        df[col] = pd.array(df[col], dtype=pd.Int64Dtype())

    count = upload_dataframe_with_schema(df, "fixture_events", FIXTURE_EVENTS_SCHEMA, "WRITE_APPEND")
    print(f"fixture_events: {count} rows appended from {fetched} fixtures "
          f"(skipped {skipped} already-loaded).")
    return count


# ---------------------------------------------------------------------------
# TABLE 3: team_stats
# ---------------------------------------------------------------------------

TEAM_STATS_SCHEMA: list[bigquery.SchemaField] = [
    # --- Identity ---
    bigquery.SchemaField("team_id",                  "INT64",  description="API-Football team ID"),
    bigquery.SchemaField("team_name",                "STRING", description="Full team name"),
    bigquery.SchemaField("season",                   "INT64",  description="World Cup season year"),
    bigquery.SchemaField("league_id",                "INT64",  description="API-Football league ID (1 = World Cup)"),
    # --- Recent form ---
    bigquery.SchemaField("form",                     "STRING", description="Latest N-match result string e.g. WWDLW (null until matches played)"),
    # --- Matches played ---
    bigquery.SchemaField("played_home",              "INT64",  description="Home matches played in this tournament"),
    bigquery.SchemaField("played_away",              "INT64",  description="Away matches played in this tournament"),
    bigquery.SchemaField("played_total",             "INT64",  description="Total matches played"),
    # --- Results ---
    bigquery.SchemaField("wins_home",                "INT64"),
    bigquery.SchemaField("wins_away",                "INT64"),
    bigquery.SchemaField("wins_total",               "INT64"),
    bigquery.SchemaField("draws_home",               "INT64"),
    bigquery.SchemaField("draws_away",               "INT64"),
    bigquery.SchemaField("draws_total",              "INT64"),
    bigquery.SchemaField("losses_home",              "INT64"),
    bigquery.SchemaField("losses_away",              "INT64"),
    bigquery.SchemaField("losses_total",             "INT64"),
    # --- Goals scored ---
    bigquery.SchemaField("goals_for_home",           "INT64",   description="Goals scored in home matches"),
    bigquery.SchemaField("goals_for_away",           "INT64",   description="Goals scored in away matches"),
    bigquery.SchemaField("goals_for_total",          "INT64",   description="Total goals scored"),
    bigquery.SchemaField("goals_for_avg_home",       "FLOAT64", description="Average goals scored per home match"),
    bigquery.SchemaField("goals_for_avg_away",       "FLOAT64", description="Average goals scored per away match"),
    bigquery.SchemaField("goals_for_avg_total",      "FLOAT64", description="Average goals scored per match"),
    # --- Goals conceded ---
    bigquery.SchemaField("goals_against_home",       "INT64",   description="Goals conceded in home matches"),
    bigquery.SchemaField("goals_against_away",       "INT64",   description="Goals conceded in away matches"),
    bigquery.SchemaField("goals_against_total",      "INT64",   description="Total goals conceded"),
    bigquery.SchemaField("goals_against_avg_home",   "FLOAT64", description="Average goals conceded per home match"),
    bigquery.SchemaField("goals_against_avg_away",   "FLOAT64", description="Average goals conceded per away match"),
    bigquery.SchemaField("goals_against_avg_total",  "FLOAT64", description="Average goals conceded per match"),
    # --- Defensive record ---
    bigquery.SchemaField("clean_sheets_home",        "INT64",   description="Home matches with no goals conceded"),
    bigquery.SchemaField("clean_sheets_away",        "INT64",   description="Away matches with no goals conceded"),
    bigquery.SchemaField("clean_sheets_total",       "INT64",   description="Total clean sheets"),
    bigquery.SchemaField("failed_to_score_home",     "INT64",   description="Home matches with zero goals scored"),
    bigquery.SchemaField("failed_to_score_away",     "INT64",   description="Away matches with zero goals scored"),
    bigquery.SchemaField("failed_to_score_total",    "INT64",   description="Total matches with zero goals scored"),
    # --- Penalties ---
    bigquery.SchemaField("penalty_scored_total",     "INT64",   description="Penalties scored across the tournament"),
    bigquery.SchemaField("penalty_missed_total",     "INT64",   description="Penalties missed across the tournament"),
    bigquery.SchemaField("penalty_total",            "INT64",   description="Total penalties taken"),
    # --- Discipline ---
    bigquery.SchemaField("yellow_cards_total",       "INT64",   description="Sum of yellow cards across all time buckets"),
    bigquery.SchemaField("red_cards_total",          "INT64",   description="Sum of red cards across all time buckets"),
    # --- Metadata ---
    bigquery.SchemaField("ingested_at",              "TIMESTAMP", description="UTC timestamp this row was written to BQ"),
    bigquery.SchemaField("data_source",              "STRING",    description="Origin of data: api-football-v3"),
]


def _sum_card_buckets(card_dict: dict | None) -> int:
    """Cards are reported in time buckets (0-15, 16-30 …). Sum all non-null totals."""
    if not card_dict:
        return 0
    return sum(
        _safe_int(v.get("total")) or 0
        for v in card_dict.values()
        if isinstance(v, dict)
    )


def _normalise_team_stats(resp: dict, team_id: int, team_name: str, season: int) -> dict:
    fix   = resp.get("fixtures") or {}
    goals = resp.get("goals")    or {}
    gf    = goals.get("for")     or {}
    ga    = goals.get("against") or {}
    cs    = resp.get("clean_sheet")    or {}
    fts   = resp.get("failed_to_score") or {}
    pen   = resp.get("penalty")         or {}
    cards = resp.get("cards")           or {}

    return {
        "team_id":               team_id,
        "team_name":             team_name,
        "season":                season,
        "league_id":             WC_LEAGUE_ID,
        "form":                  resp.get("form") or "",
        # played
        "played_home":           _safe_int((fix.get("played") or {}).get("home")),
        "played_away":           _safe_int((fix.get("played") or {}).get("away")),
        "played_total":          _safe_int((fix.get("played") or {}).get("total")),
        # wins
        "wins_home":             _safe_int((fix.get("wins") or {}).get("home")),
        "wins_away":             _safe_int((fix.get("wins") or {}).get("away")),
        "wins_total":            _safe_int((fix.get("wins") or {}).get("total")),
        # draws
        "draws_home":            _safe_int((fix.get("draws") or {}).get("home")),
        "draws_away":            _safe_int((fix.get("draws") or {}).get("away")),
        "draws_total":           _safe_int((fix.get("draws") or {}).get("total")),
        # losses
        "losses_home":           _safe_int((fix.get("loses") or {}).get("home")),
        "losses_away":           _safe_int((fix.get("loses") or {}).get("away")),
        "losses_total":          _safe_int((fix.get("loses") or {}).get("total")),
        # goals for
        "goals_for_home":        _safe_int((gf.get("total") or {}).get("home")),
        "goals_for_away":        _safe_int((gf.get("total") or {}).get("away")),
        "goals_for_total":       _safe_int((gf.get("total") or {}).get("total")),
        "goals_for_avg_home":    _safe_float((gf.get("average") or {}).get("home")),
        "goals_for_avg_away":    _safe_float((gf.get("average") or {}).get("away")),
        "goals_for_avg_total":   _safe_float((gf.get("average") or {}).get("total")),
        # goals against
        "goals_against_home":      _safe_int((ga.get("total") or {}).get("home")),
        "goals_against_away":      _safe_int((ga.get("total") or {}).get("away")),
        "goals_against_total":     _safe_int((ga.get("total") or {}).get("total")),
        "goals_against_avg_home":  _safe_float((ga.get("average") or {}).get("home")),
        "goals_against_avg_away":  _safe_float((ga.get("average") or {}).get("away")),
        "goals_against_avg_total": _safe_float((ga.get("average") or {}).get("total")),
        # defence
        "clean_sheets_home":     _safe_int(cs.get("home")),
        "clean_sheets_away":     _safe_int(cs.get("away")),
        "clean_sheets_total":    _safe_int(cs.get("total")),
        "failed_to_score_home":  _safe_int(fts.get("home")),
        "failed_to_score_away":  _safe_int(fts.get("away")),
        "failed_to_score_total": _safe_int(fts.get("total")),
        # penalties
        "penalty_scored_total":  _safe_int((pen.get("scored") or {}).get("total")),
        "penalty_missed_total":  _safe_int((pen.get("missed") or {}).get("total")),
        "penalty_total":         _safe_int(pen.get("total")),
        # discipline
        "yellow_cards_total":    _sum_card_buckets(cards.get("yellow")),
        "red_cards_total":       _sum_card_buckets(cards.get("red")),
        # metadata
        "ingested_at":           _INGESTED_AT,
        "data_source":           "api-football-v3",
    }


def ingest_team_stats(seasons: list[int] = SEASONS) -> int:
    rows: list[dict] = []

    for season in seasons:
        teams_resp = _get("teams", {"league": WC_LEAGUE_ID, "season": season})
        teams = teams_resp.get("response", [])
        print(f"  team_stats season={season}: fetching stats for {len(teams)} teams …")

        for item in teams:
            team = item.get("team") or {}
            team_id   = team.get("id")
            team_name = team.get("name", "")
            stats_resp = _get(
                "teams/statistics",
                {"league": WC_LEAGUE_ID, "season": season, "team": team_id},
            )
            rows.append(_normalise_team_stats(
                stats_resp.get("response", {}), team_id, team_name, season
            ))
            time.sleep(0.15)  # stay within API rate limits

    if not rows:
        print("team_stats: no data returned.")
        return 0

    df = pd.DataFrame(rows)
    nullable_ints = [
        "team_id", "season", "league_id",
        "played_home", "played_away", "played_total",
        "wins_home", "wins_away", "wins_total",
        "draws_home", "draws_away", "draws_total",
        "losses_home", "losses_away", "losses_total",
        "goals_for_home", "goals_for_away", "goals_for_total",
        "goals_against_home", "goals_against_away", "goals_against_total",
        "clean_sheets_home", "clean_sheets_away", "clean_sheets_total",
        "failed_to_score_home", "failed_to_score_away", "failed_to_score_total",
        "penalty_scored_total", "penalty_missed_total", "penalty_total",
        "yellow_cards_total", "red_cards_total",
    ]
    for col in nullable_ints:
        df[col] = pd.array(df[col], dtype=pd.Int64Dtype())

    count = upload_dataframe_with_schema(df, "team_stats", TEAM_STATS_SCHEMA, "WRITE_TRUNCATE")
    print(f"team_stats: {count} rows written (WRITE_TRUNCATE) — seasons {seasons}")
    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_enriched_ingestion(seasons: list[int] = SEASONS) -> None:
    print("=" * 60)
    print("  World Cup 2026 — Priority-1 Enrichment Ingestion")
    print(f"  Seasons: {seasons}")
    print("=" * 60)

    n_standings = ingest_standings(seasons)
    n_events    = ingest_fixture_events(seasons)
    n_team_stats = ingest_team_stats(seasons)

    print()
    print("=" * 60)
    print("  Summary")
    print(f"  standings      : {n_standings} rows")
    print(f"  fixture_events : {n_events} rows")
    print(f"  team_stats     : {n_team_stats} rows")
    print("=" * 60)


if __name__ == "__main__":
    run_enriched_ingestion()
