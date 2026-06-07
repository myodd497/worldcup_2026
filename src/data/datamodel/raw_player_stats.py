"""raw_player_stats — append-only raw layer for /fixtures/players endpoint.

Grain: one row per (match_id, team_id, player_id, ingested_at).
Stores per-player match statistics: minutes, goals, assists, shots, passes, xG, etc.

Strategy (BQ-first):
  1) For completed matches in raw_fixtures with zero player stats → call API
     (scopable by competition_ids and/or since_date)
  2) Writes are batched: rows accumulate and flush every BATCH_SIZE matches

Public entrypoints:
  - ensure_table()
  - ingest_missing(limit=None, competition_ids=None, since_date=None)
  - run(...)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import httpx
from google.cloud import bigquery
from google.cloud.bigquery import LoadJobConfig, WriteDisposition

from src.tools import bigquery_tools as _bq_tools
from src.tools.api_usage_tracker import record_api_call

logger = logging.getLogger(__name__)

TABLE_NAME = "raw_player_stats"
API_BASE = "https://v3.football.api-sports.io"
API_SLEEP_SECONDS = 0.12
BATCH_SIZE = 50  # flush every N matches


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

RAW_PLAYER_STATS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("match_id",         "INT64",     mode="REQUIRED",
                         description="FK to raw_fixtures.match_id. API-Football fixture ID."),
    bigquery.SchemaField("match_date",       "DATE",
                         description="UTC calendar date of the match. Partition key."),
    bigquery.SchemaField("team_id",          "INT64",     mode="REQUIRED",
                         description="Team this player belongs to."),
    bigquery.SchemaField("team_name",        "STRING",    description="Team name at match time."),
    bigquery.SchemaField("player_id",        "INT64",     mode="REQUIRED",
                         description="Player ID from API-Football."),
    bigquery.SchemaField("player_name",      "STRING",    description="Player full name."),
    bigquery.SchemaField("player_number",    "INT64",
                         description="Jersey number. NULL if not available."),
    bigquery.SchemaField("player_position",  "STRING",
                         description="Position: G, D, M, F. NULL if not available."),
    bigquery.SchemaField("is_starter",       "BOOL",
                         description="True if player was in the starting XI."),
    bigquery.SchemaField("minutes_played",   "INT64",
                         description="Minutes played in regulation (excl. stoppage)."),
    bigquery.SchemaField("goals",            "INT64",
                         description="Goals scored by this player in this match."),
    bigquery.SchemaField("assists",          "INT64",
                         description="Assists by this player in this match."),
    bigquery.SchemaField("shots_total",      "INT64",
                         description="Total shots attempted."),
    bigquery.SchemaField("shots_on_target",  "INT64",
                         description="Shots on target."),
    bigquery.SchemaField("passes_total",     "INT64",
                         description="Total passes attempted."),
    bigquery.SchemaField("passes_accurate",  "INT64",
                         description="Accurate passes."),
    bigquery.SchemaField("passes_key",       "INT64",
                         description="Key passes (leading to shots)."),
    bigquery.SchemaField("dribbles_total",   "INT64",
                         description="Total dribble attempts."),
    bigquery.SchemaField("dribbles_success", "INT64",
                         description="Successful dribbles."),
    bigquery.SchemaField("tackles_total",    "INT64",
                         description="Total tackles."),
    bigquery.SchemaField("interceptions",    "INT64",
                         description="Interceptions."),
    bigquery.SchemaField("fouls_committed",  "INT64",
                         description="Fouls committed by this player."),
    bigquery.SchemaField("fouls_drawn",      "INT64",
                         description="Fouls drawn by this player."),
    bigquery.SchemaField("yellow_cards",     "INT64",
                         description="Yellow cards received."),
    bigquery.SchemaField("red_cards",        "INT64",
                         description="Red cards received."),
    bigquery.SchemaField("penalty_scored",   "INT64",
                         description="Penalties scored."),
    bigquery.SchemaField("penalty_missed",   "INT64",
                         description="Penalties missed."),
    bigquery.SchemaField("saves",            "INT64",
                         description="Goalkeeper saves. NULL for outfield players."),
    bigquery.SchemaField("goals_conceded",   "INT64",
                         description="Goals conceded (goalkeeper). NULL for outfield players."),
    bigquery.SchemaField("rating",           "FLOAT64",
                         description="Player rating (API-Football scale, typically 0-10)."),
    bigquery.SchemaField("raw_payload",      "STRING",
                         description="Full JSON payload from API."),
    bigquery.SchemaField("data_source",      "STRING", mode="REQUIRED",
                         description="Origin: 'api-football-v3'."),
    bigquery.SchemaField("ingested_at",      "TIMESTAMP", mode="REQUIRED",
                         description="UTC timestamp this row was written to BigQuery."),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fqn(table: str = TABLE_NAME) -> str:
    project = os.environ["BIGQUERY_PROJECT_ID"]
    dataset = os.environ["BIGQUERY_DATASET_ID"]
    return f"{project}.{dataset}.{table}"


def _ref(table: str = TABLE_NAME) -> str:
    return f"`{_fqn(table)}`"


def _table_exists(table: str) -> bool:
    client = _bq_tools._client()
    try:
        client.get_table(_fqn(table))
        return True
    except Exception:
        return False


def _safe_int(val) -> int | None:
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None


def _headers() -> dict[str, str]:
    return {"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")}


def _get(endpoint: str, params: dict) -> dict:
    with httpx.Client(timeout=20) as client:
        resp = client.get(f"{API_BASE}/{endpoint}", headers=_headers(), params=params)
        resp.raise_for_status()
        record_api_call(endpoint=endpoint, response_headers=dict(resp.headers))
        return resp.json()


# ---------------------------------------------------------------------------
# Table management
# ---------------------------------------------------------------------------

def ensure_table() -> None:
    client = _bq_tools._client()
    if _table_exists(TABLE_NAME):
        return
    table = bigquery.Table(_fqn(), schema=RAW_PLAYER_STATS_SCHEMA)
    table.description = (
        "Append-only raw layer for the API-Football /fixtures/players endpoint. "
        "Grain: one row per (match_id, team_id, player_id, ingested_at). "
        "Downstream dim_player + fact_player_match_stat deduplicate by latest ingested_at."
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="match_date",
    )
    table.clustering_fields = ["match_id", "team_id", "player_id"]
    client.create_table(table)
    logger.info("Created table %s", _fqn())


# ---------------------------------------------------------------------------
# Parse API response → flat rows
# ---------------------------------------------------------------------------

def _parse_player_stat_response(
    match_id: int,
    match_date: str,
    response: list[dict],
    data_source: str,
    now_iso: str,
) -> list[dict]:
    """Convert API /fixtures/players response into flat row dicts."""
    rows: list[dict] = []
    for team_block in response:
        team = team_block.get("team", {})
        team_id = _safe_int(team.get("id"))
        team_name = team.get("name", "")
        players = team_block.get("players", [])

        for player in players:
            p = player.get("player", {})
            stats_list = player.get("statistics", [])
            if not stats_list:
                continue
            stats = stats_list[0]  # Usually one stats block per player

            games = stats.get("games", {})
            shots = stats.get("shots", {})
            passes = stats.get("passes", {})
            dribbles = stats.get("dribbles", {})
            tackles = stats.get("tackles", {})
            duels = stats.get("duels", {})
            fouls = stats.get("fouls", {})
            cards = stats.get("cards", {})
            penalty = stats.get("penalty", {})
            goalkeeper = stats.get("goalkeeper", {})

            rows.append({
                "match_id": match_id,
                "match_date": match_date,
                "team_id": team_id,
                "team_name": team_name,
                "player_id": _safe_int(p.get("id")),
                "player_name": (p.get("name") or "").strip(),
                "player_number": _safe_int(p.get("number")),
                "player_position": (stats.get("position") or "").strip() or None,
                "is_starter": bool(games.get("position") and games.get("position") != "Substitute"),
                "minutes_played": _safe_int(games.get("minutes")),
                "goals": _safe_int(stats.get("goals", {}).get("total")) or 0,
                "assists": _safe_int(stats.get("goals", {}).get("assists")) or 0,
                "shots_total": _safe_int(shots.get("total")) or 0,
                "shots_on_target": _safe_int(shots.get("on")) or 0,
                "passes_total": _safe_int(passes.get("total")) or 0,
                "passes_accurate": _safe_int(passes.get("accurate")) or 0,
                "passes_key": _safe_int(passes.get("key")) or 0,
                "dribbles_total": _safe_int(dribbles.get("attempts")) or 0,
                "dribbles_success": _safe_int(dribbles.get("success")) or 0,
                "tackles_total": _safe_int(tackles.get("total")) or 0,
                "interceptions": _safe_int(tackles.get("interceptions")) or 0,
                "fouls_committed": _safe_int(fouls.get("committed")) or 0,
                "fouls_drawn": _safe_int(fouls.get("drawn")) or 0,
                "yellow_cards": _safe_int(cards.get("yellow")) or 0,
                "red_cards": _safe_int(cards.get("red")) or 0,
                "penalty_scored": _safe_int(penalty.get("scored")) or 0,
                "penalty_missed": _safe_int(penalty.get("missed")) or 0,
                "saves": _safe_int(goalkeeper.get("saves")) if goalkeeper else None,
                "goals_conceded": _safe_int(goalkeeper.get("goals_conceded")) if goalkeeper else None,
                "rating": float(stats.get("rating")) if stats.get("rating") else None,
                "raw_payload": json.dumps(player, default=str),
                "data_source": data_source,
                "ingested_at": now_iso,
            })
    return rows


# ---------------------------------------------------------------------------
# Ingest missing player stats
# ---------------------------------------------------------------------------

def ingest_missing(
    limit: int | None = None,
    competition_ids: list[int] | None = None,
    since_date: str | None = None,
) -> dict:
    """Fetch player stats from API-Football for completed matches that don't yet
    have player data in raw_player_stats. Respects API rate limits."""
    ensure_table()
    client = _bq_tools._client()

    # Find match_ids that have fixtures but no player stats yet
    conditions = [
        "rf.is_completed = TRUE",
        "rf.match_date IS NOT NULL",
    ]
    if competition_ids:
        ids = ",".join(str(c) for c in competition_ids)
        conditions.append(f"rf.competition_id IN ({ids})")
    if since_date:
        conditions.append(f"rf.match_date >= '{since_date}'")

    where = " AND ".join(conditions)
    limit_clause = f"LIMIT {int(limit)}" if limit else ""

    find_sql = f"""
    SELECT rf.match_id, rf.match_date
    FROM {_ref('raw_fixtures')} rf
    LEFT JOIN (
        SELECT DISTINCT match_id FROM {_ref()}
    ) rps ON rps.match_id = rf.match_id
    WHERE {where}
      AND rps.match_id IS NULL
    ORDER BY rf.match_date DESC
    {limit_clause}
    """

    try:
        df = _bq_tools.run_query(find_sql)
    except Exception as exc:
        logger.warning("Query for missing player stats failed: %s", exc)
        return {"ingested": 0, "matches_processed": 0, "total_rows": 0, "error": str(exc)}

    if df.empty:
        logger.info("No matches missing player stats.")
        return {"ingested": 0, "matches_processed": 0, "total_rows": 0}

    match_ids = df["match_id"].tolist()
    match_dates = dict(zip(df["match_id"], df["match_date"]))
    logger.info("Found %d matches missing player stats (limit=%s)", len(match_ids), limit)

    all_rows: list[dict] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    api_calls = 0

    for i, match_id in enumerate(match_ids):
        try:
            data = _get("fixtures/players", {"fixture": match_id})
            api_calls += 1
            response = data.get("response", [])
            if not response:
                logger.debug("No player data for match_id=%s (empty response)", match_id)
                continue

            match_date = str(match_dates.get(match_id, ""))
            rows = _parse_player_stat_response(
                match_id=match_id,
                match_date=match_date,
                response=response,
                data_source="api-football-v3",
                now_iso=now_iso,
            )
            all_rows.extend(rows)
        except httpx.HTTPError as exc:
            logger.warning("API error for match_id=%s: %s", match_id, exc)
            continue
        except Exception as exc:
            logger.warning("Unexpected error for match_id=%s: %s", match_id, exc)
            continue

        # Flush in batches
        if len(all_rows) >= BATCH_SIZE * 22:  # ~22 players per match
            _flush_rows(all_rows)
            logger.info("Flushed %d rows so far (%d/%d matches)", len(all_rows), i + 1, len(match_ids))

        if i < len(match_ids) - 1:
            time.sleep(API_SLEEP_SECONDS)

    # Final flush
    if all_rows:
        _flush_rows(all_rows)

    total_rows = len(all_rows)
    logger.info(
        "Player stats ingest complete: %d matches processed, %d rows inserted, %d API calls",
        len(match_ids), total_rows, api_calls,
    )
    return {
        "ingested": total_rows,
        "matches_processed": len(match_ids),
        "total_rows": total_rows,
        "api_calls": api_calls,
    }


def _flush_rows(rows: list[dict]) -> None:
    """Write accumulated rows to BigQuery."""
    if not rows:
        return
    client = _bq_tools._client()
    import pandas as pd
    df = pd.DataFrame(rows)
    # Ensure proper date/timestamp types
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    if "ingested_at" in df.columns:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"])
    # Ensure nullable integer columns are proper Int64 (not float)
    int_cols = [
        "match_id", "team_id", "player_id", "player_number",
        "minutes_played", "goals", "assists", "shots_total", "shots_on_target",
        "passes_total", "passes_accurate", "passes_key",
        "dribbles_total", "dribbles_success", "tackles_total", "interceptions",
        "fouls_committed", "fouls_drawn", "yellow_cards", "red_cards",
        "penalty_scored", "penalty_missed", "saves", "goals_conceded",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    # Ensure string columns are proper strings (not None/NaN)
    str_cols = ["player_name", "team_name", "player_position", "data_source", "raw_payload"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    job_config = LoadJobConfig(
        schema=RAW_PLAYER_STATS_SCHEMA,
        write_disposition=WriteDisposition.WRITE_APPEND,
    )
    client.load_table_from_dataframe(df, _fqn(), job_config=job_config).result()
    rows.clear()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run(
    limit: int | None = None,
    competition_ids: list[int] | None = None,
    since_date: str | None = None,
) -> dict:
    """Fetch player stats from API-Football for matches missing data.

    Args:
        limit: Max matches to process (None = all missing).
        competition_ids: e.g. [1] for World Cup only.
        since_date: ISO date string, e.g. '2026-06-01'. Only matches >= this date.
    """
    return ingest_missing(limit=limit, competition_ids=competition_ids, since_date=since_date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    result = run(limit=5, since_date="2026-06-01")
    print(result)
