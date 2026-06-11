"""raw_standings — append-only raw layer for /standings.

Grain: one row per (competition_id, season_year, team_id, snapshot_date, ingested_at).
Each snapshot captures the league/competition standings at a point in time.

Strategy:
  1) Backfill from legacy `standings` table for snapshots not present
  2) Fetch /standings?league=&season= for default scope (WC + qualifiers)
     - Only if we don't already have a snapshot for today (or force=True)

Append-only. Downstream `fact_standings_snapshot` picks the latest snapshot per
(competition_id, season_year, team_id, snapshot_date) via QUALIFY.

Public entrypoints:
  - ensure_table()
  - backfill_from_legacy()
  - ingest_for(league_id, season)
  - run(league_seasons=None, force=False)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timezone

import httpx
import pandas as pd
from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools
from src.tools.api_usage_tracker import record_api_call

logger = logging.getLogger(__name__)

TABLE_NAME = "raw_standings"
API_BASE = "https://v3.football.api-sports.io"
API_SLEEP_SECONDS = 0.06

# Default scope: WC + likely qualifier comps. Keep small; expand as needed.
DEFAULT_LEAGUE_SEASONS: list[tuple[int, int]] = [
    (1, 2018),    # World Cup 2018
    (1, 2022),    # World Cup 2022
    (1, 2026),    # World Cup 2026
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

RAW_STANDINGS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("competition_id",     "INT64",     mode="REQUIRED",
                         description="API-Football league/competition ID."),
    bigquery.SchemaField("competition_name",   "STRING",
                         description="Competition name at the time of the snapshot."),
    bigquery.SchemaField("competition_country","STRING",
                         description="Country / confederation that owns the competition."),
    bigquery.SchemaField("season_year",        "INT64",     mode="REQUIRED",
                         description="Season year (league.season)."),
    bigquery.SchemaField("group_name",         "STRING",
                         description="Group / phase label inside the competition (e.g. 'Group A', 'Promotion - Final')."),
    bigquery.SchemaField("team_id",            "INT64",     mode="REQUIRED",
                         description="Team ID for this standing row."),
    bigquery.SchemaField("team_name",          "STRING",   description="Team name at snapshot time."),
    bigquery.SchemaField("standing_rank",      "INT64",
                         description="Rank within the group at snapshot time (1 = top)."),
    bigquery.SchemaField("points",             "INT64",    description="Points."),
    bigquery.SchemaField("played",             "INT64",    description="Matches played."),
    bigquery.SchemaField("wins",               "INT64",    description="Total wins."),
    bigquery.SchemaField("draws",              "INT64",    description="Total draws."),
    bigquery.SchemaField("losses",             "INT64",    description="Total losses."),
    bigquery.SchemaField("goals_for",          "INT64",    description="Goals scored across all matches."),
    bigquery.SchemaField("goals_against",      "INT64",    description="Goals conceded across all matches."),
    bigquery.SchemaField("goal_diff",          "INT64",    description="Goal difference (goals_for - goals_against)."),
    bigquery.SchemaField("form",               "STRING",   description="Recent results string (e.g. 'WWDLW')."),
    bigquery.SchemaField("standing_status",    "STRING",
                         description="API status string (e.g. 'same', 'up', 'down')."),
    bigquery.SchemaField("standing_description","STRING",
                         description="Free-text description (e.g. 'Promotion - Champions League')."),
    bigquery.SchemaField("snapshot_date",      "DATE",     mode="REQUIRED",
                         description="UTC calendar date of the snapshot. Partition key."),
    bigquery.SchemaField("raw_payload",        "STRING",
                         description="Full JSON-encoded API standing row. NULL for legacy backfill."),
    bigquery.SchemaField("data_source",        "STRING", mode="REQUIRED",
                         description="Origin: 'api-football-v3' or 'legacy:standings'."),
    bigquery.SchemaField("ingested_at",        "TIMESTAMP", mode="REQUIRED",
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
    table = bigquery.Table(_fqn(), schema=RAW_STANDINGS_SCHEMA)
    table.description = (
        "Append-only raw layer for the API-Football /standings endpoint. "
        "One row per (competition_id, season_year, team_id, snapshot_date). "
        "Downstream fact_standings_snapshot dedups by latest ingested_at."
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="snapshot_date",
    )
    table.clustering_fields = ["competition_id", "season_year", "team_id"]
    client.create_table(table)
    logger.info("Created table %s", _fqn())


# ---------------------------------------------------------------------------
# Backfill (zero API calls)
# ---------------------------------------------------------------------------

def backfill_from_legacy() -> int:
    """Copies legacy standings into raw_standings for (competition_id, season_year,
    team_id, snapshot_date) tuples not present yet."""
    ensure_table()
    if not _table_exists("standings"):
        logger.info("Legacy standings table missing; nothing to backfill.")
        return 0

    client = _bq_tools._client()
    now_iso = datetime.now(timezone.utc).isoformat()

    insert_sql = f"""
    INSERT INTO {_ref()} (
      competition_id, competition_name, competition_country, season_year,
      group_name, team_id, team_name, standing_rank,
      points, played, wins, draws, losses,
      goals_for, goals_against, goal_diff,
      form, standing_status, standing_description,
      snapshot_date, raw_payload, data_source, ingested_at
    )
    WITH src AS (
      SELECT
        s.league_id                                  AS competition_id,
        CAST(NULL AS STRING)                         AS competition_name,
        CAST(NULL AS STRING)                         AS competition_country,
        s.season                                     AS season_year,
        s.group_name,
        s.team_id, s.team_name,
        s.rank                                       AS standing_rank,
        s.points, s.played, s.wins, s.draws, s.losses,
        s.goals_for, s.goals_against, s.goal_diff,
        s.form,
        s.status                                     AS standing_status,
        s.description                                AS standing_description,
        s.snapshot_date,
        CAST(NULL AS STRING)                         AS raw_payload,
        'legacy:standings'                           AS data_source,
        TIMESTAMP('{now_iso}')                       AS ingested_at
      FROM {_ref('standings')} s
      WHERE s.league_id IS NOT NULL
        AND s.season IS NOT NULL
        AND s.team_id IS NOT NULL
        AND s.snapshot_date IS NOT NULL
    )
    SELECT src.* FROM src
    LEFT JOIN (
      SELECT DISTINCT competition_id, season_year, team_id, snapshot_date
      FROM {_ref()}
    ) existing
      USING (competition_id, season_year, team_id, snapshot_date)
    WHERE existing.competition_id IS NULL
    """
    job = client.query(insert_sql)
    job.result()
    inserted = int(job.num_dml_affected_rows or 0)
    logger.info("Backfill from legacy standings: +%d rows", inserted)
    return inserted


# ---------------------------------------------------------------------------
# API ingestion
# ---------------------------------------------------------------------------

def _have_snapshot_today(competition_id: int, season_year: int) -> bool:
    sql = f"""
    SELECT 1 AS x
    FROM {_ref()}
    WHERE competition_id = {int(competition_id)}
      AND season_year = {int(season_year)}
      AND snapshot_date = CURRENT_DATE('UTC')
    LIMIT 1
    """
    return not _bq_tools.run_query(sql).empty


def _normalise_standing(row: dict, *, competition: dict, season_year: int,
                        group_name: str | None, snapshot_date: date,
                        ingested_at: datetime) -> dict:
    team = row.get("team") or {}
    all_ = row.get("all") or {}
    goals = all_.get("goals") or {}
    return {
        "competition_id":       _safe_int(competition.get("id")),
        "competition_name":     competition.get("name") or None,
        "competition_country":  competition.get("country") or None,
        "season_year":          int(season_year),
        "group_name":           group_name,
        "team_id":              _safe_int(team.get("id")),
        "team_name":            team.get("name") or None,
        "standing_rank":        _safe_int(row.get("rank")),
        "points":               _safe_int(row.get("points")),
        "played":               _safe_int(all_.get("played")),
        "wins":                 _safe_int(all_.get("win")),
        "draws":                _safe_int(all_.get("draw")),
        "losses":               _safe_int(all_.get("lose")),
        "goals_for":            _safe_int(goals.get("for")),
        "goals_against":        _safe_int(goals.get("against")),
        "goal_diff":            _safe_int(row.get("goalsDiff")),
        "form":                 row.get("form") or None,
        "standing_status":      row.get("status") or None,
        "standing_description": row.get("description") or None,
        "snapshot_date":        snapshot_date,
        "raw_payload":          json.dumps(row, ensure_ascii=False),
        "data_source":          "api-football-v3",
        "ingested_at":          ingested_at,
    }


def _write_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    int_cols = [
        "competition_id", "season_year", "team_id", "standing_rank",
        "points", "played", "wins", "draws", "losses",
        "goals_for", "goals_against", "goal_diff",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    return _bq_tools.upload_dataframe_with_schema(
        df, TABLE_NAME, RAW_STANDINGS_SCHEMA, write_disposition="WRITE_APPEND"
    )


def ingest_for(league_id: int, season: int, *, force: bool = False) -> dict[str, int]:
    """Fetches /standings for a (league, season). Skips if today's snapshot already exists."""
    ensure_table()
    if not force and _have_snapshot_today(league_id, season):
        logger.info("Snapshot for league=%s season=%s already exists today; skipping.",
                    league_id, season)
        return {"api_calls": 0, "written": 0, "skipped": 1}

    data = _get("standings", {"league": league_id, "season": season})
    snapshot_date = datetime.now(timezone.utc).date()
    now = datetime.now(timezone.utc)

    league_blocks = (data.get("response") or [])
    rows: list[dict] = []
    for lb in league_blocks:
        league = lb.get("league") or {}
        groups = league.get("standings") or []
        # standings is list of groups; each group is list of team rows
        for group in groups:
            # API sometimes returns one flat list (single group), sometimes list-of-lists
            if group and isinstance(group, list) and group and not isinstance(group[0], dict):
                # nested - treat each sub-group separately
                for sub in group:
                    group_name = (sub[0].get("group") if sub else None)
                    for r in sub:
                        rows.append(_normalise_standing(
                            r, competition=league, season_year=season,
                            group_name=group_name, snapshot_date=snapshot_date,
                            ingested_at=now,
                        ))
            else:
                group_name = (group[0].get("group") if group else None)
                for r in group:
                    rows.append(_normalise_standing(
                        r, competition=league, season_year=season,
                        group_name=group_name, snapshot_date=snapshot_date,
                        ingested_at=now,
                    ))

    written = _write_rows(rows)
    summary = {"api_calls": 1, "written": written, "skipped": 0}
    logger.info("standings league=%s season=%s: %s", league_id, season, summary)
    return summary


def run(
    league_seasons: list[tuple[int, int]] | None = None,
    *,
    force: bool = False,
) -> dict[str, object]:
    ensure_table()
    backfilled = backfill_from_legacy()

    pairs = league_seasons or DEFAULT_LEAGUE_SEASONS
    totals = {"api_calls": 0, "written": 0, "skipped": 0}
    for league_id, season in pairs:
        res = ingest_for(league_id, season, force=force)
        for k in totals:
            totals[k] += res[k]
        time.sleep(API_SLEEP_SECONDS)

    return {"backfilled_rows": backfilled, "ingest": totals}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch even if today's snapshot exists.")
    args = parser.parse_args()
    print(run(force=args.force))
