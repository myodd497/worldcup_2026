"""raw_fixture_statistics — append-only raw layer for /fixtures/statistics.

Grain: one row per (match_id, team_id, stat_type, ingested_at) — long format
mirroring the API. The downstream fact_match_team will pivot this to wide.

Strategy (always BQ-first):
  1) Backfill from legacy `fixture_stats` for match_ids not present
  2) For completed matches in raw_fixtures with zero stats yet -> call API
     (scopable by competition_ids and/or since_date to stay inside API budget)

Writes are batched: rows accumulate in memory and flush every BATCH_SIZE
matches (or at end of run). Cuts BQ load-job overhead vs. one job per match.

Public entrypoints:
  - ensure_table()
  - backfill_from_legacy()
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
import pandas as pd
from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools
from src.tools.api_usage_tracker import record_api_call

logger = logging.getLogger(__name__)

TABLE_NAME = "raw_fixture_statistics"
API_BASE = "https://v3.football.api-sports.io"
API_SLEEP_SECONDS = 0.12
BATCH_SIZE = 50  # flush every N matches


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

RAW_FIXTURE_STATISTICS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("match_id",        "INT64",     mode="REQUIRED",
                         description="FK to raw_fixtures.match_id. API-Football fixture ID."),
    bigquery.SchemaField("match_date",      "DATE",
                         description="UTC calendar date of the match. Partition key."),
    bigquery.SchemaField("team_id",         "INT64",     mode="REQUIRED",
                         description="Team this stat row belongs to."),
    bigquery.SchemaField("team_name",       "STRING",   description="Team name at the time of the match."),
    bigquery.SchemaField("stat_type",       "STRING",   mode="REQUIRED",
                         description="Stat name from API (e.g. 'Shots on Goal', 'Ball Possession', 'Total passes')."),
    bigquery.SchemaField("stat_value_num",  "FLOAT64",
                         description="Numeric value when the stat is numeric or a percent. Percent stored as 0-100."),
    bigquery.SchemaField("stat_value_text", "STRING",
                         description="Raw textual value as returned by API (e.g. '60%', '12'). Always populated; numeric is parsed alongside."),
    bigquery.SchemaField("stat_value_unit", "STRING",
                         description="Unit suffix when present: 'pct' for percents, otherwise NULL."),
    bigquery.SchemaField("raw_payload",     "STRING",
                         description="Full JSON for {type,value}. NULL for legacy backfill rows."),
    bigquery.SchemaField("data_source",     "STRING", mode="REQUIRED",
                         description="Origin: 'api-football-v3' or 'legacy:fixture_stats'."),
    bigquery.SchemaField("ingested_at",     "TIMESTAMP", mode="REQUIRED",
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


def _parse_stat_value(value) -> tuple[float | None, str | None, str | None]:
    """Returns (numeric, text, unit). Handles ints, floats, percent strings, None."""
    if value is None:
        return None, None, None
    if isinstance(value, (int, float)):
        return float(value), str(value), None
    text = str(value).strip()
    if not text:
        return None, "", None
    if text.endswith("%"):
        try:
            return float(text[:-1]), text, "pct"
        except ValueError:
            return None, text, "pct"
    try:
        return float(text), text, None
    except ValueError:
        return None, text, None


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
    table = bigquery.Table(_fqn(), schema=RAW_FIXTURE_STATISTICS_SCHEMA)
    table.description = (
        "Append-only raw layer for the API-Football /fixtures/statistics endpoint. "
        "Long format: one row per (match_id, team_id, stat_type). Downstream "
        "fact_match_team pivots to wide and dedups by latest ingested_at."
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="match_date",
    )
    table.clustering_fields = ["match_id", "team_id"]
    client.create_table(table)
    logger.info("Created table %s", _fqn())


# ---------------------------------------------------------------------------
# Backfill (zero API calls)
# ---------------------------------------------------------------------------

def backfill_from_legacy() -> int:
    ensure_table()
    if not _table_exists("fixture_stats"):
        logger.info("Legacy fixture_stats table missing; nothing to backfill.")
        return 0

    client = _bq_tools._client()
    now_iso = datetime.now(timezone.utc).isoformat()

    insert_sql = f"""
    INSERT INTO {_ref()} (
      match_id, match_date, team_id, team_name,
      stat_type, stat_value_num, stat_value_text, stat_value_unit,
      raw_payload, data_source, ingested_at
    )
    WITH src AS (
      SELECT
        fs.fixture_id                                AS match_id,
        COALESCE(fs.match_date, rf.match_date)       AS match_date,
        fs.team_id, fs.team_name,
        fs.stat_type,
        fs.stat_value_num,
        fs.stat_value_text,
        fs.stat_value_unit,
        CAST(NULL AS STRING)                         AS raw_payload,
        'legacy:fixture_stats'                       AS data_source,
        TIMESTAMP('{now_iso}')                       AS ingested_at
      FROM {_ref('fixture_stats')} fs
      LEFT JOIN (
        SELECT match_id, ANY_VALUE(match_date) AS match_date
        FROM {_ref('raw_fixtures')}
        GROUP BY match_id
      ) rf
        ON rf.match_id = fs.fixture_id
      WHERE fs.fixture_id IS NOT NULL
    )
    SELECT src.* FROM src
    LEFT JOIN (SELECT DISTINCT match_id FROM {_ref()}) existing
      USING (match_id)
    WHERE existing.match_id IS NULL
    """
    job = client.query(insert_sql)
    job.result()
    inserted = int(job.num_dml_affected_rows or 0)
    logger.info("Backfill from legacy fixture_stats: +%d rows", inserted)
    return inserted


# ---------------------------------------------------------------------------
# API ingestion (delta only, scopable, batched)
# ---------------------------------------------------------------------------

def _matches_missing_stats(
    limit: int | None = None,
    *,
    competition_ids: list[int] | None = None,
    since_date: str | None = None,
) -> list[tuple[int, datetime | None]]:
    limit_clause = f"LIMIT {int(limit)}" if limit else ""

    scope_clauses: list[str] = []
    if competition_ids:
        ids_csv = ",".join(str(int(x)) for x in competition_ids)
        scope_clauses.append(f"competition_id IN ({ids_csv})")
    if since_date:
        scope_clauses.append(f"match_date >= DATE('{since_date}')")
    scope_sql = " OR ".join(scope_clauses) if scope_clauses else "TRUE"

    sql = f"""
    WITH completed AS (
      SELECT
        match_id,
        ANY_VALUE(match_date)     AS match_date,
        ANY_VALUE(competition_id) AS competition_id
      FROM {_ref('raw_fixtures')}
      WHERE is_completed = TRUE AND match_id IS NOT NULL
      GROUP BY match_id
    ),
    scoped AS (
      SELECT * FROM completed WHERE {scope_sql}
    ),
    have AS (
      SELECT DISTINCT match_id FROM {_ref()}
    )
    SELECT s.match_id, s.match_date
    FROM scoped s
    LEFT JOIN have h USING (match_id)
    WHERE h.match_id IS NULL
    ORDER BY s.match_date DESC
    {limit_clause}
    """
    df = _bq_tools.run_query(sql)
    return [(int(r.match_id), r.match_date) for r in df.itertuples()]


def _normalise_team_block(
    team_block: dict, *, match_id: int, match_date, ingested_at: datetime
) -> list[dict]:
    team = team_block.get("team") or {}
    team_id = _safe_int(team.get("id"))
    team_name = team.get("name") or None
    rows: list[dict] = []
    for stat in team_block.get("statistics") or []:
        stat_type = stat.get("type") or None
        if not stat_type or team_id is None:
            continue
        num, text, unit = _parse_stat_value(stat.get("value"))
        rows.append({
            "match_id":        match_id,
            "match_date":      match_date,
            "team_id":         team_id,
            "team_name":       team_name,
            "stat_type":       stat_type,
            "stat_value_num":  num,
            "stat_value_text": text,
            "stat_value_unit": unit,
            "raw_payload":     json.dumps(stat, ensure_ascii=False),
            "data_source":     "api-football-v3",
            "ingested_at":     ingested_at,
        })
    return rows


def _flush(rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    for c in ("match_id", "team_id"):
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    written = _bq_tools.upload_dataframe_with_schema(
        df, TABLE_NAME, RAW_FIXTURE_STATISTICS_SCHEMA, write_disposition="WRITE_APPEND"
    )
    return written


def ingest_missing(
    limit: int | None = None,
    *,
    competition_ids: list[int] | None = None,
    since_date: str | None = None,
) -> dict[str, int]:
    ensure_table()
    targets = _matches_missing_stats(
        limit=limit, competition_ids=competition_ids, since_date=since_date,
    )
    logger.info(
        "ingest_missing: %d match(es) need stats (competition_ids=%s since_date=%s)",
        len(targets), competition_ids, since_date,
    )

    api_calls = 0
    written = 0
    matches_with_stats = 0
    matches_with_no_stats = 0
    now = datetime.now(timezone.utc)
    buffer: list[dict] = []

    for idx, (match_id, match_date) in enumerate(targets, start=1):
        data = _get("fixtures/statistics", {"fixture": match_id})
        api_calls += 1
        blocks = data.get("response", []) or []

        if not blocks:
            matches_with_no_stats += 1
        else:
            match_rows: list[dict] = []
            for block in blocks:
                match_rows.extend(
                    _normalise_team_block(
                        block, match_id=match_id, match_date=match_date, ingested_at=now,
                    )
                )
            if match_rows:
                buffer.extend(match_rows)
                matches_with_stats += 1
            else:
                matches_with_no_stats += 1

        if idx % BATCH_SIZE == 0:
            written += _flush(buffer)
            buffer.clear()
            logger.info(
                "  progress: api=%d/%d written=%d matches_with_stats=%d empty=%d",
                api_calls, len(targets), written, matches_with_stats, matches_with_no_stats,
            )

        time.sleep(API_SLEEP_SECONDS)

    written += _flush(buffer)

    summary = {
        "api_calls": api_calls,
        "written": written,
        "matches_with_stats": matches_with_stats,
        "matches_with_no_stats": matches_with_no_stats,
    }
    logger.info("raw_fixture_statistics ingest_missing: %s", summary)
    return summary


def run(
    limit: int | None = None,
    *,
    competition_ids: list[int] | None = None,
    since_date: str | None = None,
) -> dict[str, object]:
    ensure_table()
    backfilled = backfill_from_legacy()
    ingest_stats = ingest_missing(
        limit=limit, competition_ids=competition_ids, since_date=since_date,
    )
    return {
        "backfilled_rows": backfilled,
        "ingest": ingest_stats,
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--competition-ids", type=str, default=None,
                        help="Comma-separated competition IDs (e.g. '1').")
    parser.add_argument("--since-date", type=str, default=None,
                        help="ISO date YYYY-MM-DD.")
    args = parser.parse_args()
    comp_ids = [int(x) for x in args.competition_ids.split(",")] if args.competition_ids else None
    print(run(limit=args.limit, competition_ids=comp_ids, since_date=args.since_date))
