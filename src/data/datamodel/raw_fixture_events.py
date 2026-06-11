"""raw_fixture_events — append-only raw layer for /fixtures/events.

Grain: one row per (match_id, event_seq, ingested_at).
  event_seq orders events within a match by (time_elapsed, time_extra, payload order).

Strategy (always BQ-first):
  1) Backfill from legacy `fixture_events` table for match_ids not present
  2) For completed matches in raw_fixtures with zero events yet -> call API
     (scopable by competition_ids and/or since_date to stay inside API budget)

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

TABLE_NAME = "raw_fixture_events"
API_BASE = "https://v3.football.api-sports.io"
API_SLEEP_SECONDS = 0.06


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

RAW_FIXTURE_EVENTS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("match_id",        "INT64",     mode="REQUIRED",
                         description="FK to raw_fixtures.match_id. API-Football fixture ID."),
    bigquery.SchemaField("event_seq",       "INT64",     mode="REQUIRED",
                         description="1-based sequence number of the event within the match (ordered by time)."),
    bigquery.SchemaField("match_date",      "DATE",
                         description="UTC calendar date of the match. Partition key."),
    bigquery.SchemaField("minute",          "INT64",
                         description="Regulation minute (time.elapsed). 0-90+ typically."),
    bigquery.SchemaField("minute_extra",    "INT64",
                         description="Stoppage-time / extra-time minute offset (time.extra). NULL if none."),
    bigquery.SchemaField("team_id",         "INT64",
                         description="Team responsible for / affected by the event."),
    bigquery.SchemaField("team_name",       "STRING",   description="Team name at the time of the match."),
    bigquery.SchemaField("player_id",       "INT64",    description="Primary player ID (scorer, fouler, subbed-off, etc.)."),
    bigquery.SchemaField("player_name",     "STRING",   description="Primary player name."),
    bigquery.SchemaField("assist_id",       "INT64",    description="Secondary player ID (assister, subbed-on)."),
    bigquery.SchemaField("assist_name",     "STRING",   description="Secondary player name."),
    bigquery.SchemaField("event_type",      "STRING",
                         description="Event type from API: Goal, Card, subst, Var, etc."),
    bigquery.SchemaField("event_detail",    "STRING",
                         description="Sub-type: Normal Goal, Own Goal, Penalty, Yellow Card, Red Card, Substitution 1, Goal cancelled, etc."),
    bigquery.SchemaField("event_comments",  "STRING",   description="Free-text comments from API (often NULL)."),
    bigquery.SchemaField("raw_payload",     "STRING",
                         description="Full JSON-encoded API event object. NULL when row was synthesized from legacy BQ tables."),
    bigquery.SchemaField("data_source",     "STRING", mode="REQUIRED",
                         description="Origin: 'api-football-v3' or 'legacy:fixture_events'."),
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
    table = bigquery.Table(_fqn(), schema=RAW_FIXTURE_EVENTS_SCHEMA)
    table.description = (
        "Append-only raw layer for the API-Football /fixtures/events endpoint. "
        "One row per match event (goal, card, sub, VAR). Downstream fact_match_event "
        "dedups by (match_id, event_seq) using the latest ingested_at."
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
    """Copies legacy fixture_events into raw_fixture_events for any match_id not
    already present. Assigns event_seq via ROW_NUMBER over (time_elapsed, time_extra).
    """
    ensure_table()
    if not _table_exists("fixture_events"):
        logger.info("Legacy fixture_events table missing; nothing to backfill.")
        return 0

    client = _bq_tools._client()
    now_iso = datetime.now(timezone.utc).isoformat()

    insert_sql = f"""
    INSERT INTO {_ref()} (
      match_id, event_seq, match_date, minute, minute_extra,
      team_id, team_name, player_id, player_name, assist_id, assist_name,
      event_type, event_detail, event_comments,
      raw_payload, data_source, ingested_at
    )
    WITH src AS (
      SELECT
        fe.fixture_id                                AS match_id,
        ROW_NUMBER() OVER (
          PARTITION BY fe.fixture_id
          ORDER BY fe.time_elapsed, IFNULL(fe.time_extra, 0)
        )                                            AS event_seq,
        rf.match_date                                AS match_date,
        fe.time_elapsed                              AS minute,
        fe.time_extra                                AS minute_extra,
        fe.team_id, fe.team_name,
        fe.player_id, fe.player_name,
        fe.assist_id, fe.assist_name,
        fe.event_type, fe.event_detail, fe.event_comments,
        CAST(NULL AS STRING)                         AS raw_payload,
        'legacy:fixture_events'                      AS data_source,
        TIMESTAMP('{now_iso}')                       AS ingested_at
      FROM {_ref('fixture_events')} fe
      LEFT JOIN (
        SELECT match_id, ANY_VALUE(match_date) AS match_date
        FROM {_ref('raw_fixtures')}
        GROUP BY match_id
      ) rf
        ON rf.match_id = fe.fixture_id
      WHERE fe.fixture_id IS NOT NULL
    )
    SELECT src.* FROM src
    LEFT JOIN (SELECT DISTINCT match_id FROM {_ref()}) existing
      USING (match_id)
    WHERE existing.match_id IS NULL
    """
    job = client.query(insert_sql)
    job.result()
    inserted = int(job.num_dml_affected_rows or 0)
    logger.info("Backfill from legacy fixture_events: +%d rows", inserted)
    return inserted


# ---------------------------------------------------------------------------
# API ingestion (delta only, scopable)
# ---------------------------------------------------------------------------

def _matches_missing_events(
    limit: int | None = None,
    *,
    competition_ids: list[int] | None = None,
    since_date: str | None = None,
    max_age_days: int | None = 30,
) -> list[tuple[int, datetime | None]]:
    """Returns completed matches in raw_fixtures still missing from raw_fixture_events.

    Args:
        limit: cap the number of returned matches.
        competition_ids: include matches whose competition_id is in this list.
        since_date: 'YYYY-MM-DD'; include matches on/after this date.
        max_age_days: hard floor on match_date >= today - N days. Prevents
            burning daily API quota on ancient fixtures that the API will
            never have events for. Pass None to disable.

    Scoping is a union of (competition_ids, since_date), intersected with the
    recency floor.
    """
    limit_clause = f"LIMIT {int(limit)}" if limit else ""

    scope_clauses: list[str] = []
    if competition_ids:
        ids_csv = ",".join(str(int(x)) for x in competition_ids)
        scope_clauses.append(f"competition_id IN ({ids_csv})")
    if since_date:
        scope_clauses.append(f"match_date >= DATE('{since_date}')")
    scope_sql = " OR ".join(scope_clauses) if scope_clauses else "TRUE"

    age_sql = (
        f"match_date >= DATE_SUB(CURRENT_DATE('UTC'), INTERVAL {int(max_age_days)} DAY)"
        if max_age_days is not None else "TRUE"
    )

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
      SELECT * FROM completed WHERE ({scope_sql}) AND ({age_sql})
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


def _normalise_event(
    ev: dict, *, match_id: int, match_date, event_seq: int, ingested_at: datetime
) -> dict:
    t = ev.get("time") or {}
    team = ev.get("team") or {}
    player = ev.get("player") or {}
    assist = ev.get("assist") or {}
    return {
        "match_id":       match_id,
        "event_seq":      event_seq,
        "match_date":     match_date,
        "minute":         _safe_int(t.get("elapsed")),
        "minute_extra":   _safe_int(t.get("extra")),
        "team_id":        _safe_int(team.get("id")),
        "team_name":      team.get("name") or None,
        "player_id":      _safe_int(player.get("id")),
        "player_name":    player.get("name") or None,
        "assist_id":      _safe_int(assist.get("id")),
        "assist_name":    assist.get("name") or None,
        "event_type":     ev.get("type") or None,
        "event_detail":   ev.get("detail") or None,
        "event_comments": ev.get("comments") or None,
        "raw_payload":    json.dumps(ev, ensure_ascii=False),
        "data_source":    "api-football-v3",
        "ingested_at":    ingested_at,
    }


def _write_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    int_cols = [
        "match_id", "event_seq", "minute", "minute_extra",
        "team_id", "player_id", "assist_id",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    return _bq_tools.upload_dataframe_with_schema(
        df, TABLE_NAME, RAW_FIXTURE_EVENTS_SCHEMA, write_disposition="WRITE_APPEND"
    )


def ingest_missing(
    limit: int | None = None,
    *,
    competition_ids: list[int] | None = None,
    since_date: str | None = None,
    max_age_days: int | None = 30,
) -> dict[str, int]:
    """Fetches /fixtures/events for completed matches still missing from raw_fixture_events.

    See _matches_missing_events for scoping semantics.
    """
    ensure_table()
    targets = _matches_missing_events(
        limit=limit, competition_ids=competition_ids,
        since_date=since_date, max_age_days=max_age_days,
    )
    logger.info(
        "ingest_missing: %d match(es) need events (competition_ids=%s since_date=%s max_age_days=%s)",
        len(targets), competition_ids, since_date, max_age_days,
    )

    api_calls = 0
    written = 0
    matches_with_events = 0
    matches_with_no_events = 0
    now = datetime.now(timezone.utc)

    for match_id, match_date in targets:
        data = _get("fixtures/events", {"fixture": match_id})
        api_calls += 1
        events = data.get("response", []) or []

        if not events:
            matches_with_no_events += 1
            time.sleep(API_SLEEP_SECONDS)
            continue

        rows = [
            _normalise_event(
                ev, match_id=match_id, match_date=match_date,
                event_seq=i + 1, ingested_at=now,
            )
            for i, ev in enumerate(events)
        ]
        written += _write_rows(rows)
        matches_with_events += 1
        time.sleep(API_SLEEP_SECONDS)

        if api_calls % 50 == 0:
            logger.info(
                "  progress: api=%d written=%d matches_with_events=%d empty=%d",
                api_calls, written, matches_with_events, matches_with_no_events,
            )

    summary = {
        "api_calls": api_calls,
        "written": written,
        "matches_with_events": matches_with_events,
        "matches_with_no_events": matches_with_no_events,
    }
    logger.info("raw_fixture_events ingest_missing: %s", summary)
    return summary


def run(
    limit: int | None = None,
    *,
    competition_ids: list[int] | None = None,
    since_date: str | None = None,
    max_age_days: int | None = 30,
) -> dict[str, object]:
    ensure_table()
    backfilled = backfill_from_legacy()
    ingest_stats = ingest_missing(
        limit=limit, competition_ids=competition_ids,
        since_date=since_date, max_age_days=max_age_days,
    )
    return {
        "backfilled_rows": backfilled,
        "ingest": ingest_stats,
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of API calls (one per missing match) for this run.")
    parser.add_argument("--competition-ids", type=str, default=None,
                        help="Comma-separated competition IDs to include (e.g. '1').")
    parser.add_argument("--since-date", type=str, default=None,
                        help="ISO date YYYY-MM-DD. Include matches on/after this date.")
    args = parser.parse_args()
    comp_ids = [int(x) for x in args.competition_ids.split(",")] if args.competition_ids else None
    print(run(limit=args.limit, competition_ids=comp_ids, since_date=args.since_date))
