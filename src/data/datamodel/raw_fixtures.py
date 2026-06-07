"""raw_fixtures — append-only raw layer for the API-Football /fixtures endpoint.

Grain: one row per (match_id, ingested_at). Append-only audit log of every
fixture payload we have seen. Downstream `fact_match` will pick the latest
row per match_id via QUALIFY ROW_NUMBER().

Sources, in priority order (we always prefer reading BQ over calling the API):
  1) Existing canonical BQ tables  (team_match_history, fixtures_historical)
     -> seed/backfill, zero API calls
  2) API-Football /fixtures
     -> only for match_ids missing from raw_fixtures, or for non-FINISHED
        matches whose last snapshot is older than FRESHNESS_TTL_HOURS

Public entrypoints:
  - ensure_table()                : create the table if it does not exist
  - backfill_from_legacy()        : seed raw_fixtures from existing BQ tables
  - ingest_by_league(...)         : fetch /fixtures?league=&season= (delta only)
  - ingest_by_team(...)           : fetch /fixtures?team=&season=   (delta only)
  - run()                         : orchestrates the above for default scope
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import httpx
import pandas as pd
from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools
from src.tools.api_usage_tracker import record_api_call

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TABLE_NAME = "raw_fixtures"

API_BASE = "https://v3.football.api-sports.io"
WC_LEAGUE_ID = 1
WC_LEAGUE_SEASONS = [2018, 2022, 2026]
TEAM_HISTORY_SEASONS = list(range(2020, 2027))   # 2020..2026 inclusive

# A finished match is immutable; never re-fetch.
FINAL_STATUSES = {"FT", "AET", "PEN", "AWD", "WO"}
# Re-fetch non-final fixtures if our latest snapshot is older than this.
FRESHNESS_TTL_HOURS = 24
# Polite throttling between API calls.
API_SLEEP_SECONDS = 0.12


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

RAW_FIXTURES_SCHEMA: list[bigquery.SchemaField] = [
    # --- Identity ---
    bigquery.SchemaField("match_id",            "INT64",     mode="REQUIRED",
                         description="API-Football fixture ID. PK at downstream layer (dedup by ingested_at)."),
    # --- Time ---
    bigquery.SchemaField("kickoff_at",          "TIMESTAMP",
                         description="UTC kickoff timestamp from fixture.date."),
    bigquery.SchemaField("match_date",          "DATE",
                         description="UTC calendar date of kickoff. Partition key."),
    bigquery.SchemaField("season_year",         "INT64",
                         description="Season as defined by the API (league.season)."),
    # --- Competition ---
    bigquery.SchemaField("competition_id",      "INT64",
                         description="API-Football league ID (league.id)."),
    bigquery.SchemaField("competition_name",    "STRING",
                         description="Competition name (league.name)."),
    bigquery.SchemaField("competition_country", "STRING",
                         description="Country / confederation owning the competition (league.country)."),
    bigquery.SchemaField("competition_round",   "STRING",
                         description="Round / matchday label (league.round)."),
    # --- Teams ---
    bigquery.SchemaField("home_team_id",        "INT64",  description="API-Football team ID of the home side."),
    bigquery.SchemaField("home_team_name",      "STRING", description="Home team name at the time of the match."),
    bigquery.SchemaField("away_team_id",        "INT64",  description="API-Football team ID of the away side."),
    bigquery.SchemaField("away_team_name",      "STRING", description="Away team name at the time of the match."),
    # --- Venue / officials ---
    bigquery.SchemaField("venue_id",            "INT64",  description="API-Football venue ID (fixture.venue.id), if present."),
    bigquery.SchemaField("venue_name",          "STRING", description="Stadium name (fixture.venue.name)."),
    bigquery.SchemaField("venue_city",          "STRING", description="City where the match was played (fixture.venue.city)."),
    bigquery.SchemaField("referee_name",        "STRING", description="Referee name (fixture.referee)."),
    # --- Status ---
    bigquery.SchemaField("match_status_raw",    "STRING",
                         description="Raw status short code (FT, AET, PEN, NS, PST, CANC, LIVE, ...)."),
    bigquery.SchemaField("is_completed",        "BOOL",
                         description="True if match_status_raw is in {FT, AET, PEN, AWD, WO}."),
    # --- Scores (as reported in this payload snapshot) ---
    bigquery.SchemaField("home_goals",          "INT64", description="Final home goals at this snapshot. NULL if not played yet."),
    bigquery.SchemaField("away_goals",          "INT64", description="Final away goals at this snapshot. NULL if not played yet."),
    bigquery.SchemaField("home_goals_halftime", "INT64", description="Home half-time goals (score.halftime.home)."),
    bigquery.SchemaField("away_goals_halftime", "INT64", description="Away half-time goals (score.halftime.away)."),
    bigquery.SchemaField("home_goals_extratime","INT64", description="Home goals scored in extra time only (score.extratime.home)."),
    bigquery.SchemaField("away_goals_extratime","INT64", description="Away goals scored in extra time only (score.extratime.away)."),
    bigquery.SchemaField("home_goals_penalty",  "INT64", description="Home penalty shootout goals (score.penalty.home)."),
    bigquery.SchemaField("away_goals_penalty",  "INT64", description="Away penalty shootout goals (score.penalty.away)."),
    # --- Raw payload + lineage ---
    bigquery.SchemaField("raw_payload",         "STRING",
                         description="Full JSON-encoded API-Football fixture object for replay/debug. NULL when row was synthesized from legacy BQ tables."),
    bigquery.SchemaField("data_source",         "STRING", mode="REQUIRED",
                         description="Origin of this row: 'api-football-v3', 'legacy:team_match_history', 'legacy:fixtures_historical'."),
    bigquery.SchemaField("ingested_at",         "TIMESTAMP", mode="REQUIRED",
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
    """Creates raw_fixtures with explicit schema, partitioning and clustering."""
    client = _bq_tools._client()
    if _table_exists(TABLE_NAME):
        return
    table = bigquery.Table(_fqn(), schema=RAW_FIXTURES_SCHEMA)
    table.description = (
        "Append-only raw layer for the API-Football /fixtures endpoint. "
        "One row per (match_id, ingested_at). Downstream fact_match dedups by "
        "match_id, keeping the latest ingested_at. NEVER queried directly by the app."
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="match_date",
    )
    table.clustering_fields = ["match_id", "competition_id"]
    client.create_table(table)
    logger.info("Created table %s", _fqn())


# ---------------------------------------------------------------------------
# Freshness lookups
# ---------------------------------------------------------------------------

@dataclass
class FixtureSnapshot:
    last_ingested_at: datetime
    last_status: str


def _latest_snapshot_per_match() -> dict[int, FixtureSnapshot]:
    """Returns {match_id: latest snapshot} from raw_fixtures. Empty if table missing."""
    if not _table_exists(TABLE_NAME):
        return {}
    sql = f"""
    SELECT
      match_id,
      MAX(ingested_at) AS last_ingested_at,
      ARRAY_AGG(match_status_raw ORDER BY ingested_at DESC LIMIT 1)[OFFSET(0)] AS last_status
    FROM {_ref()}
    GROUP BY match_id
    """
    df = _bq_tools.run_query(sql)
    out: dict[int, FixtureSnapshot] = {}
    for r in df.itertuples():
        out[int(r.match_id)] = FixtureSnapshot(
            last_ingested_at=r.last_ingested_at,
            last_status=(r.last_status or ""),
        )
    return out


def _needs_refresh(match_id: int, snapshots: dict[int, FixtureSnapshot]) -> bool:
    snap = snapshots.get(match_id)
    if snap is None:
        return True
    if snap.last_status in FINAL_STATUSES:
        return False
    age = datetime.now(timezone.utc) - snap.last_ingested_at
    return age >= timedelta(hours=FRESHNESS_TTL_HOURS)


# ---------------------------------------------------------------------------
# Normalization (API payload -> row)
# ---------------------------------------------------------------------------

def _normalise_api_fixture(fix: dict, *, ingested_at: datetime) -> dict | None:
    fixture = fix.get("fixture") or {}
    league  = fix.get("league")  or {}
    teams   = fix.get("teams")   or {}
    goals   = fix.get("goals")   or {}
    score   = fix.get("score")   or {}
    venue   = fixture.get("venue") or {}
    status  = (fixture.get("status") or {}).get("short", "") or ""

    match_id = _safe_int(fixture.get("id"))
    if match_id is None:
        return None

    raw_date = fixture.get("date", "")
    try:
        dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        dt = None

    ht  = score.get("halftime")  or {}
    et  = score.get("extratime") or {}
    pen = score.get("penalty")   or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}

    return {
        "match_id":             match_id,
        "kickoff_at":           dt,
        "match_date":           dt.date() if dt else None,
        "season_year":          _safe_int(league.get("season")),
        "competition_id":       _safe_int(league.get("id")),
        "competition_name":     league.get("name") or None,
        "competition_country":  league.get("country") or None,
        "competition_round":    league.get("round") or None,
        "home_team_id":         _safe_int(home.get("id")),
        "home_team_name":       home.get("name") or None,
        "away_team_id":         _safe_int(away.get("id")),
        "away_team_name":       away.get("name") or None,
        "venue_id":             _safe_int(venue.get("id")),
        "venue_name":           venue.get("name") or None,
        "venue_city":           venue.get("city") or None,
        "referee_name":         fixture.get("referee") or None,
        "match_status_raw":     status,
        "is_completed":         status in FINAL_STATUSES,
        "home_goals":           _safe_int(goals.get("home")),
        "away_goals":           _safe_int(goals.get("away")),
        "home_goals_halftime":  _safe_int(ht.get("home")),
        "away_goals_halftime":  _safe_int(ht.get("away")),
        "home_goals_extratime": _safe_int(et.get("home")),
        "away_goals_extratime": _safe_int(et.get("away")),
        "home_goals_penalty":   _safe_int(pen.get("home")),
        "away_goals_penalty":   _safe_int(pen.get("away")),
        "raw_payload":          json.dumps(fix, ensure_ascii=False),
        "data_source":          "api-football-v3",
        "ingested_at":          ingested_at,
    }


def _write_rows(rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    # Nullable integer columns to keep BQ types happy
    int_cols = [
        "match_id", "season_year", "competition_id",
        "home_team_id", "away_team_id", "venue_id",
        "home_goals", "away_goals",
        "home_goals_halftime", "away_goals_halftime",
        "home_goals_extratime", "away_goals_extratime",
        "home_goals_penalty", "away_goals_penalty",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    return _bq_tools.upload_dataframe_with_schema(
        df, TABLE_NAME, RAW_FIXTURES_SCHEMA, write_disposition="WRITE_APPEND"
    )


# ---------------------------------------------------------------------------
# Backfill (zero API calls)
# ---------------------------------------------------------------------------

def backfill_from_legacy() -> dict[str, int]:
    """Seeds raw_fixtures from existing BQ tables for any match_id not already
    present. Zero API calls.

    Returns:
        Dict like {'team_match_history': N1, 'fixtures_historical': N2}.
    """
    ensure_table()
    client = _bq_tools._client()
    results: dict[str, int] = {}
    now_iso = datetime.now(timezone.utc).isoformat()

    legacy_sources: list[tuple[str, str]] = []

    if _table_exists("team_match_history"):
        # team_match_history has one row per (team, fixture); collapse via DISTINCT-on-match_id
        legacy_sources.append((
            "team_match_history",
            f"""
            SELECT
              fixture_id                                       AS match_id,
              TIMESTAMP(match_datetime)                        AS kickoff_at,
              DATE(match_datetime)                             AS match_date,
              season                                           AS season_year,
              competition_id,
              competition_name,
              competition_country,
              competition_round,
              home_team_id,
              home_team_name,
              away_team_id,
              away_team_name,
              CAST(NULL AS INT64)                              AS venue_id,
              venue_name,
              venue_city,
              referee                                          AS referee_name,
              status                                           AS match_status_raw,
              status IN ('FT','AET','PEN','AWD','WO')          AS is_completed,
              home_goals,
              away_goals,
              home_goals_ht                                    AS home_goals_halftime,
              away_goals_ht                                    AS away_goals_halftime,
              home_goals_et                                    AS home_goals_extratime,
              away_goals_et                                    AS away_goals_extratime,
              home_goals_pen                                   AS home_goals_penalty,
              away_goals_pen                                   AS away_goals_penalty,
              CAST(NULL AS STRING)                             AS raw_payload,
              'legacy:team_match_history'                      AS data_source,
              TIMESTAMP('{now_iso}')                           AS ingested_at
            FROM {_ref('team_match_history')}
            WHERE fixture_id IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (PARTITION BY fixture_id ORDER BY ingested_at DESC) = 1
            """,
        ))

    if _table_exists("fixtures_historical"):
        legacy_sources.append((
            "fixtures_historical",
            f"""
            SELECT
              fixture_id                                       AS match_id,
              TIMESTAMP(date)                                  AS kickoff_at,
              DATE(date)                                       AS match_date,
              season                                           AS season_year,
              CAST(NULL AS INT64)                              AS competition_id,
              CAST(NULL AS STRING)                             AS competition_name,
              CAST(NULL AS STRING)                             AS competition_country,
              CAST(NULL AS STRING)                             AS competition_round,
              CAST(NULL AS INT64)                              AS home_team_id,
              home_team                                        AS home_team_name,
              CAST(NULL AS INT64)                              AS away_team_id,
              away_team                                        AS away_team_name,
              CAST(NULL AS INT64)                              AS venue_id,
              venue                                            AS venue_name,
              venue_city,
              referee                                          AS referee_name,
              status                                           AS match_status_raw,
              status IN ('FT','AET','PEN','AWD','WO')          AS is_completed,
              SAFE_CAST(home_goals AS INT64)                   AS home_goals,
              SAFE_CAST(away_goals AS INT64)                   AS away_goals,
              CAST(NULL AS INT64)                              AS home_goals_halftime,
              CAST(NULL AS INT64)                              AS away_goals_halftime,
              CAST(NULL AS INT64)                              AS home_goals_extratime,
              CAST(NULL AS INT64)                              AS away_goals_extratime,
              CAST(NULL AS INT64)                              AS home_goals_penalty,
              CAST(NULL AS INT64)                              AS away_goals_penalty,
              CAST(NULL AS STRING)                             AS raw_payload,
              'legacy:fixtures_historical'                     AS data_source,
              TIMESTAMP('{now_iso}')                           AS ingested_at
            FROM {_ref('fixtures_historical')}
            WHERE fixture_id IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (PARTITION BY fixture_id ORDER BY ingested_at DESC) = 1
            """,
        ))

    for label, src_sql in legacy_sources:
        # Only insert fixture_ids that aren't already in raw_fixtures.
        insert_sql = f"""
        INSERT INTO {_ref()} (
          match_id, kickoff_at, match_date, season_year,
          competition_id, competition_name, competition_country, competition_round,
          home_team_id, home_team_name, away_team_id, away_team_name,
          venue_id, venue_name, venue_city, referee_name,
          match_status_raw, is_completed,
          home_goals, away_goals,
          home_goals_halftime, away_goals_halftime,
          home_goals_extratime, away_goals_extratime,
          home_goals_penalty, away_goals_penalty,
          raw_payload, data_source, ingested_at
        )
        WITH src AS ({src_sql})
        SELECT src.* FROM src
        LEFT JOIN (SELECT DISTINCT match_id FROM {_ref()}) existing
          USING (match_id)
        WHERE existing.match_id IS NULL
        """
        job = client.query(insert_sql)
        job.result()
        inserted = int(job.num_dml_affected_rows or 0)
        results[label] = inserted
        logger.info("Backfill from %s: +%d rows into raw_fixtures", label, inserted)

    return results


# ---------------------------------------------------------------------------
# API ingestion (delta only)
# ---------------------------------------------------------------------------

def _ingest_response(
    fixtures: list[dict],
    snapshots: dict[int, FixtureSnapshot],
) -> tuple[int, int, int]:
    """Given an API response list, writes only the rows that need refresh.

    Returns (written, skipped_final, skipped_fresh).
    """
    now = datetime.now(timezone.utc)
    to_write: list[dict] = []
    skipped_final = 0
    skipped_fresh = 0

    for fix in fixtures:
        mid = _safe_int(((fix.get("fixture") or {}).get("id")))
        if mid is None:
            continue
        snap = snapshots.get(mid)
        if snap is not None:
            if snap.last_status in FINAL_STATUSES:
                skipped_final += 1
                continue
            if (now - snap.last_ingested_at) < timedelta(hours=FRESHNESS_TTL_HOURS):
                skipped_fresh += 1
                continue
        row = _normalise_api_fixture(fix, ingested_at=now)
        if row is not None:
            to_write.append(row)
            snapshots[mid] = FixtureSnapshot(last_ingested_at=now, last_status=row["match_status_raw"])

    written = _write_rows(to_write)
    return written, skipped_final, skipped_fresh


def ingest_by_league(
    league_id: int,
    seasons: Iterable[int],
    *,
    snapshots: dict[int, FixtureSnapshot] | None = None,
) -> dict[str, int]:
    """Fetches /fixtures?league=&season= per season, writing only delta rows."""
    ensure_table()
    if snapshots is None:
        snapshots = _latest_snapshot_per_match()

    api_calls = 0
    written = skipped_final = skipped_fresh = 0
    for season in seasons:
        data = _get("fixtures", {"league": league_id, "season": season})
        api_calls += 1
        fixtures = data.get("response", []) or []
        w, sf, sr = _ingest_response(fixtures, snapshots)
        written += w
        skipped_final += sf
        skipped_fresh += sr
        logger.info(
            "league=%s season=%s: api=1 fixtures=%d written=%d skipped_final=%d skipped_fresh=%d",
            league_id, season, len(fixtures), w, sf, sr,
        )
        time.sleep(API_SLEEP_SECONDS)

    return {
        "api_calls": api_calls,
        "written": written,
        "skipped_final": skipped_final,
        "skipped_fresh": skipped_fresh,
    }


def ingest_by_team(
    team_id: int,
    seasons: Iterable[int],
    *,
    snapshots: dict[int, FixtureSnapshot] | None = None,
) -> dict[str, int]:
    """Fetches /fixtures?team=&season= per season, writing only delta rows."""
    ensure_table()
    if snapshots is None:
        snapshots = _latest_snapshot_per_match()

    api_calls = 0
    written = skipped_final = skipped_fresh = 0
    for season in seasons:
        data = _get("fixtures", {"team": team_id, "season": season})
        api_calls += 1
        fixtures = data.get("response", []) or []
        w, sf, sr = _ingest_response(fixtures, snapshots)
        written += w
        skipped_final += sf
        skipped_fresh += sr
        time.sleep(API_SLEEP_SECONDS)

    logger.info(
        "team=%s seasons=%s: api=%d written=%d skipped_final=%d skipped_fresh=%d",
        team_id, list(seasons), api_calls, written, skipped_final, skipped_fresh,
    )
    return {
        "api_calls": api_calls,
        "written": written,
        "skipped_final": skipped_final,
        "skipped_fresh": skipped_fresh,
    }


def ingest_by_date_range(
    start_date: str,
    end_date: str | None = None,
    *,
    snapshots: dict[int, FixtureSnapshot] | None = None,
) -> dict[str, int]:
    """Fetches /fixtures?date=YYYY-MM-DD for each day in [start_date, end_date].

    Returns ALL fixtures worldwide for those days in one call per day. This is
    how completed matches across friendlies/qualifiers/club competitions enter
    the warehouse without needing per-team fetches.

    Args:
        start_date: ISO 'YYYY-MM-DD' (inclusive).
        end_date:   ISO 'YYYY-MM-DD' (inclusive). Defaults to today (UTC).
    """
    ensure_table()
    if snapshots is None:
        snapshots = _latest_snapshot_per_match()

    start_dt = datetime.fromisoformat(start_date).date()
    end_dt = (
        datetime.fromisoformat(end_date).date()
        if end_date else datetime.now(timezone.utc).date()
    )
    if start_dt > end_dt:
        start_dt = end_dt

    api_calls = 0
    written = skipped_final = skipped_fresh = 0
    day_count = 0

    cur = start_dt
    while cur <= end_dt:
        iso = cur.isoformat()
        data = _get("fixtures", {"date": iso})
        api_calls += 1
        day_count += 1
        fixtures = data.get("response", []) or []
        w, sf, sr = _ingest_response(fixtures, snapshots)
        written += w
        skipped_final += sf
        skipped_fresh += sr
        logger.info(
            "date=%s: api=1 fixtures=%d written=%d skipped_final=%d skipped_fresh=%d",
            iso, len(fixtures), w, sf, sr,
        )
        cur = cur + timedelta(days=1)
        time.sleep(API_SLEEP_SECONDS)

    return {
        "api_calls": api_calls,
        "written": written,
        "skipped_final": skipped_final,
        "skipped_fresh": skipped_fresh,
        "days": day_count,
    }


# ---------------------------------------------------------------------------
# Default orchestration
# ---------------------------------------------------------------------------

def _load_wc_team_ids_from_legacy() -> list[tuple[int, str]]:
    """Returns (team_id, team_name) for WC 2026 participants from the legacy
    team_stats table. Returns [] if the table is missing."""
    if not _table_exists("team_stats"):
        return []
    df = _bq_tools.run_query(
        f"SELECT DISTINCT team_id, team_name FROM {_ref('team_stats')} "
        f"WHERE season = 2026 AND team_id IS NOT NULL ORDER BY team_name"
    )
    return [(int(r.team_id), r.team_name) for r in df.itertuples()]


def run(
    *,
    league_seasons: list[int] | None = None,
    team_seasons: list[int] | None = None,
    skip_team_fetch: bool = False,
    since_date: str | None = None,
) -> dict[str, object]:
    """Default end-to-end ingestion:
      1) Ensure table exists
      2) Backfill from legacy BQ tables (zero API)
      3) Delta-fetch /fixtures by WC league + season
      4) Delta-fetch /fixtures by date in [since_date, today] — picks up
         friendlies, qualifiers, and club games across all competitions.
         Skipped when since_date is None.
      5) Delta-fetch /fixtures by each WC participant team + season
         (skipped when skip_team_fetch=True; default behaviour for cron).
    """
    ensure_table()
    backfill_counts = backfill_from_legacy()

    snapshots = _latest_snapshot_per_match()

    league_stats = ingest_by_league(
        WC_LEAGUE_ID,
        league_seasons or WC_LEAGUE_SEASONS,
        snapshots=snapshots,
    )

    date_stats: dict[str, int] = {
        "api_calls": 0, "written": 0,
        "skipped_final": 0, "skipped_fresh": 0, "days": 0,
    }
    if since_date:
        date_stats = ingest_by_date_range(since_date, snapshots=snapshots)

    team_stats: dict[str, int] = {"api_calls": 0, "written": 0,
                                  "skipped_final": 0, "skipped_fresh": 0}
    if not skip_team_fetch:
        teams = _load_wc_team_ids_from_legacy()
        for tid, _ in teams:
            res = ingest_by_team(
                tid,
                team_seasons or TEAM_HISTORY_SEASONS,
                snapshots=snapshots,
            )
            for k in team_stats:
                team_stats[k] += res[k]

    summary = {
        "backfill": backfill_counts,
        "league_fetch": league_stats,
        "date_fetch": date_stats,
        "team_fetch": team_stats,
        "total_api_calls": (
            league_stats["api_calls"]
            + date_stats["api_calls"]
            + team_stats["api_calls"]
        ),
    }
    logger.info("raw_fixtures run summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
