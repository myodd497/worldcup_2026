"""
Fixture statistics ingestion across all competitions for WC 2026 teams.

Source table: team_match_history
  - Uses distinct finished fixtures already discovered in BQ.

Target table: fixture_stats
  - Long format: one row per (fixture_id, team_id, stat_type)
  - Idempotent: skips fixture_ids already present in fixture_stats
  - WRITE_APPEND

Usage:
    set -a && source .env && set +a
    poetry run python -m src.data.ingest_fixture_stats
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import httpx
import pandas as pd
from google.cloud import bigquery
from httpx import HTTPStatusError

from src.tools.bigquery_tools import run_query, upload_dataframe_with_schema, _table_ref

_BASE = "https://v3.football.api-sports.io"
_FINAL_STATUSES = {"FT", "AET", "PEN", "AWD", "WO"}
_INGESTED_AT = datetime.now(timezone.utc)


def _headers() -> dict[str, str]:
    return {"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")}


def _get(endpoint: str, params: dict) -> dict:
    with httpx.Client(timeout=20) as client:
        resp = client.get(f"{_BASE}/{endpoint}", headers=_headers(), params=params)
        resp.raise_for_status()
        return resp.json()


def _get_with_retry(endpoint: str, params: dict, max_attempts: int = 6) -> dict:
    """GET wrapper with exponential backoff for transient API errors (especially 429)."""
    wait_seconds = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            return _get(endpoint, params)
        except HTTPStatusError as exc:
            code = exc.response.status_code
            if code not in {429, 500, 502, 503, 504} or attempt == max_attempts:
                raise
            print(f"  retry {attempt}/{max_attempts} on HTTP {code}, sleeping {wait_seconds:.1f}s")
            time.sleep(wait_seconds)
            wait_seconds = min(wait_seconds * 2.0, 60.0)


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


def _parse_stat_value(raw) -> tuple[float | None, str, str]:
    """Normalizes API stat values into numeric + text + unit representation.

    Returns:
        value_num: numeric representation if parseable, else None
        value_text: stable string representation
        value_unit: 'percent', 'count', or 'text'
    """
    if raw is None:
        return None, "", "count"

    if isinstance(raw, (int, float)):
        return float(raw), str(raw), "count"

    text = str(raw).strip()
    if text == "":
        return None, "", "count"

    if text.endswith("%"):
        num = _safe_float(text[:-1])
        return num, text, "percent"

    num = _safe_float(text)
    if num is not None:
        return num, text, "count"

    return None, text, "text"


FIXTURE_STATS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("fixture_id", "INT64", description="API-Football fixture ID"),
    bigquery.SchemaField("season", "INT64", description="Season year from source fixture"),
    bigquery.SchemaField("competition_id", "INT64", description="API-Football competition/league ID"),
    bigquery.SchemaField("competition_name", "STRING", description="Competition name for the fixture"),
    bigquery.SchemaField("match_date", "DATE", description="Fixture date in UTC"),
    bigquery.SchemaField("team_id", "INT64", description="Team ID this statistic belongs to"),
    bigquery.SchemaField("team_name", "STRING", description="Team name this statistic belongs to"),
    bigquery.SchemaField("is_home", "BOOL", description="True if this team played at home in this fixture"),
    bigquery.SchemaField("opponent_id", "INT64", description="Opponent team ID"),
    bigquery.SchemaField("opponent_name", "STRING", description="Opponent team name"),
    bigquery.SchemaField("stat_type", "STRING", description="Statistic name from API e.g. Ball Possession, Shots on Goal"),
    bigquery.SchemaField("stat_value_num", "FLOAT64", description="Numeric value when parseable (percentages stored as numeric without % sign)"),
    bigquery.SchemaField("stat_value_text", "STRING", description="Raw textual value from API (e.g. 56%)"),
    bigquery.SchemaField("stat_value_unit", "STRING", description="Unit classification: percent, count, or text"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", description="UTC ingestion timestamp"),
    bigquery.SchemaField("data_source", "STRING", description="Origin of data: api-football-v3"),
]


def _existing_fixture_ids() -> set[int]:
    try:
        df = run_query(f"SELECT DISTINCT fixture_id FROM `{_table_ref('fixture_stats')}`")
        return set(int(x) for x in df["fixture_id"].tolist())
    except Exception:
        return set()


def _fixture_catalog() -> pd.DataFrame:
    proj = os.environ["BIGQUERY_PROJECT_ID"]
    ds = os.environ["BIGQUERY_DATASET_ID"]
    return run_query(
        f"""
        SELECT
            fixture_id,
            ANY_VALUE(season) AS season,
            ANY_VALUE(competition_id) AS competition_id,
            ANY_VALUE(competition_name) AS competition_name,
            ANY_VALUE(match_date) AS match_date,
            ANY_VALUE(home_team_id) AS home_team_id,
            ANY_VALUE(home_team_name) AS home_team_name,
            ANY_VALUE(away_team_id) AS away_team_id,
            ANY_VALUE(away_team_name) AS away_team_name
        FROM `{proj}.{ds}.team_match_history`
        WHERE status IN ('FT', 'AET', 'PEN', 'AWD', 'WO')
        GROUP BY fixture_id
        ORDER BY fixture_id
        """
    )


def _rows_from_fixture_stats(
    fixture_id: int,
    season: int,
    competition_id: int,
    competition_name: str,
    match_date,
    home_team_id: int,
    home_team_name: str,
    away_team_id: int,
    away_team_name: str,
    api_response: list[dict],
) -> list[dict]:
    out: list[dict] = []
    for team_stats in api_response:
        team = team_stats.get("team") or {}
        team_id = _safe_int(team.get("id"))
        team_name = team.get("name") or ""
        is_home = team_id == home_team_id

        opponent_id = away_team_id if is_home else home_team_id
        opponent_name = away_team_name if is_home else home_team_name

        for stat in team_stats.get("statistics", []):
            stat_type = stat.get("type") or ""
            raw_value = stat.get("value")
            value_num, value_text, value_unit = _parse_stat_value(raw_value)
            out.append(
                {
                    "fixture_id": fixture_id,
                    "season": season,
                    "competition_id": competition_id,
                    "competition_name": competition_name,
                    "match_date": match_date,
                    "team_id": team_id,
                    "team_name": team_name,
                    "is_home": is_home,
                    "opponent_id": opponent_id,
                    "opponent_name": opponent_name,
                    "stat_type": stat_type,
                    "stat_value_num": value_num,
                    "stat_value_text": value_text,
                    "stat_value_unit": value_unit,
                    "ingested_at": _INGESTED_AT,
                    "data_source": "api-football-v3",
                }
            )
    return out


def ingest_fixture_stats() -> int:
    catalog = _fixture_catalog()
    existing = _existing_fixture_ids()

    todo = catalog[~catalog["fixture_id"].isin(existing)]
    if todo.empty:
        print("fixture_stats: nothing new to ingest.")
        return 0

    print(
        f"fixture_stats: {len(existing)} fixtures already loaded, {len(todo)} fixtures pending."
    )

    rows: list[dict] = []
    total_written = 0
    calls = 0
    missing = 0
    batch_size = 300

    for rec in todo.itertuples(index=False):
        fixture_id = int(rec.fixture_id)
        data = _get_with_retry("fixtures/statistics", {"fixture": fixture_id})
        calls += 1
        response = data.get("response", [])

        if not response:
            missing += 1
            continue

        rows.extend(
            _rows_from_fixture_stats(
                fixture_id=fixture_id,
                season=_safe_int(rec.season),
                competition_id=_safe_int(rec.competition_id),
                competition_name=rec.competition_name or "",
                match_date=rec.match_date,
                home_team_id=_safe_int(rec.home_team_id),
                home_team_name=rec.home_team_name or "",
                away_team_id=_safe_int(rec.away_team_id),
                away_team_name=rec.away_team_name or "",
                api_response=response,
            )
        )

        if calls % batch_size == 0 and rows:
            df_batch = pd.DataFrame(rows)
            int_cols = [
                "fixture_id",
                "season",
                "competition_id",
                "team_id",
                "opponent_id",
            ]
            for col in int_cols:
                df_batch[col] = pd.array(df_batch[col], dtype=pd.Int64Dtype())

            written = upload_dataframe_with_schema(
                df_batch,
                table_name="fixture_stats",
                schema=FIXTURE_STATS_SCHEMA,
                write_disposition="WRITE_APPEND",
            )
            total_written += written
            rows = []
            print(f"  checkpoint: wrote {written} rows at {calls} API calls")

        if calls % 250 == 0:
            print(f"  progress: {calls}/{len(todo)} fixtures fetched")

        # Keep steady under minute-rate limits.
        time.sleep(0.27)

    if not rows and total_written == 0:
        print(
            f"fixture_stats: no rows produced ({calls} calls, {missing} fixtures without stats)."
        )
        return 0

    count = total_written
    if rows:
        df = pd.DataFrame(rows)

        int_cols = [
            "fixture_id",
            "season",
            "competition_id",
            "team_id",
            "opponent_id",
        ]
        for col in int_cols:
            df[col] = pd.array(df[col], dtype=pd.Int64Dtype())

        count += upload_dataframe_with_schema(
            df,
            table_name="fixture_stats",
            schema=FIXTURE_STATS_SCHEMA,
            write_disposition="WRITE_APPEND",
        )

    print(
        f"fixture_stats: {count} rows appended ({calls} API calls, {missing} fixtures without stats)."
    )
    return count


def run_fixture_stats_ingestion() -> None:
    print("=" * 60)
    print("  Fixture Stats Ingestion")
    print("  Source: finished fixtures from team_match_history")
    print("  Target: fixture_stats (long format)")
    print("=" * 60)
    written = ingest_fixture_stats()
    print("=" * 60)
    print(f"  Total rows written: {written}")
    print("=" * 60)


if __name__ == "__main__":
    run_fixture_stats_ingestion()
