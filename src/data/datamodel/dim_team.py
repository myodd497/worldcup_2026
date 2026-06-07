"""dim_team — canonical team dimension.

Grain: one row per team_id.

Built entirely from raw_* tables (zero API calls). Pulls team_ids from:
  - raw_fixtures (home_team_id, away_team_id)
  - raw_standings (team_id)

For each team_id, picks the most recent non-NULL team_name observed.

is_wc2026_participant is resolved exclusively from fact_match:
any team that appears as home or away in a match with competition_id=1
(World Cup) and season_year=2026 is flagged as TRUE. This is fully
data-driven — as soon as WC2026 fixtures are ingested via the API,
participating teams are automatically detected without any hard-coded
lists.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "dim_team"

DIM_TEAM_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("team_id",     "INT64",  mode="REQUIRED",
                         description="Primary key. API-Football team ID."),
    bigquery.SchemaField("team_name",   "STRING",
                         description="Most recent name observed for this team across raw_* tables."),
    bigquery.SchemaField("is_wc2026_participant", "BOOL",
                         description="True if this team is an official FIFA World Cup 2026 participant (48-team tournament)."),
    bigquery.SchemaField("first_seen_date", "DATE",
                         description="Earliest match_date this team appears in raw_fixtures."),
    bigquery.SchemaField("last_seen_date",  "DATE",
                         description="Latest match_date this team appears in raw_fixtures."),
    bigquery.SchemaField("match_count",  "INT64",
                         description="Total matches in raw_fixtures involving this team (home + away)."),
    bigquery.SchemaField("built_at",     "TIMESTAMP", mode="REQUIRED",
                         description="UTC timestamp this dimension row was built."),
]


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


def build() -> dict[str, object]:
    """Builds dim_team via CREATE OR REPLACE. Idempotent.

    is_wc2026_participant is resolved exclusively from fact_match:
    any team with competition_id=1 (World Cup) and season_year=2026
    is flagged TRUE. Fully data-driven — no hard-coded team lists.
    """
    client = _bq_tools._client()
    fqn = _fqn()
    now_iso = datetime.now(timezone.utc).isoformat()

    select_sql = f"""
    WITH all_team_rows AS (
      SELECT home_team_id AS team_id, home_team_name AS team_name,
             match_date, kickoff_at
      FROM {_ref('raw_fixtures')}
      WHERE home_team_id IS NOT NULL
      UNION ALL
      SELECT away_team_id AS team_id, away_team_name AS team_name,
             match_date, kickoff_at
      FROM {_ref('raw_fixtures')}
      WHERE away_team_id IS NOT NULL
      UNION ALL
      SELECT team_id, team_name,
             snapshot_date AS match_date,
             TIMESTAMP(snapshot_date) AS kickoff_at
      FROM {_ref('raw_standings')}
      WHERE team_id IS NOT NULL
    ),
    name_per_team AS (
      SELECT
        team_id,
        ARRAY_AGG(team_name IGNORE NULLS ORDER BY kickoff_at DESC LIMIT 1)[SAFE_OFFSET(0)] AS team_name,
        MIN(match_date) AS first_seen_date,
        MAX(match_date) AS last_seen_date,
        COUNT(*)        AS match_count
      FROM all_team_rows
      GROUP BY team_id
    ),
    wc2026_ids AS (
      SELECT home_team_id AS team_id
      FROM {_ref('fact_match')}
      WHERE competition_id = 1 AND season_year = 2026 AND home_team_id IS NOT NULL
      UNION DISTINCT
      SELECT away_team_id AS team_id
      FROM {_ref('fact_match')}
      WHERE competition_id = 1 AND season_year = 2026 AND away_team_id IS NOT NULL
    )
    SELECT
      n.team_id,
      n.team_name,
      (w.team_id IS NOT NULL) AS is_wc2026_participant,
      n.first_seen_date,
      n.last_seen_date,
      n.match_count,
      TIMESTAMP('{now_iso}') AS built_at
    FROM name_per_team n
    LEFT JOIN (SELECT DISTINCT team_id FROM wc2026_ids) w USING (team_id)
    """

    table = bigquery.Table(fqn, schema=DIM_TEAM_SCHEMA)
    table.description = (
        "Canonical team dimension. Grain: one row per team_id. "
        "Built from raw_fixtures + raw_standings. "
        "is_wc2026_participant is set exclusively from fact_match "
        "(competition_id=1, season_year=2026). "
        "Fully data-driven — zero hard-coded IDs, zero API calls."
    )
    table.clustering_fields = ["team_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)

    job_config = bigquery.QueryJobConfig(
        destination=fqn,
        write_disposition="WRITE_TRUNCATE",
    )
    client.query(select_sql, job_config=job_config).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("dim_team built: %d teams", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
