"""dim_team — canonical team dimension.

Grain: one row per team_id.

Built entirely from raw_* tables (zero API calls). Pulls team_ids from:
  - raw_fixtures (home_team_id, away_team_id)
  - raw_standings (team_id)

For each team_id, picks the most recent non-NULL team_name observed.
Also tags is_wc2026_participant via legacy `team_stats` season=2026 when present.
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
                         description="True if this team appears as a WC 2026 participant in legacy team_stats(season=2026)."),
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
    """Builds dim_team via CREATE OR REPLACE. Idempotent."""
    client = _bq_tools._client()
    fqn = _fqn()
    now_iso = datetime.now(timezone.utc).isoformat()

    has_team_stats = _table_exists("team_stats")
    wc_cte = ""
    wc_join = ""
    wc_select = "FALSE AS is_wc2026_participant"
    if has_team_stats:
        wc_cte = f"""
        , wc AS (
          SELECT DISTINCT team_id
          FROM {_ref('team_stats')}
          WHERE season = 2026 AND team_id IS NOT NULL
        )
        """
        wc_join = "LEFT JOIN wc USING (team_id)"
        wc_select = "(wc.team_id IS NOT NULL) AS is_wc2026_participant"

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
    )
    {wc_cte}
    SELECT
      n.team_id,
      n.team_name,
      {wc_select},
      n.first_seen_date,
      n.last_seen_date,
      n.match_count,
      TIMESTAMP('{now_iso}') AS built_at
    FROM name_per_team n
    {wc_join}
    """

    table = bigquery.Table(fqn, schema=DIM_TEAM_SCHEMA)
    table.description = (
        "Canonical team dimension. Grain: one row per team_id. "
        "Built from raw_fixtures + raw_standings (+ legacy team_stats for WC2026 flag). "
        "Zero API calls."
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
