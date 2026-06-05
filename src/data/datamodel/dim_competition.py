"""dim_competition — canonical competition (league) dimension.

Grain: one row per competition_id.
Built from raw_fixtures + raw_standings (zero API).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "dim_competition"


DIM_COMPETITION_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("competition_id",      "INT64",  mode="REQUIRED",
                         description="Primary key. API-Football league/competition ID."),
    bigquery.SchemaField("competition_name",    "STRING",
                         description="Most recent name observed for this competition."),
    bigquery.SchemaField("competition_country", "STRING",
                         description="Country or confederation owning the competition."),
    bigquery.SchemaField("is_world_cup",        "BOOL",
                         description="True if competition_id = 1 (FIFA World Cup in API-Football)."),
    bigquery.SchemaField("seasons_observed",    "INT64", mode="REPEATED",
                         description="Sorted distinct season_year values seen for this competition."),
    bigquery.SchemaField("first_seen_date",     "DATE",  description="Earliest match_date for this competition."),
    bigquery.SchemaField("last_seen_date",      "DATE",  description="Latest match_date for this competition."),
    bigquery.SchemaField("match_count",         "INT64",
                         description="Number of matches in raw_fixtures with this competition_id."),
    bigquery.SchemaField("built_at",            "TIMESTAMP", mode="REQUIRED",
                         description="UTC timestamp this dimension row was built."),
]


def _fqn(table: str = TABLE_NAME) -> str:
    project = os.environ["BIGQUERY_PROJECT_ID"]
    dataset = os.environ["BIGQUERY_DATASET_ID"]
    return f"{project}.{dataset}.{table}"


def _ref(table: str = TABLE_NAME) -> str:
    return f"`{_fqn(table)}`"


def build() -> dict[str, object]:
    client = _bq_tools._client()
    fqn = _fqn()
    now_iso = datetime.now(timezone.utc).isoformat()

    select_sql = f"""
    WITH src AS (
      SELECT competition_id, competition_name, competition_country,
             season_year, match_date, kickoff_at
      FROM {_ref('raw_fixtures')}
      WHERE competition_id IS NOT NULL
      UNION ALL
      SELECT competition_id, competition_name, competition_country,
             season_year, snapshot_date AS match_date,
             TIMESTAMP(snapshot_date) AS kickoff_at
      FROM {_ref('raw_standings')}
      WHERE competition_id IS NOT NULL
    ),
    seasons AS (
      SELECT competition_id, ARRAY_AGG(season_year ORDER BY season_year) AS seasons_observed
      FROM (
        SELECT DISTINCT competition_id, season_year
        FROM src WHERE season_year IS NOT NULL
      )
      GROUP BY competition_id
    ),
    aggregated AS (
      SELECT
        competition_id,
        ARRAY_AGG(competition_name IGNORE NULLS ORDER BY kickoff_at DESC LIMIT 1)[SAFE_OFFSET(0)] AS competition_name,
        ARRAY_AGG(competition_country IGNORE NULLS ORDER BY kickoff_at DESC LIMIT 1)[SAFE_OFFSET(0)] AS competition_country,
        MIN(match_date) AS first_seen_date,
        MAX(match_date) AS last_seen_date,
        COUNTIF(match_date IS NOT NULL) AS match_count
      FROM src
      GROUP BY competition_id
    )
    SELECT
      a.competition_id,
      a.competition_name,
      a.competition_country,
      (a.competition_id = 1) AS is_world_cup,
      IFNULL(s.seasons_observed, []) AS seasons_observed,
      a.first_seen_date,
      a.last_seen_date,
      a.match_count,
      TIMESTAMP('{now_iso}') AS built_at
    FROM aggregated a
    LEFT JOIN seasons s USING (competition_id)
    """

    table = bigquery.Table(fqn, schema=DIM_COMPETITION_SCHEMA)
    table.description = (
        "Canonical competition dimension. Grain: one row per competition_id. "
        "Built from raw_fixtures + raw_standings. Zero API calls."
    )
    table.clustering_fields = ["competition_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)

    job_config = bigquery.QueryJobConfig(
        destination=fqn,
        write_disposition="WRITE_TRUNCATE",
    )
    client.query(select_sql, job_config=job_config).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("dim_competition built: %d competitions", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
