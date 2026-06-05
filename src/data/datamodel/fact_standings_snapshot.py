"""fact_standings_snapshot — canonical standings snapshot fact.

Grain: one row per (competition_id, season_year, team_id, snapshot_date).
Built from raw_standings (latest snapshot per grain key).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "fact_standings_snapshot"


FACT_STANDINGS_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("competition_id",     "INT64",  mode="REQUIRED",
                         description="FK to dim_competition.competition_id."),
    bigquery.SchemaField("competition_name",   "STRING", description="Denormalized competition name."),
    bigquery.SchemaField("season_year",        "INT64",  mode="REQUIRED",
                         description="Season year."),
    bigquery.SchemaField("team_id",            "INT64",  mode="REQUIRED",
                         description="FK to dim_team.team_id."),
    bigquery.SchemaField("team_name",          "STRING", description="Team name at snapshot time."),
    bigquery.SchemaField("group_name",         "STRING",
                         description="Group / phase label inside the competition."),
    bigquery.SchemaField("snapshot_date",      "DATE",   mode="REQUIRED",
                         description="UTC calendar date of the snapshot. Partition key."),
    bigquery.SchemaField("standing_rank",      "INT64",  description="Rank within the group at snapshot time."),
    bigquery.SchemaField("points",             "INT64",  description="Points."),
    bigquery.SchemaField("played",             "INT64",  description="Matches played."),
    bigquery.SchemaField("wins",               "INT64",  description="Total wins."),
    bigquery.SchemaField("draws",              "INT64",  description="Total draws."),
    bigquery.SchemaField("losses",             "INT64",  description="Total losses."),
    bigquery.SchemaField("goals_for",          "INT64",  description="Goals scored."),
    bigquery.SchemaField("goals_against",      "INT64",  description="Goals conceded."),
    bigquery.SchemaField("goal_diff",          "INT64",  description="Goal difference."),
    bigquery.SchemaField("form",               "STRING", description="Recent results string e.g. 'WWDLW'."),
    bigquery.SchemaField("standing_status",    "STRING", description="API status: same, up, down."),
    bigquery.SchemaField("standing_description","STRING",description="Free-text description e.g. 'Promotion - Champions League'."),
    bigquery.SchemaField("built_at",           "TIMESTAMP", mode="REQUIRED",
                         description="UTC timestamp this fact row was built."),
]


def _fqn(t: str = TABLE_NAME) -> str:
    p = os.environ["BIGQUERY_PROJECT_ID"]; d = os.environ["BIGQUERY_DATASET_ID"]
    return f"{p}.{d}.{t}"


def _ref(t: str = TABLE_NAME) -> str:
    return f"`{_fqn(t)}`"


def build() -> dict[str, object]:
    client = _bq_tools._client()
    fqn = _fqn()
    now_iso = datetime.now(timezone.utc).isoformat()

    select_sql = f"""
    WITH latest AS (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT *,
          ROW_NUMBER() OVER (
            PARTITION BY competition_id, season_year, team_id, snapshot_date
            ORDER BY ingested_at DESC
          ) AS rn
        FROM {_ref('raw_standings')}
        WHERE competition_id IS NOT NULL
          AND season_year   IS NOT NULL
          AND team_id       IS NOT NULL
          AND snapshot_date IS NOT NULL
      )
      WHERE rn = 1
    )
    SELECT
      competition_id,
      competition_name,
      season_year,
      team_id,
      team_name,
      group_name,
      snapshot_date,
      standing_rank,
      points,
      played,
      wins,
      draws,
      losses,
      goals_for,
      goals_against,
      goal_diff,
      form,
      standing_status,
      standing_description,
      TIMESTAMP('{now_iso}') AS built_at
    FROM latest
    """

    table = bigquery.Table(fqn, schema=FACT_STANDINGS_SCHEMA)
    table.description = (
        "Canonical standings snapshot fact. Grain: one row per "
        "(competition_id, season_year, team_id, snapshot_date). "
        "Built from raw_standings."
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="snapshot_date",
    )
    table.clustering_fields = ["competition_id", "season_year", "team_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)

    job_config = bigquery.QueryJobConfig(
        destination=fqn,
        write_disposition="WRITE_TRUNCATE",
    )
    client.query(select_sql, job_config=job_config).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("fact_standings_snapshot built: %d rows", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
