"""mart_team_form — rolling recent form per team.

Grain: one row per team_id (most-recent up to N=10 completed matches summarized).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "mart_team_form"


SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("team_id",                "INT64",  mode="REQUIRED",
                         description="FK to dim_team.team_id."),
    bigquery.SchemaField("team_name",              "STRING", description="Team name."),
    bigquery.SchemaField("is_wc2026_participant",  "BOOL",   description="True if WC 2026 team."),
    bigquery.SchemaField("last_match_date",        "DATE",   description="Most recent completed match date."),
    bigquery.SchemaField("recent_matches",         "INT64",  description="How many recent matches included (max 10)."),
    bigquery.SchemaField("recent_form_string",     "STRING", description="Concatenated WDL for last 5 (newest first), e.g. 'WWDLW'."),
    bigquery.SchemaField("last10_wins",            "INT64",  description="Wins in last 10."),
    bigquery.SchemaField("last10_draws",           "INT64",  description="Draws in last 10."),
    bigquery.SchemaField("last10_losses",          "INT64",  description="Losses in last 10."),
    bigquery.SchemaField("last10_points",          "INT64",  description="Points in last 10."),
    bigquery.SchemaField("last10_goals_for",       "INT64",  description="Goals scored in last 10."),
    bigquery.SchemaField("last10_goals_against",   "INT64",  description="Goals conceded in last 10."),
    bigquery.SchemaField("last10_goal_diff",       "INT64",  description="Goal difference in last 10."),
    bigquery.SchemaField("last10_clean_sheets",    "INT64",  description="Clean sheets in last 10."),
    bigquery.SchemaField("last10_failed_to_score", "INT64",  description="Failed-to-score matches in last 10."),
    bigquery.SchemaField("built_at",               "TIMESTAMP", mode="REQUIRED",
                         description="UTC timestamp this mart row was built."),
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

    sql = f"""
    WITH ranked AS (
      SELECT
        team_id, match_id, match_date, result, points,
        goals_for, goals_against, is_clean_sheet, did_score,
        ROW_NUMBER() OVER (
          PARTITION BY team_id ORDER BY match_date DESC, match_id DESC
        ) AS rn
      FROM {_ref('fact_match_team')}
      WHERE result IS NOT NULL
    ),
    last10 AS (
      SELECT
        team_id,
        MAX(match_date)              AS last_match_date,
        COUNT(*)                     AS recent_matches,
        COUNTIF(result = 'W')        AS last10_wins,
        COUNTIF(result = 'D')        AS last10_draws,
        COUNTIF(result = 'L')        AS last10_losses,
        SUM(points)                  AS last10_points,
        SUM(goals_for)               AS last10_goals_for,
        SUM(goals_against)           AS last10_goals_against,
        COUNTIF(is_clean_sheet)      AS last10_clean_sheets,
        COUNTIF(NOT did_score)       AS last10_failed_to_score
      FROM ranked
      WHERE rn <= 10
      GROUP BY team_id
    ),
    last5_string AS (
      SELECT
        team_id,
        STRING_AGG(result, '' ORDER BY rn) AS recent_form_string
      FROM ranked
      WHERE rn <= 5
      GROUP BY team_id
    )
    SELECT
      t.team_id,
      t.team_name,
      t.is_wc2026_participant,
      l.last_match_date,
      l.recent_matches,
      f.recent_form_string,
      l.last10_wins, l.last10_draws, l.last10_losses,
      l.last10_points,
      l.last10_goals_for, l.last10_goals_against,
      (l.last10_goals_for - l.last10_goals_against) AS last10_goal_diff,
      l.last10_clean_sheets,
      l.last10_failed_to_score,
      TIMESTAMP('{now_iso}') AS built_at
    FROM {_ref('dim_team')} t
    LEFT JOIN last10       l USING (team_id)
    LEFT JOIN last5_string f USING (team_id)
    """

    table = bigquery.Table(fqn, schema=SCHEMA)
    table.description = "Agent-facing rolling team form over the last 10 completed matches."
    table.clustering_fields = ["is_wc2026_participant", "team_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)
    client.query(sql, job_config=bigquery.QueryJobConfig(
        destination=fqn, write_disposition="WRITE_TRUNCATE")).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("mart_team_form built: %d rows", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
