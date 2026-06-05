"""mart_team_profile — agent-facing team profile.

Grain: one row per team_id.
Combines dim_team + lifetime aggregates from fact_match_team.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "mart_team_profile"


SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("team_id",                "INT64",  mode="REQUIRED",
                         description="Primary key. FK to dim_team.team_id."),
    bigquery.SchemaField("team_name",              "STRING", mode="REQUIRED",
                         description="Latest known team name."),
    bigquery.SchemaField("is_wc2026_participant",  "BOOL",   mode="REQUIRED",
                         description="True if team appears in the 2026 World Cup."),
    bigquery.SchemaField("first_match_date",       "DATE",   description="Earliest match observed for this team."),
    bigquery.SchemaField("last_match_date",        "DATE",   description="Most recent match observed for this team."),
    bigquery.SchemaField("matches_played",         "INT64",  description="Total completed matches."),
    bigquery.SchemaField("wins",                   "INT64",  description="Lifetime wins."),
    bigquery.SchemaField("draws",                  "INT64",  description="Lifetime draws."),
    bigquery.SchemaField("losses",                 "INT64",  description="Lifetime losses."),
    bigquery.SchemaField("win_pct",                "FLOAT64",description="wins / matches_played, 0-100."),
    bigquery.SchemaField("goals_for_total",        "INT64",  description="Lifetime goals scored."),
    bigquery.SchemaField("goals_against_total",    "INT64",  description="Lifetime goals conceded."),
    bigquery.SchemaField("goal_diff_total",        "INT64",  description="Lifetime goal difference."),
    bigquery.SchemaField("goals_for_per_match",    "FLOAT64",description="Average goals scored per completed match."),
    bigquery.SchemaField("goals_against_per_match","FLOAT64",description="Average goals conceded per completed match."),
    bigquery.SchemaField("clean_sheets",           "INT64",  description="Matches with 0 goals conceded."),
    bigquery.SchemaField("failed_to_score",        "INT64",  description="Completed matches with 0 goals scored."),
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
    WITH agg AS (
      SELECT
        team_id,
        MIN(match_date)                                AS first_match_date,
        MAX(match_date)                                AS last_match_date,
        COUNTIF(result IS NOT NULL)                    AS matches_played,
        COUNTIF(result = 'W')                          AS wins,
        COUNTIF(result = 'D')                          AS draws,
        COUNTIF(result = 'L')                          AS losses,
        SUM(IF(result IS NOT NULL, goals_for,     0))  AS goals_for_total,
        SUM(IF(result IS NOT NULL, goals_against, 0))  AS goals_against_total,
        COUNTIF(is_clean_sheet)                        AS clean_sheets,
        COUNTIF(result IS NOT NULL AND NOT did_score)  AS failed_to_score
      FROM {_ref('fact_match_team')}
      GROUP BY team_id
    )
    SELECT
      t.team_id,
      t.team_name,
      t.is_wc2026_participant,
      a.first_match_date,
      a.last_match_date,
      a.matches_played,
      a.wins, a.draws, a.losses,
      SAFE_DIVIDE(a.wins * 100.0, a.matches_played)             AS win_pct,
      a.goals_for_total,
      a.goals_against_total,
      (a.goals_for_total - a.goals_against_total)               AS goal_diff_total,
      SAFE_DIVIDE(a.goals_for_total,     a.matches_played)      AS goals_for_per_match,
      SAFE_DIVIDE(a.goals_against_total, a.matches_played)      AS goals_against_per_match,
      a.clean_sheets,
      a.failed_to_score,
      TIMESTAMP('{now_iso}') AS built_at
    FROM {_ref('dim_team')} t
    LEFT JOIN agg a USING (team_id)
    """

    table = bigquery.Table(fqn, schema=SCHEMA)
    table.description = "Agent-facing team profile. Grain: one row per team_id."
    table.clustering_fields = ["is_wc2026_participant", "team_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)
    client.query(sql, job_config=bigquery.QueryJobConfig(
        destination=fqn, write_disposition="WRITE_TRUNCATE")).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("mart_team_profile built: %d rows", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
