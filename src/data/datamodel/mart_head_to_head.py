"""mart_head_to_head — pairwise team aggregates.

Grain: one row per unordered pair (team_lo, team_hi) where team_lo < team_hi,
considering only completed matches.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "mart_head_to_head"


SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("team_lo_id",        "INT64",  mode="REQUIRED",
                         description="FK to dim_team. The team with the smaller team_id in the pair."),
    bigquery.SchemaField("team_lo_name",      "STRING", description="Name of team_lo."),
    bigquery.SchemaField("team_hi_id",        "INT64",  mode="REQUIRED",
                         description="FK to dim_team. The team with the larger team_id in the pair."),
    bigquery.SchemaField("team_hi_name",      "STRING", description="Name of team_hi."),
    bigquery.SchemaField("matches_played",    "INT64",  description="Completed meetings between the two teams."),
    bigquery.SchemaField("team_lo_wins",      "INT64",  description="Wins by team_lo across all meetings."),
    bigquery.SchemaField("team_hi_wins",      "INT64",  description="Wins by team_hi across all meetings."),
    bigquery.SchemaField("draws",             "INT64",  description="Draws (regular-time)."),
    bigquery.SchemaField("team_lo_goals",     "INT64",  description="Total goals scored by team_lo across meetings."),
    bigquery.SchemaField("team_hi_goals",     "INT64",  description="Total goals scored by team_hi across meetings."),
    bigquery.SchemaField("first_meeting_date","DATE",   description="Date of the earliest completed meeting."),
    bigquery.SchemaField("last_meeting_date", "DATE",   description="Date of the most recent completed meeting."),
    bigquery.SchemaField("last_meeting_match_id", "INT64", description="match_id of the most recent meeting."),
    bigquery.SchemaField("last_meeting_winner_team_id", "INT64",
                         description="Winner team_id of the most recent meeting. NULL on draw."),
    bigquery.SchemaField("built_at",          "TIMESTAMP", mode="REQUIRED",
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
    WITH pairs AS (
      SELECT
        LEAST(home_team_id, away_team_id)    AS team_lo_id,
        GREATEST(home_team_id, away_team_id) AS team_hi_id,
        match_id, match_date, winner_team_id, is_draw,
        IF(home_team_id = LEAST(home_team_id, away_team_id), home_goals, away_goals) AS lo_goals,
        IF(home_team_id = LEAST(home_team_id, away_team_id), away_goals, home_goals) AS hi_goals
      FROM {_ref('fact_match')}
      WHERE is_completed
        AND home_team_id IS NOT NULL
        AND away_team_id IS NOT NULL
    ),
    agg AS (
      SELECT
        team_lo_id, team_hi_id,
        COUNT(*)                                   AS matches_played,
        COUNTIF(winner_team_id = team_lo_id)       AS team_lo_wins,
        COUNTIF(winner_team_id = team_hi_id)       AS team_hi_wins,
        COUNTIF(is_draw)                           AS draws,
        SUM(lo_goals)                              AS team_lo_goals,
        SUM(hi_goals)                              AS team_hi_goals,
        MIN(match_date)                            AS first_meeting_date,
        MAX(match_date)                            AS last_meeting_date
      FROM pairs
      GROUP BY team_lo_id, team_hi_id
    ),
    last_meeting AS (
      SELECT * EXCEPT(rn) FROM (
        SELECT
          team_lo_id, team_hi_id, match_id, match_date, winner_team_id,
          ROW_NUMBER() OVER (
            PARTITION BY team_lo_id, team_hi_id
            ORDER BY match_date DESC, match_id DESC
          ) AS rn
        FROM pairs
      ) WHERE rn = 1
    )
    SELECT
      a.team_lo_id,
      tl.team_name AS team_lo_name,
      a.team_hi_id,
      th.team_name AS team_hi_name,
      a.matches_played,
      a.team_lo_wins, a.team_hi_wins, a.draws,
      a.team_lo_goals, a.team_hi_goals,
      a.first_meeting_date,
      a.last_meeting_date,
      lm.match_id        AS last_meeting_match_id,
      lm.winner_team_id  AS last_meeting_winner_team_id,
      TIMESTAMP('{now_iso}') AS built_at
    FROM agg a
    LEFT JOIN {_ref('dim_team')} tl ON tl.team_id = a.team_lo_id
    LEFT JOIN {_ref('dim_team')} th ON th.team_id = a.team_hi_id
    LEFT JOIN last_meeting lm USING (team_lo_id, team_hi_id)
    """

    table = bigquery.Table(fqn, schema=SCHEMA)
    table.description = "Agent-facing head-to-head aggregates per unordered team pair."
    table.clustering_fields = ["team_lo_id", "team_hi_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)
    client.query(sql, job_config=bigquery.QueryJobConfig(
        destination=fqn, write_disposition="WRITE_TRUNCATE")).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("mart_head_to_head built: %d rows", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
