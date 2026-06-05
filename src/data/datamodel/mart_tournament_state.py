"""mart_tournament_state — current state of a tournament (per team).

Grain: one row per (competition_id, season_year, team_id).
Combines latest standings snapshot with next/previous fixture and form.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "mart_tournament_state"


SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("competition_id",       "INT64",  mode="REQUIRED",
                         description="FK to dim_competition."),
    bigquery.SchemaField("competition_name",     "STRING", description="Competition name."),
    bigquery.SchemaField("season_year",          "INT64",  mode="REQUIRED",
                         description="Season year."),
    bigquery.SchemaField("team_id",              "INT64",  mode="REQUIRED",
                         description="FK to dim_team."),
    bigquery.SchemaField("team_name",            "STRING", description="Team name."),
    bigquery.SchemaField("group_name",           "STRING", description="Group / phase label."),
    bigquery.SchemaField("snapshot_date",        "DATE",   description="Date of the latest standings snapshot."),
    bigquery.SchemaField("standing_rank",        "INT64",  description="Latest rank in group."),
    bigquery.SchemaField("points",               "INT64",  description="Latest points."),
    bigquery.SchemaField("played",               "INT64",  description="Matches played in this tournament."),
    bigquery.SchemaField("wins",                 "INT64",  description="Wins in this tournament."),
    bigquery.SchemaField("draws",                "INT64",  description="Draws in this tournament."),
    bigquery.SchemaField("losses",               "INT64",  description="Losses in this tournament."),
    bigquery.SchemaField("goals_for",            "INT64",  description="Goals scored in this tournament."),
    bigquery.SchemaField("goals_against",        "INT64",  description="Goals conceded in this tournament."),
    bigquery.SchemaField("goal_diff",            "INT64",  description="Goal difference in this tournament."),
    bigquery.SchemaField("standing_description", "STRING", description="Free-text qualification description."),
    bigquery.SchemaField("next_match_id",        "INT64",  description="match_id of the team's next match in this competition/season."),
    bigquery.SchemaField("next_match_date",      "DATE",   description="Date of next match."),
    bigquery.SchemaField("next_opponent_team_id","INT64",  description="Opponent in the next match."),
    bigquery.SchemaField("next_opponent_name",   "STRING", description="Opponent name in the next match."),
    bigquery.SchemaField("last_match_id",        "INT64",  description="match_id of the team's most recent completed match in this competition/season."),
    bigquery.SchemaField("last_match_date",      "DATE",   description="Date of last completed match."),
    bigquery.SchemaField("last_match_result",    "STRING", description="W / D / L of the most recent completed match."),
    bigquery.SchemaField("built_at",             "TIMESTAMP", mode="REQUIRED",
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
    WITH latest_standings AS (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT *,
          ROW_NUMBER() OVER (
            PARTITION BY competition_id, season_year, team_id
            ORDER BY snapshot_date DESC
          ) AS rn
        FROM {_ref('fact_standings_snapshot')}
      )
      WHERE rn = 1
    ),
    -- Next upcoming match per (competition_id, season_year, team_id)
    next_match AS (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT
          fmt.competition_id, fmt.season_year, fmt.team_id,
          fmt.match_id   AS next_match_id,
          fmt.match_date AS next_match_date,
          fmt.opponent_team_id   AS next_opponent_team_id,
          fmt.opponent_team_name AS next_opponent_name,
          ROW_NUMBER() OVER (
            PARTITION BY fmt.competition_id, fmt.season_year, fmt.team_id
            ORDER BY fmt.match_date ASC, fmt.match_id ASC
          ) AS rn
        FROM {_ref('fact_match_team')} fmt
        WHERE fmt.match_date >= CURRENT_DATE()
          AND fmt.result IS NULL
      )
      WHERE rn = 1
    ),
    -- Last completed match per (competition_id, season_year, team_id)
    last_match AS (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT
          fmt.competition_id, fmt.season_year, fmt.team_id,
          fmt.match_id   AS last_match_id,
          fmt.match_date AS last_match_date,
          fmt.result     AS last_match_result,
          ROW_NUMBER() OVER (
            PARTITION BY fmt.competition_id, fmt.season_year, fmt.team_id
            ORDER BY fmt.match_date DESC, fmt.match_id DESC
          ) AS rn
        FROM {_ref('fact_match_team')} fmt
        WHERE fmt.result IS NOT NULL
      )
      WHERE rn = 1
    )
    SELECT
      s.competition_id,
      s.competition_name,
      s.season_year,
      s.team_id,
      s.team_name,
      s.group_name,
      s.snapshot_date,
      s.standing_rank,
      s.points,
      s.played, s.wins, s.draws, s.losses,
      s.goals_for, s.goals_against, s.goal_diff,
      s.standing_description,
      n.next_match_id, n.next_match_date,
      n.next_opponent_team_id, n.next_opponent_name,
      l.last_match_id, l.last_match_date, l.last_match_result,
      TIMESTAMP('{now_iso}') AS built_at
    FROM latest_standings s
    LEFT JOIN next_match n
      USING (competition_id, season_year, team_id)
    LEFT JOIN last_match l
      USING (competition_id, season_year, team_id)
    """

    table = bigquery.Table(fqn, schema=SCHEMA)
    table.description = "Agent-facing tournament state: latest standings + next/last match per team."
    table.clustering_fields = ["competition_id", "season_year", "team_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)
    client.query(sql, job_config=bigquery.QueryJobConfig(
        destination=fqn, write_disposition="WRITE_TRUNCATE")).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("mart_tournament_state built: %d rows", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
