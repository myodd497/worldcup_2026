"""fact_player_match_stat — canonical player match stat fact.

Grain: one row per (match_id, team_id, player_id).

Built from raw_player_stats (latest snapshot per (match_id, team_id, player_id)).
Deduplicates by ingested_at DESC — newest record wins.

Zero API calls.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "fact_player_match_stat"


FACT_PLAYER_MATCH_STAT_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("match_id",          "INT64",     mode="REQUIRED",
                         description="FK to fact_match.match_id."),
    bigquery.SchemaField("match_date",        "DATE",      mode="REQUIRED",
                         description="UTC calendar date of the match. Partition key."),
    bigquery.SchemaField("season_year",       "INT64",
                         description="Season year (denormalized from fact_match)."),
    bigquery.SchemaField("competition_id",    "INT64",
                         description="FK to dim_competition.competition_id."),
    bigquery.SchemaField("competition_name",  "STRING",
                         description="Denormalized competition name."),
    bigquery.SchemaField("team_id",           "INT64",     mode="REQUIRED",
                         description="FK to dim_team.team_id."),
    bigquery.SchemaField("team_name",         "STRING",
                         description="Team name at match time."),
    bigquery.SchemaField("player_id",         "INT64",     mode="REQUIRED",
                         description="FK to dim_player.player_id."),
    bigquery.SchemaField("player_name",       "STRING",
                         description="Player name at match time."),
    bigquery.SchemaField("player_position",   "STRING",
                         description="Position: G, D, M, F."),
    bigquery.SchemaField("is_starter",        "BOOL",
                         description="True if in starting XI."),
    bigquery.SchemaField("minutes_played",    "INT64",
                         description="Minutes played."),
    bigquery.SchemaField("goals",             "INT64",
                         description="Goals scored."),
    bigquery.SchemaField("assists",           "INT64",
                         description="Assists provided."),
    bigquery.SchemaField("goal_contributions","INT64",
                         description="goals + assists."),
    bigquery.SchemaField("shots_total",       "INT64",
                         description="Total shots."),
    bigquery.SchemaField("shots_on_target",   "INT64",
                         description="Shots on target."),
    bigquery.SchemaField("passes_total",      "INT64",
                         description="Total passes."),
    bigquery.SchemaField("passes_accurate",   "INT64",
                         description="Accurate passes."),
    bigquery.SchemaField("passes_key",        "INT64",
                         description="Key passes."),
    bigquery.SchemaField("pass_accuracy_pct", "FLOAT64",
                         description="SAFE_DIVIDE(passes_accurate, passes_total) * 100."),
    bigquery.SchemaField("dribbles_total",    "INT64",
                         description="Dribble attempts."),
    bigquery.SchemaField("dribbles_success",  "INT64",
                         description="Successful dribbles."),
    bigquery.SchemaField("tackles_total",     "INT64",
                         description="Total tackles."),
    bigquery.SchemaField("interceptions",     "INT64",
                         description="Interceptions."),
    bigquery.SchemaField("fouls_committed",   "INT64",
                         description="Fouls committed."),
    bigquery.SchemaField("fouls_drawn",       "INT64",
                         description="Fouls drawn."),
    bigquery.SchemaField("yellow_cards",      "INT64",
                         description="Yellow cards."),
    bigquery.SchemaField("red_cards",         "INT64",
                         description="Red cards."),
    bigquery.SchemaField("penalty_scored",    "INT64",
                         description="Penalties scored."),
    bigquery.SchemaField("penalty_missed",    "INT64",
                         description="Penalties missed."),
    bigquery.SchemaField("saves",             "INT64",
                         description="Goalkeeper saves."),
    bigquery.SchemaField("goals_conceded",    "INT64",
                         description="Goals conceded (goalkeeper)."),
    bigquery.SchemaField("rating",            "FLOAT64",
                         description="Player rating (0-10 scale)."),
    bigquery.SchemaField("built_at",          "TIMESTAMP", mode="REQUIRED",
                         description="UTC timestamp this fact row was built."),
]


def _fqn(t: str = TABLE_NAME) -> str:
    p = os.environ["BIGQUERY_PROJECT_ID"]; d = os.environ["BIGQUERY_DATASET_ID"]
    return f"{p}.{d}.{t}"


def _ref(t: str = TABLE_NAME) -> str:
    return f"`{_fqn(t)}`"


def build() -> dict[str, object]:
    """Builds fact_player_match_stat via CREATE OR REPLACE. Idempotent."""
    client = _bq_tools._client()
    fqn = _fqn()
    now_iso = datetime.now(timezone.utc).isoformat()

    select_sql = f"""
    WITH latest AS (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT *,
          ROW_NUMBER() OVER (
            PARTITION BY match_id, team_id, player_id
            ORDER BY ingested_at DESC
          ) AS rn
        FROM {_ref('raw_player_stats')}
        WHERE match_id IS NOT NULL AND team_id IS NOT NULL AND player_id IS NOT NULL
      )
      WHERE rn = 1
    ),
    enriched AS (
      SELECT
        l.match_id,
        l.match_date,
        fm.season_year,
        fm.competition_id,
        fm.competition_name,
        l.team_id,
        l.team_name,
        l.player_id,
        l.player_name,
        l.player_position,
        l.is_starter,
        l.minutes_played,
        l.goals,
        l.assists,
        (l.goals + l.assists) AS goal_contributions,
        l.shots_total,
        l.shots_on_target,
        l.passes_total,
        l.passes_accurate,
        l.passes_key,
        SAFE_DIVIDE(l.passes_accurate, l.passes_total) * 100 AS pass_accuracy_pct,
        l.dribbles_total,
        l.dribbles_success,
        l.tackles_total,
        l.interceptions,
        l.fouls_committed,
        l.fouls_drawn,
        l.yellow_cards,
        l.red_cards,
        l.penalty_scored,
        l.penalty_missed,
        l.saves,
        l.goals_conceded,
        l.rating
      FROM latest l
      LEFT JOIN {_ref('fact_match')} fm ON fm.match_id = l.match_id
    )
    SELECT
      *,
      TIMESTAMP('{now_iso}') AS built_at
    FROM enriched
    """

    table = bigquery.Table(fqn, schema=FACT_PLAYER_MATCH_STAT_SCHEMA)
    table.description = (
        "Canonical player match stat fact. Grain: one row per (match_id, team_id, player_id). "
        "Built from raw_player_stats (latest snapshot). Zero API calls."
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="match_date",
    )
    table.clustering_fields = ["player_id", "team_id", "competition_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)

    job_config = bigquery.QueryJobConfig(
        destination=fqn,
        write_disposition="WRITE_TRUNCATE",
    )
    client.query(select_sql, job_config=job_config).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("fact_player_match_stat built: %d rows", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
