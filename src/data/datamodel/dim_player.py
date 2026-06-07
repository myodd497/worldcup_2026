"""dim_player — canonical player dimension.

Grain: one row per player_id.

Built from raw_player_stats (zero API calls). Pulls player_ids and names
from the raw layer and picks the most recent non-NULL values.

is_wc2026_participant is resolved from fact_match: any player appearing
in a match with competition_id=1 AND season_year=2026 is flagged TRUE.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "dim_player"

DIM_PLAYER_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("player_id",              "INT64",  mode="REQUIRED",
                         description="Primary key. API-Football player ID."),
    bigquery.SchemaField("player_name",            "STRING",
                         description="Most recent name observed for this player."),
    bigquery.SchemaField("primary_position",       "STRING",
                         description="Most frequent position: G, D, M, F."),
    bigquery.SchemaField("primary_team_id",        "INT64",
                         description="FK to dim_team. Most recent team this player appeared for."),
    bigquery.SchemaField("primary_team_name",      "STRING",
                         description="Denormalized primary team name."),
    bigquery.SchemaField("is_wc2026_participant",  "BOOL",
                         description="True if this player appeared in a WC2026 match."),
    bigquery.SchemaField("is_goalkeeper",           "BOOL",
                         description="True if primary position is G."),
    bigquery.SchemaField("first_seen_date",        "DATE",
                         description="Earliest match_date in raw_player_stats."),
    bigquery.SchemaField("last_seen_date",         "DATE",
                         description="Latest match_date in raw_player_stats."),
    bigquery.SchemaField("total_matches_played",   "INT64",
                         description="Total matches this player appears in (all competitions)."),
    bigquery.SchemaField("total_minutes_played",   "INT64",
                         description="Total minutes played across all matches."),
    bigquery.SchemaField("total_goals",            "INT64",
                         description="Total goals scored across all matches."),
    bigquery.SchemaField("total_assists",           "INT64",
                         description="Total assists across all matches."),
    bigquery.SchemaField("built_at",               "TIMESTAMP", mode="REQUIRED",
                         description="UTC timestamp this dimension row was built."),
]


def _fqn(table: str = TABLE_NAME) -> str:
    project = os.environ["BIGQUERY_PROJECT_ID"]
    dataset = os.environ["BIGQUERY_DATASET_ID"]
    return f"{project}.{dataset}.{table}"


def _ref(table: str = TABLE_NAME) -> str:
    return f"`{_fqn(table)}`"


def build() -> dict[str, object]:
    """Builds dim_player via CREATE OR REPLACE. Idempotent. Zero API calls."""
    client = _bq_tools._client()
    fqn = _fqn()
    now_iso = datetime.now(timezone.utc).isoformat()

    select_sql = f"""
    WITH latest_stats AS (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT *,
          ROW_NUMBER() OVER (
            PARTITION BY match_id, team_id, player_id
            ORDER BY ingested_at DESC
          ) AS rn
        FROM {_ref('raw_player_stats')}
        WHERE player_id IS NOT NULL
      )
      WHERE rn = 1
    ),
    player_aggregates AS (
      SELECT
        player_id,
        ARRAY_AGG(player_name IGNORE NULLS ORDER BY ingested_at DESC LIMIT 1)[SAFE_OFFSET(0)] AS player_name,
        -- Most frequent position
        ARRAY_AGG(player_position IGNORE NULLS ORDER BY
          CASE player_position
            WHEN 'G' THEN 1 WHEN 'D' THEN 2 WHEN 'M' THEN 3 WHEN 'F' THEN 4 ELSE 5
          END
          LIMIT 1)[SAFE_OFFSET(0)] AS primary_position,
        -- Most recent team
        ARRAY_AGG(team_id IGNORE NULLS ORDER BY match_date DESC LIMIT 1)[SAFE_OFFSET(0)] AS primary_team_id,
        ARRAY_AGG(team_name IGNORE NULLS ORDER BY match_date DESC LIMIT 1)[SAFE_OFFSET(0)] AS primary_team_name,
        MIN(match_date) AS first_seen_date,
        MAX(match_date) AS last_seen_date,
        COUNT(DISTINCT match_id) AS total_matches_played,
        SUM(minutes_played) AS total_minutes_played,
        SUM(goals) AS total_goals,
        SUM(assists) AS total_assists
      FROM latest_stats
      GROUP BY player_id
    ),
    wc2026_players AS (
      SELECT DISTINCT rps.player_id
      FROM {_ref('raw_player_stats')} rps
      JOIN {_ref('fact_match')} fm ON fm.match_id = rps.match_id
      WHERE fm.competition_id = 1 AND fm.season_year = 2026
    )
    SELECT
      pa.player_id,
      pa.player_name,
      pa.primary_position,
      pa.primary_team_id,
      pa.primary_team_name,
      (wc.player_id IS NOT NULL) AS is_wc2026_participant,
      (pa.primary_position = 'G') AS is_goalkeeper,
      pa.first_seen_date,
      pa.last_seen_date,
      pa.total_matches_played,
      pa.total_minutes_played,
      pa.total_goals,
      pa.total_assists,
      TIMESTAMP('{now_iso}') AS built_at
    FROM player_aggregates pa
    LEFT JOIN wc2026_players wc ON wc.player_id = pa.player_id
    """

    table = bigquery.Table(fqn, schema=DIM_PLAYER_SCHEMA)
    table.description = (
        "Canonical player dimension. Grain: one row per player_id. "
        "Built from raw_player_stats. Zero API calls. "
        "is_wc2026_participant set from fact_match (competition_id=1, season_year=2026)."
    )
    table.clustering_fields = ["player_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)

    job_config = bigquery.QueryJobConfig(
        destination=fqn,
        write_disposition="WRITE_TRUNCATE",
    )
    client.query(select_sql, job_config=job_config).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("dim_player built: %d players", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
