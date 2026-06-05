"""fact_match — canonical match fact.

Grain: one row per match_id.

Built from raw_fixtures (latest snapshot per match_id) joined to
dim_venue (via venue_key) for the canonical venue surrogate key.

Zero API calls.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "fact_match"


FACT_MATCH_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("match_id",            "INT64",     mode="REQUIRED",
                         description="Primary key. API-Football fixture ID."),
    bigquery.SchemaField("match_date",          "DATE",      mode="REQUIRED",
                         description="UTC calendar date of kickoff. Partition key."),
    bigquery.SchemaField("kickoff_at",          "TIMESTAMP", mode="REQUIRED",
                         description="UTC kickoff timestamp."),
    bigquery.SchemaField("season_year",         "INT64",
                         description="Season year as defined by the data provider."),
    bigquery.SchemaField("competition_id",      "INT64",
                         description="FK to dim_competition.competition_id."),
    bigquery.SchemaField("competition_name",    "STRING",    description="Denormalized competition name."),
    bigquery.SchemaField("competition_country", "STRING",    description="Denormalized competition country/confederation."),
    bigquery.SchemaField("competition_round",   "STRING",    description="Round / matchday label."),
    bigquery.SchemaField("home_team_id",        "INT64",
                         description="FK to dim_team.team_id for the home team."),
    bigquery.SchemaField("home_team_name",      "STRING",    description="Denormalized home team name."),
    bigquery.SchemaField("away_team_id",        "INT64",
                         description="FK to dim_team.team_id for the away team."),
    bigquery.SchemaField("away_team_name",      "STRING",    description="Denormalized away team name."),
    bigquery.SchemaField("venue_key",           "STRING",
                         description="FK to dim_venue.venue_key. Derived from venue_id when present, else name+city hash."),
    bigquery.SchemaField("venue_name",          "STRING",    description="Stadium name."),
    bigquery.SchemaField("venue_city",          "STRING",    description="City where the match was played."),
    bigquery.SchemaField("referee_name",        "STRING",    description="Referee name."),
    bigquery.SchemaField("match_status",        "STRING",
                         description="Normalized status enum: SCHEDULED, LIVE, FINISHED, POSTPONED, CANCELLED, ABANDONED, AWARDED."),
    bigquery.SchemaField("match_status_raw",    "STRING",    description="Raw status short code from API."),
    bigquery.SchemaField("is_completed",        "BOOL",      description="True if the match has reached a final result."),
    bigquery.SchemaField("home_goals",          "INT64",     description="Final home goals."),
    bigquery.SchemaField("away_goals",          "INT64",     description="Final away goals."),
    bigquery.SchemaField("home_goals_halftime", "INT64",     description="Home goals at half-time."),
    bigquery.SchemaField("away_goals_halftime", "INT64",     description="Away goals at half-time."),
    bigquery.SchemaField("home_goals_extratime","INT64",     description="Home goals scored during extra time only."),
    bigquery.SchemaField("away_goals_extratime","INT64",     description="Away goals scored during extra time only."),
    bigquery.SchemaField("home_goals_penalty",  "INT64",     description="Home penalty shootout goals."),
    bigquery.SchemaField("away_goals_penalty",  "INT64",     description="Away penalty shootout goals."),
    bigquery.SchemaField("has_extra_time",      "BOOL",      description="True if extra time was played."),
    bigquery.SchemaField("has_penalty_shootout","BOOL",      description="True if the match was decided by a shootout."),
    bigquery.SchemaField("winner_team_id",      "INT64",     description="FK to dim_team. NULL on draw or not completed."),
    bigquery.SchemaField("is_draw",             "BOOL",      description="True if the match ended as a draw in regular time."),
    bigquery.SchemaField("built_at",            "TIMESTAMP", mode="REQUIRED",
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
          ROW_NUMBER() OVER (PARTITION BY match_id ORDER BY ingested_at DESC) AS rn
        FROM {_ref('raw_fixtures')}
        WHERE match_id IS NOT NULL AND kickoff_at IS NOT NULL
      )
      WHERE rn = 1
    )
    SELECT
      l.match_id,
      l.match_date,
      l.kickoff_at,
      l.season_year,
      l.competition_id,
      l.competition_name,
      l.competition_country,
      l.competition_round,
      l.home_team_id,
      l.home_team_name,
      l.away_team_id,
      l.away_team_name,
      CASE
        WHEN l.venue_id IS NOT NULL THEN CONCAT('id:', CAST(l.venue_id AS STRING))
        WHEN l.venue_name IS NOT NULL OR l.venue_city IS NOT NULL THEN
          CONCAT('name:',
            TRIM(LOWER(IFNULL(l.venue_name, ''))), '|',
            TRIM(LOWER(IFNULL(l.venue_city, '')))
          )
        ELSE NULL
      END AS venue_key,
      l.venue_name,
      l.venue_city,
      l.referee_name,
      CASE
        WHEN UPPER(l.match_status_raw) IN ('FT','AET','PEN','AWD','WO')           THEN 'FINISHED'
        WHEN UPPER(l.match_status_raw) IN ('NS','TBD')                            THEN 'SCHEDULED'
        WHEN UPPER(l.match_status_raw) IN ('1H','2H','HT','ET','BT','P','LIVE','INT') THEN 'LIVE'
        WHEN UPPER(l.match_status_raw) IN ('PST','POSTPONED')                     THEN 'POSTPONED'
        WHEN UPPER(l.match_status_raw) IN ('CANC','CANCELLED')                    THEN 'CANCELLED'
        WHEN UPPER(l.match_status_raw) IN ('ABD','ABANDONED')                     THEN 'ABANDONED'
        ELSE 'UNKNOWN'
      END                                                            AS match_status,
      l.match_status_raw,
      l.is_completed,
      l.home_goals,
      l.away_goals,
      l.home_goals_halftime,
      l.away_goals_halftime,
      l.home_goals_extratime,
      l.away_goals_extratime,
      l.home_goals_penalty,
      l.away_goals_penalty,
      (l.home_goals_extratime IS NOT NULL OR l.away_goals_extratime IS NOT NULL
        OR l.match_status_raw IN ('AET','PEN'))                       AS has_extra_time,
      (l.home_goals_penalty IS NOT NULL OR l.away_goals_penalty IS NOT NULL
        OR l.match_status_raw = 'PEN')                                AS has_penalty_shootout,
      CASE
        WHEN NOT l.is_completed THEN NULL
        WHEN l.home_goals_penalty IS NOT NULL AND l.away_goals_penalty IS NOT NULL THEN
          CASE
            WHEN l.home_goals_penalty > l.away_goals_penalty THEN l.home_team_id
            WHEN l.away_goals_penalty > l.home_goals_penalty THEN l.away_team_id
            ELSE NULL
          END
        WHEN l.home_goals > l.away_goals THEN l.home_team_id
        WHEN l.away_goals > l.home_goals THEN l.away_team_id
        ELSE NULL
      END                                                              AS winner_team_id,
      (l.match_status_raw IN ('FT','AET') AND l.home_goals = l.away_goals
        AND l.home_goals_penalty IS NULL)                              AS is_draw,
      TIMESTAMP('{now_iso}')                                           AS built_at
    FROM latest l
    """

    table = bigquery.Table(fqn, schema=FACT_MATCH_SCHEMA)
    table.description = (
        "Canonical match fact. Grain: one row per match_id. "
        "Built from raw_fixtures (latest snapshot per match). App queries should read this."
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="match_date",
    )
    table.clustering_fields = ["competition_id", "home_team_id", "away_team_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)

    job_config = bigquery.QueryJobConfig(
        destination=fqn,
        write_disposition="WRITE_TRUNCATE",
    )
    client.query(select_sql, job_config=job_config).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("fact_match built: %d matches", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
