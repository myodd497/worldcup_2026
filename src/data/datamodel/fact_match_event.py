"""fact_match_event — canonical match event fact.

Grain: one row per (match_id, event_seq).
Built from raw_fixture_events (latest snapshot per (match_id, event_seq)).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "fact_match_event"


FACT_MATCH_EVENT_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("match_id",        "INT64",     mode="REQUIRED",
                         description="FK to fact_match.match_id."),
    bigquery.SchemaField("event_seq",       "INT64",     mode="REQUIRED",
                         description="1-based event order within the match."),
    bigquery.SchemaField("match_date",      "DATE",
                         description="UTC calendar date of the match. Partition key."),
    bigquery.SchemaField("minute",          "INT64",     description="Regulation minute (0-90+)."),
    bigquery.SchemaField("minute_extra",    "INT64",     description="Stoppage / extra minute offset. NULL if none."),
    bigquery.SchemaField("team_id",         "INT64",     description="FK to dim_team. Team responsible for / affected by the event."),
    bigquery.SchemaField("team_name",       "STRING",    description="Team name at match time."),
    bigquery.SchemaField("player_id",       "INT64",     description="Primary player (scorer, fouler, subbed-off)."),
    bigquery.SchemaField("player_name",     "STRING",    description="Primary player name."),
    bigquery.SchemaField("assist_id",       "INT64",     description="Secondary player (assister, subbed-on)."),
    bigquery.SchemaField("assist_name",     "STRING",    description="Secondary player name."),
    bigquery.SchemaField("event_type",      "STRING",    description="API event type: Goal, Card, subst, Var."),
    bigquery.SchemaField("event_detail",    "STRING",    description="Sub-type: Normal Goal, Penalty, Yellow Card, Substitution 1, etc."),
    bigquery.SchemaField("event_comments",  "STRING",    description="Free-text comments from API (often NULL)."),
    bigquery.SchemaField("is_goal",         "BOOL",      description="True for any Goal event (incl. penalty, own goal)."),
    bigquery.SchemaField("is_own_goal",     "BOOL",      description="True if event_detail = 'Own Goal'."),
    bigquery.SchemaField("is_penalty_goal", "BOOL",      description="True if event_detail = 'Penalty'."),
    bigquery.SchemaField("is_yellow_card",  "BOOL",      description="True if event_detail = 'Yellow Card'."),
    bigquery.SchemaField("is_red_card",     "BOOL",      description="True if event_detail = 'Red Card'."),
    bigquery.SchemaField("is_substitution", "BOOL",      description="True for any substitution event."),
    bigquery.SchemaField("built_at",        "TIMESTAMP", mode="REQUIRED",
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
            PARTITION BY match_id, event_seq
            ORDER BY ingested_at DESC
          ) AS rn
        FROM {_ref('raw_fixture_events')}
        WHERE match_id IS NOT NULL AND event_seq IS NOT NULL
      )
      WHERE rn = 1
    )
    SELECT
      match_id,
      event_seq,
      match_date,
      minute,
      minute_extra,
      team_id,
      team_name,
      player_id,
      player_name,
      assist_id,
      assist_name,
      event_type,
      event_detail,
      event_comments,
      (event_type = 'Goal')                        AS is_goal,
      (event_detail = 'Own Goal')                  AS is_own_goal,
      (event_detail = 'Penalty' AND event_type = 'Goal') AS is_penalty_goal,
      (event_detail = 'Yellow Card')               AS is_yellow_card,
      (event_detail = 'Red Card')                  AS is_red_card,
      (event_type = 'subst')                       AS is_substitution,
      TIMESTAMP('{now_iso}')                       AS built_at
    FROM latest
    """

    table = bigquery.Table(fqn, schema=FACT_MATCH_EVENT_SCHEMA)
    table.description = (
        "Canonical match event fact. Grain: one row per (match_id, event_seq). "
        "Built from raw_fixture_events."
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="match_date",
    )
    table.clustering_fields = ["match_id", "team_id"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)

    job_config = bigquery.QueryJobConfig(
        destination=fqn,
        write_disposition="WRITE_TRUNCATE",
    )
    client.query(select_sql, job_config=job_config).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("fact_match_event built: %d events", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
