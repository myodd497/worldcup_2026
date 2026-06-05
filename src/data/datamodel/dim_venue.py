"""dim_venue — canonical venue (stadium) dimension.

Grain: one row per stable venue key.

API-Football's venue_id is sometimes NULL on international matches. We therefore
use COALESCE(venue_id, hashed-name-city) as the surrogate key, exposed as venue_key.
The real API venue_id is preserved as venue_id_source.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "dim_venue"


DIM_VENUE_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("venue_key",        "STRING", mode="REQUIRED",
                         description="Primary key. Either 'id:<venue_id>' (when API provides one) or 'name:<normalized>' otherwise."),
    bigquery.SchemaField("venue_id_source",  "INT64",
                         description="Original API-Football venue ID. NULL when API did not provide one."),
    bigquery.SchemaField("venue_name",       "STRING",
                         description="Most recent venue name observed."),
    bigquery.SchemaField("venue_city",       "STRING",
                         description="Most recent city observed."),
    bigquery.SchemaField("first_seen_date",  "DATE", description="Earliest match_date this venue appears in raw_fixtures."),
    bigquery.SchemaField("last_seen_date",   "DATE", description="Latest match_date this venue appears in raw_fixtures."),
    bigquery.SchemaField("match_count",      "INT64",
                         description="Total matches played at this venue (from raw_fixtures)."),
    bigquery.SchemaField("built_at",         "TIMESTAMP", mode="REQUIRED",
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
      SELECT
        venue_id,
        TRIM(LOWER(IFNULL(venue_name, ''))) AS name_norm,
        TRIM(LOWER(IFNULL(venue_city, ''))) AS city_norm,
        venue_name, venue_city,
        match_date, kickoff_at
      FROM {_ref('raw_fixtures')}
      WHERE venue_name IS NOT NULL OR venue_id IS NOT NULL
    ),
    keyed AS (
      SELECT
        CASE
          WHEN venue_id IS NOT NULL THEN CONCAT('id:', CAST(venue_id AS STRING))
          ELSE CONCAT('name:', name_norm, '|', city_norm)
        END AS venue_key,
        venue_id, venue_name, venue_city, match_date, kickoff_at
      FROM src
    )
    SELECT
      venue_key,
      ARRAY_AGG(venue_id IGNORE NULLS ORDER BY kickoff_at DESC LIMIT 1)[SAFE_OFFSET(0)] AS venue_id_source,
      ARRAY_AGG(venue_name IGNORE NULLS ORDER BY kickoff_at DESC LIMIT 1)[SAFE_OFFSET(0)] AS venue_name,
      ARRAY_AGG(venue_city IGNORE NULLS ORDER BY kickoff_at DESC LIMIT 1)[SAFE_OFFSET(0)] AS venue_city,
      MIN(match_date) AS first_seen_date,
      MAX(match_date) AS last_seen_date,
      COUNT(*) AS match_count,
      TIMESTAMP('{now_iso}') AS built_at
    FROM keyed
    GROUP BY venue_key
    """

    table = bigquery.Table(fqn, schema=DIM_VENUE_SCHEMA)
    table.description = (
        "Canonical venue dimension. Grain: one row per venue_key. "
        "Built from raw_fixtures. Zero API calls."
    )
    table.clustering_fields = ["venue_key"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)

    job_config = bigquery.QueryJobConfig(
        destination=fqn,
        write_disposition="WRITE_TRUNCATE",
    )
    client.query(select_sql, job_config=job_config).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("dim_venue built: %d venues", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
