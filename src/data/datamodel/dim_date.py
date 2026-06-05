"""dim_date — canonical calendar date dimension.

Grain: one row per calendar date.

Covers MIN(match_date) in raw_fixtures - 7 days through MAX(match_date) + 365 days
so future scheduled matches and analysis windows always have a dim row.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from google.cloud import bigquery

from src.tools import bigquery_tools as _bq_tools

logger = logging.getLogger(__name__)

TABLE_NAME = "dim_date"


DIM_DATE_SCHEMA: list[bigquery.SchemaField] = [
    bigquery.SchemaField("calendar_date",  "DATE",   mode="REQUIRED",
                         description="Primary key. The calendar date in UTC."),
    bigquery.SchemaField("day_of_week",    "INT64",
                         description="Day of week (1=Sunday ... 7=Saturday) per BigQuery EXTRACT(DAYOFWEEK)."),
    bigquery.SchemaField("day_name",       "STRING",
                         description="Day name in English: Sunday, Monday, ..."),
    bigquery.SchemaField("is_weekend",     "BOOL",
                         description="True if day_of_week is Saturday or Sunday."),
    bigquery.SchemaField("month",          "INT64",  description="Month 1-12."),
    bigquery.SchemaField("month_name",     "STRING", description="Month name in English."),
    bigquery.SchemaField("quarter",        "INT64",  description="Quarter 1-4."),
    bigquery.SchemaField("year",           "INT64",  description="Year (Gregorian)."),
    bigquery.SchemaField("year_month",     "STRING", description="ISO YYYY-MM string."),
    bigquery.SchemaField("iso_week",       "INT64",  description="ISO week number."),
    bigquery.SchemaField("built_at",       "TIMESTAMP", mode="REQUIRED",
                         description="UTC timestamp this dimension row was built."),
]

_DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
_MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]


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

    day_names_sql = ",".join(f"'{n}'" for n in _DAY_NAMES)
    month_names_sql = ",".join(f"'{n}'" for n in _MONTH_NAMES)

    select_sql = f"""
    WITH bounds AS (
      SELECT
        DATE_SUB(MIN(match_date), INTERVAL 7 DAY)   AS start_date,
        DATE_ADD(MAX(match_date), INTERVAL 365 DAY) AS end_date
      FROM {_ref('raw_fixtures')}
      WHERE match_date IS NOT NULL
    ),
    cal AS (
      SELECT calendar_date
      FROM bounds, UNNEST(GENERATE_DATE_ARRAY(start_date, end_date, INTERVAL 1 DAY)) AS calendar_date
    ),
    names AS (
      SELECT
        [{day_names_sql}]   AS day_names,
        [{month_names_sql}] AS month_names
    )
    SELECT
      c.calendar_date,
      EXTRACT(DAYOFWEEK FROM c.calendar_date)                       AS day_of_week,
      n.day_names[OFFSET(EXTRACT(DAYOFWEEK FROM c.calendar_date) - 1)]   AS day_name,
      EXTRACT(DAYOFWEEK FROM c.calendar_date) IN (1, 7)             AS is_weekend,
      EXTRACT(MONTH FROM c.calendar_date)                           AS month,
      n.month_names[OFFSET(EXTRACT(MONTH FROM c.calendar_date) - 1)] AS month_name,
      EXTRACT(QUARTER FROM c.calendar_date)                         AS quarter,
      EXTRACT(YEAR FROM c.calendar_date)                            AS year,
      FORMAT_DATE('%Y-%m', c.calendar_date)                         AS year_month,
      EXTRACT(ISOWEEK FROM c.calendar_date)                         AS iso_week,
      TIMESTAMP('{now_iso}')                                        AS built_at
    FROM cal c
    CROSS JOIN names n
    """

    table = bigquery.Table(fqn, schema=DIM_DATE_SCHEMA)
    table.description = (
        "Canonical calendar date dimension. Grain: one row per calendar_date. "
        "Range: MIN(raw_fixtures.match_date) - 7 days .. MAX + 365 days. Zero API calls."
    )
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.YEAR,
        field="calendar_date",
    )
    table.clustering_fields = ["calendar_date"]
    client.delete_table(fqn, not_found_ok=True)
    client.create_table(table)

    job_config = bigquery.QueryJobConfig(
        destination=fqn,
        write_disposition="WRITE_TRUNCATE",
    )
    client.query(select_sql, job_config=job_config).result()
    rows = int(client.get_table(fqn).num_rows)
    logger.info("dim_date built: %d dates", rows)
    return {"built": True, "rows": rows}


def run() -> dict[str, object]:
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(run())
