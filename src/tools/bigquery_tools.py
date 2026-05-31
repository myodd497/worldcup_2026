"""
BigQuery tools — upload a DataFrame and run SQL queries.
"""
from __future__ import annotations

import os
from typing import Literal

import pandas as pd
from google.cloud import bigquery

WriteDisposition = Literal["WRITE_APPEND", "WRITE_TRUNCATE", "WRITE_EMPTY"]


def _client() -> bigquery.Client:
    return bigquery.Client(project=os.environ["BIGQUERY_PROJECT_ID"])


def _table_ref(table_name: str) -> str:
    return f"{os.environ['BIGQUERY_PROJECT_ID']}.{os.environ['BIGQUERY_DATASET_ID']}.{table_name}"


def upload_dataframe(df: pd.DataFrame, table_name: str) -> int:
    """Uploads a DataFrame to BigQuery (WRITE_APPEND, schema auto-detected).
    Returns the number of rows uploaded."""
    client = _client()
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(df, _table_ref(table_name), job_config=job_config)
    job.result()
    return len(df)


def upload_dataframe_with_schema(
    df: pd.DataFrame,
    table_name: str,
    schema: list[bigquery.SchemaField],
    write_disposition: WriteDisposition = "WRITE_APPEND",
) -> int:
    """Uploads a DataFrame with an explicit BigQuery schema.

    Args:
        df: Data to upload.
        table_name: Destination table (dataset taken from env BIGQUERY_DATASET_ID).
        schema: List of bigquery.SchemaField definitions — enforces column types
                and attaches field-level descriptions to the BQ table.
        write_disposition: WRITE_APPEND (default), WRITE_TRUNCATE, or WRITE_EMPTY.

    Returns:
        Number of rows written.
    """
    client = _client()
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=write_disposition,
    )
    job = client.load_table_from_dataframe(df, _table_ref(table_name), job_config=job_config)
    job.result()
    return len(df)


def run_query(sql: str) -> pd.DataFrame:
    """Runs a SQL query and returns the result as a DataFrame."""
    client = _client()
    return client.query(sql).to_dataframe()


if __name__ == "__main__":
    df = run_query("SELECT 1 as test")
    print(df)
