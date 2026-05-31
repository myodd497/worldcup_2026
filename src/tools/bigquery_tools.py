"""
BigQuery tools — upload a DataFrame and run SQL queries.
"""
from __future__ import annotations

import os
import pandas as pd
from google.cloud import bigquery


def _client() -> bigquery.Client:
    return bigquery.Client(project=os.environ["BIGQUERY_PROJECT_ID"])


def upload_dataframe(df: pd.DataFrame, table_name: str) -> int:
    """Uploads a DataFrame to BigQuery. Returns the number of rows uploaded."""
    client = _client()
    dataset = os.environ["BIGQUERY_DATASET_ID"]
    table_ref = f"{os.environ['BIGQUERY_PROJECT_ID']}.{dataset}.{table_name}"

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # wait for completion
    return len(df)


def run_query(sql: str) -> pd.DataFrame:
    """Runs a SQL query and returns the result as a DataFrame."""
    client = _client()
    return client.query(sql).to_dataframe()


if __name__ == "__main__":
    df = run_query(f"SELECT 1 as test")
    print(df)
