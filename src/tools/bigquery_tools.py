"""
BigQuery tools — upload a DataFrame and run SQL queries.
"""
from __future__ import annotations

import json
import os
from typing import Literal
import logging

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

WriteDisposition = Literal["WRITE_APPEND", "WRITE_TRUNCATE", "WRITE_EMPTY"]
logger = logging.getLogger(__name__)


def _validate_private_key(private_key: str, private_key_id: str) -> str:
    """Normalizes and validates service-account private key text."""
    key = (private_key or "").replace("\\n", "\n").strip()
    key_id = (private_key_id or "").strip()

    placeholders = ("REPLACE_ME", "YOUR_PRIVATE_KEY", "<PRIVATE_KEY>", "PRIVATE_KEY")
    if any(marker in key for marker in placeholders) or any(marker in key_id for marker in placeholders):
        raise ValueError("Service account placeholders detected in GOOGLE_SERVICE_ACCOUNT_INFO.")

    if "BEGIN PRIVATE KEY" not in key or "END PRIVATE KEY" not in key:
        raise ValueError("Invalid service account private_key format.")

    # Heuristic guardrail for accidentally truncated PEM content.
    pem_lines = [ln for ln in key.splitlines() if ln.strip()]
    if len(pem_lines) < 8:
        raise ValueError("Service account private_key appears truncated (too few PEM lines).")

    return key


def _client() -> bigquery.Client:
    project_id = os.environ["BIGQUERY_PROJECT_ID"]

    # Cloud mode: credentials passed as JSON payload in env var.
    # This is useful on Streamlit Community Cloud where local file paths are not available.
    service_account_info = os.environ.get("GOOGLE_SERVICE_ACCOUNT_INFO", "").strip()
    if service_account_info:
        try:
            info = json.loads(service_account_info)

            info["private_key"] = _validate_private_key(
                str(info.get("private_key", "")),
                str(info.get("private_key_id", "")),
            )

            credentials = service_account.Credentials.from_service_account_info(info)
            return bigquery.Client(project=project_id, credentials=credentials)
        except Exception as exc:  # pragma: no cover - defensive for runtime secrets issues
            raise RuntimeError(
                "Invalid GOOGLE_SERVICE_ACCOUNT_INFO. "
                "Please fix Streamlit secrets [gcp_service_account] with a complete valid service account key. "
                f"Underlying error: {exc}"
            )

    # Local mode: use GOOGLE_APPLICATION_CREDENTIALS file if configured.
    return bigquery.Client(project=project_id)


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


def execute_sql(sql: str) -> None:
    """Executes SQL (DDL/DML) and waits for completion."""
    client = _client()
    job = client.query(sql)
    job.result()


if __name__ == "__main__":
    df = run_query("SELECT 1 as test")
    print(df)
