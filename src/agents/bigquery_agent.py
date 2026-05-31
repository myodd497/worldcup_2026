"""
BigQuery Agent — uploads DataFrames and runs analytical queries.
"""
from src.tools.bigquery_tools import upload_dataframe, run_query
import pandas as pd


def upload_match_facts(df: pd.DataFrame, table: str = "match_facts") -> str:
    rows = upload_dataframe(df, table)
    return f"Uploaded {rows} rows to BigQuery table `{table}`."


def query(sql: str) -> str:
    df = run_query(sql)
    return df.to_string(index=False)


if __name__ == "__main__":
    sql = "SELECT COUNT(*) as total_matches FROM worldcup2026.match_facts"
    print(query(sql))
