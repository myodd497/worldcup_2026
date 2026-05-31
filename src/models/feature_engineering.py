"""
Feature engineering — builds the training feature matrix from BigQuery fixtures.
"""
from __future__ import annotations

import pandas as pd
from src.tools.bigquery_tools import run_query


def load_fixtures_df() -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM `{__import__('os').environ['BIGQUERY_PROJECT_ID']}.{__import__('os').environ['BIGQUERY_DATASET_ID']}.fixtures_historical`
    """
    return run_query(sql)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X, y) where:
      X = feature matrix
      y = target (0=away win, 1=draw, 2=home win)
    """
    # Select completed matches only
    completed = df[df["fixture.status.short"] == "FT"].copy()

    home_goals = completed["goals.home"].fillna(0).astype(int)
    away_goals = completed["goals.away"].fillna(0).astype(int)

    # Target variable
    def outcome(row):
        if row["goals.home"] > row["goals.away"]:
            return 2  # home win
        elif row["goals.home"] < row["goals.away"]:
            return 0  # away win
        return 1  # draw

    completed["target"] = completed.apply(outcome, axis=1)

    # Basic features — extend with ELO/FIFA rankings in next iteration
    feature_cols = []
    X = completed[feature_cols] if feature_cols else pd.DataFrame(index=completed.index)
    y = completed["target"]
    return X, y


if __name__ == "__main__":
    df = load_fixtures_df()
    X, y = build_features(df)
    print(f"Features shape: {X.shape}, target distribution:\n{y.value_counts()}")
