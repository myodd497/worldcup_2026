"""
Feature engineering — builds the training feature matrix from BigQuery fixtures.
"""
from __future__ import annotations

import pandas as pd
from src.tools.bigquery_tools import run_query


def load_fixtures_df() -> pd.DataFrame:
    sql = f"""
        SELECT
            fixture_id,
            fixture_datetime,
            season,
            status,
            home_team_id,
            away_team_id,
            home_goals,
            away_goals
        FROM `{__import__('os').environ['BIGQUERY_PROJECT_ID']}.{__import__('os').environ['BIGQUERY_DATASET_ID']}.fact_fixture`
    """
    return run_query(sql)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X, y) where:
      X = feature matrix
      y = target (0=away win, 1=draw, 2=home win)
    """
    # Select completed matches only
    completed = df[df["status"] == "FT"].copy()

    home_goals = completed["home_goals"].fillna(0).astype(int)
    away_goals = completed["away_goals"].fillna(0).astype(int)

    # Target variable
    def outcome(row):
        if row["home_goals"] > row["away_goals"]:
            return 2  # home win
        elif row["home_goals"] < row["away_goals"]:
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
