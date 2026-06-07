"""
Feature engineering — builds the training feature matrix from BigQuery fixtures.
"""
from __future__ import annotations

import pandas as pd
from src.tools.bigquery_tools import run_query


def load_fixtures_df() -> pd.DataFrame:
    import os
    project = os.environ["BIGQUERY_PROJECT_ID"]
    dataset = os.environ["BIGQUERY_DATASET_ID"]
    sql = f"""
        SELECT
            match_id        AS fixture_id,
            kickoff_at      AS fixture_datetime,
            season_year     AS season,
            match_status    AS status,
            home_team_id,
            away_team_id,
            home_goals,
            away_goals
        FROM `{project}.{dataset}.fact_match`
    """
    return run_query(sql)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X, y) where:
      X = feature matrix
      y = target (0=away win, 1=draw, 2=home win)
    """
    # Select completed matches only
    completed = df[df["status"] == "FINISHED"].copy()

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
