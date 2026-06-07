"""
Prediction — loads the trained model and returns win/draw/loss probabilities.
Falls back to a uniform prior if no model is deployed yet.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import joblib

from src.tools.bigquery_tools import run_query

_MODEL_PATH = Path(__file__).resolve().parents[2] / "bin" / "models_deployed" / "wc2026_predictor.pkl"

_model = None


def _load_model():
    global _model
    if _model is None and _MODEL_PATH.exists():
        _model = joblib.load(_MODEL_PATH)
    return _model


def _safe(text: str) -> str:
    return text.replace("'", "''")


def _table_ref() -> str:
    project = os.environ["BIGQUERY_PROJECT_ID"]
    dataset = os.environ["BIGQUERY_DATASET_ID"]
    return f"`{project}.{dataset}.fact_match`"


def _parse_matchup(query: str) -> tuple[str, str] | None:
    cleaned = query.strip()
    cleaned = re.sub(r"(?i)\b(prediction|predict|odds|probability|probabilities|match\s*facts)\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:")

    patterns = [
        r"^\s*(.+?)\s+vs\s+(.+?)\s*$",
        r"^\s*(.+?)\s+v\s+(.+?)\s*$",
        r"^\s*(.+?)\s*-\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        m = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None


def _fetch_team_history(team: str, limit: int = 10) -> list[dict]:
    team_safe = _safe(team.lower())
    sql = f"""
        SELECT CAST(kickoff_at AS STRING) AS date,
                     home_team_name AS home_team,
                     away_team_name AS away_team,
                     home_goals,
                     away_goals
    FROM {_table_ref()}
        WHERE (LOWER(home_team_name) LIKE '%{team_safe}%' OR LOWER(away_team_name) LIKE '%{team_safe}%')
      AND home_goals IS NOT NULL
      AND away_goals IS NOT NULL
        ORDER BY kickoff_at DESC
    LIMIT {int(limit)}
    """
    df = run_query(sql)
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _fetch_h2h(home_team: str, away_team: str, limit: int = 10) -> list[dict]:
    home_safe = _safe(home_team.lower())
    away_safe = _safe(away_team.lower())
    sql = f"""
        SELECT CAST(kickoff_at AS STRING) AS date,
                     home_team_name AS home_team,
                     away_team_name AS away_team,
                     home_goals,
                     away_goals
    FROM {_table_ref()}
    WHERE (
                (LOWER(home_team_name) LIKE '%{home_safe}%' AND LOWER(away_team_name) LIKE '%{away_safe}%')
        OR
                (LOWER(home_team_name) LIKE '%{away_safe}%' AND LOWER(away_team_name) LIKE '%{home_safe}%')
    )
      AND home_goals IS NOT NULL
      AND away_goals IS NOT NULL
        ORDER BY kickoff_at DESC
    LIMIT {int(limit)}
    """
    df = run_query(sql)
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _team_points_per_match(team: str, rows: list[dict]) -> float:
    if not rows:
        return 1.0
    points = 0.0
    for row in rows:
        hg = float(row["home_goals"])
        ag = float(row["away_goals"])
        home = str(row["home_team"]).lower()
        team_is_home = team.lower() in home
        if hg == ag:
            points += 1.0
        elif (hg > ag and team_is_home) or (ag > hg and not team_is_home):
            points += 3.0
    return points / len(rows)


def _team_goal_diff_per_match(team: str, rows: list[dict]) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        hg = float(row["home_goals"])
        ag = float(row["away_goals"])
        home = str(row["home_team"]).lower()
        if team.lower() in home:
            total += hg - ag
        else:
            total += ag - hg
    return total / len(rows)


def _h2h_home_edge(home_team: str, away_team: str, rows: list[dict]) -> float:
    if not rows:
        return 0.0
    edge = 0.0
    for row in rows:
        hg = float(row["home_goals"])
        ag = float(row["away_goals"])
        home = str(row["home_team"]).lower()
        away = str(row["away_team"]).lower()

        if home_team.lower() in home and away_team.lower() in away:
            edge += 1.0 if hg > ag else (-1.0 if hg < ag else 0.0)
        elif home_team.lower() in away and away_team.lower() in home:
            edge += 1.0 if ag > hg else (-1.0 if ag < hg else 0.0)
    return edge / len(rows)


def _warm_cache_from_api(home_team: str, away_team: str) -> None:
    # No-op: live API ingestion is owned by the ETL scheduler, not the chat path.
    # Runtime prediction relies on BigQuery-only data.
    _ = (home_team, away_team)


def _heuristic_probs(home_team: str, away_team: str, source: str) -> dict | None:
    home_rows = _fetch_team_history(home_team, limit=10)
    away_rows = _fetch_team_history(away_team, limit=10)
    h2h_rows = _fetch_h2h(home_team, away_team, limit=10)

    if not home_rows or not away_rows:
        return None

    ppm_home = _team_points_per_match(home_team, home_rows)
    ppm_away = _team_points_per_match(away_team, away_rows)
    gd_home = _team_goal_diff_per_match(home_team, home_rows)
    gd_away = _team_goal_diff_per_match(away_team, away_rows)
    h2h_edge = _h2h_home_edge(home_team, away_team, h2h_rows)

    # Linear edge score, then calibrated into 3-way probabilities.
    edge = 0.9 * (ppm_home - ppm_away) + 0.5 * (gd_home - gd_away) + 0.35 * h2h_edge
    p_home = 1.0 / (1.0 + math.exp(-edge))
    draw = max(0.12, min(0.30, 0.24 - 0.05 * abs(edge)))

    home = p_home * (1.0 - draw)
    away = (1.0 - p_home) * (1.0 - draw)
    total = home + draw + away
    if total <= 0:
        return None

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_pct": round((home / total) * 100, 1),
        "draw_pct": round((draw / total) * 100, 1),
        "away_win_pct": round((away / total) * 100, 1),
        "model_version": "bq_heuristic_v1",
        "data_source": source,
        "samples": {
            "home_recent": len(home_rows),
            "away_recent": len(away_rows),
            "h2h": len(h2h_rows),
        },
    }


def predict_match(query: str) -> dict | None:
    """
    Parses a query like 'Portugal vs Morocco' and returns probability dict.
    NOTE: extend this to build a real feature vector from team stats.
    """
    parsed = _parse_matchup(query)
    if not parsed:
        return None

    home_team, away_team = parsed

    # 1) Cache-first from BigQuery.
    try:
        probs = _heuristic_probs(home_team, away_team, source="bigquery")
    except Exception:
        probs = None
    if probs:
        return probs

    # 2) Legacy model path (if deployed and feature builder exists).
    model = _load_model()
    if model is not None:
        # Feature builder not implemented yet; keep fallback path explicit.
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_pct": 33.3,
            "draw_pct": 33.3,
            "away_win_pct": 33.3,
            "model_version": "xgboost_features_missing_fallback",
        }

    # 4) Final fallback if no usable data and no model features.
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_pct": 33.3,
        "draw_pct": 33.3,
        "away_win_pct": 33.3,
        "model_version": "fallback_uniform",
    }


if __name__ == "__main__":
    print(predict_match("Portugal vs Morocco"))
