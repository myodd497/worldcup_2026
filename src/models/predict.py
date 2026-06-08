"""Prediction — win/draw/loss probabilities computed over the gold datamodel.

Uses the deterministic team-name → team_id resolver (no `LOWER LIKE`) and
queries the agent-visible facts/marts (no source-table sneak path).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import joblib

from src.tools.bigquery_tools import run_query
from src.tools.entity_resolver import resolve_team

_MODEL_PATH = Path(__file__).resolve().parents[2] / "bin" / "models_deployed" / "wc2026_predictor.pkl"

_model = None


def _load_model():
    global _model
    if _model is None and _MODEL_PATH.exists():
        _model = joblib.load(_MODEL_PATH)
    return _model


def _parse_matchup(query: str) -> tuple[str, str] | None:
    cleaned = re.sub(
        r"(?i)\b(prediction|predict|odds|probability|probabilities|match\s*facts)\b",
        "", (query or "").strip(),
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:")
    for pattern in (r"^\s*(.+?)\s+vs\s+(.+?)\s*$",
                    r"^\s*(.+?)\s+v\s+(.+?)\s*$",
                    r"^\s*(.+?)\s*-\s*(.+?)\s*$"):
        m = re.match(pattern, cleaned, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None


def _team_form(team_id: int, limit: int = 10) -> dict | None:
    sql = f"""
    WITH ranked AS (
      SELECT
        team_id, result, goals_for, goals_against, match_date,
        ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY match_date DESC) AS rn
      FROM {fqn('fact_match_team')}
      WHERE team_id = {int(team_id)} AND result IS NOT NULL
    )
    SELECT
      COUNT(*)                                          AS matches,
      SUM(IF(result='W',3,IF(result='D',1,0)))          AS points,
      SUM(goals_for) - SUM(goals_against)               AS goal_diff
    FROM ranked
    WHERE rn <= {int(limit)}
    """
    df = run_query(sql)
    if df.empty or int(df["matches"].iloc[0] or 0) == 0:
        return None
    n = int(df["matches"].iloc[0])
    return {
        "matches": n,
        "ppm": float(df["points"].iloc[0]) / n,
        "gd_per_match": float(df["goal_diff"].iloc[0]) / n,
    }


def _h2h_edge(team_a_id: int, team_b_id: int) -> tuple[float, int]:
    """Return (edge_for_a, sample_size). edge ∈ [-1, +1]."""
    lo, hi = (team_a_id, team_b_id) if team_a_id < team_b_id else (team_b_id, team_a_id)
    sql = f"""
    SELECT matches_played, team_lo_wins, team_hi_wins, draws
    FROM {fqn('mart_head_to_head')}
    WHERE team_lo_id = {int(lo)} AND team_hi_id = {int(hi)}
    """
    df = run_query(sql)
    if df.empty:
        return 0.0, 0
    n = int(df["matches_played"].iloc[0] or 0)
    if n == 0:
        return 0.0, 0
    lo_wins = int(df["team_lo_wins"].iloc[0] or 0)
    hi_wins = int(df["team_hi_wins"].iloc[0] or 0)
    a_wins = lo_wins if team_a_id == lo else hi_wins
    b_wins = hi_wins if team_a_id == lo else lo_wins
    return (a_wins - b_wins) / n, n


def _heuristic_probs(home_name: str, away_name: str) -> dict | None:
    home = resolve_team(home_name)
    away = resolve_team(away_name)
    if not (home.matched and away.matched):
        return None

    home_form = _team_form(home.id, limit=10)
    away_form = _team_form(away.id, limit=10)
    if not home_form or not away_form:
        return None

    edge_h2h, h2h_n = _h2h_edge(home.id, away.id)
    edge = (
        0.9 * (home_form["ppm"] - away_form["ppm"])
        + 0.5 * (home_form["gd_per_match"] - away_form["gd_per_match"])
        + 0.35 * edge_h2h
    )
    p_home_axis = 1.0 / (1.0 + math.exp(-edge))
    draw = max(0.12, min(0.30, 0.24 - 0.05 * abs(edge)))

    home_p = p_home_axis * (1.0 - draw)
    away_p = (1.0 - p_home_axis) * (1.0 - draw)
    total = home_p + draw + away_p
    if total <= 0:
        return None

    return {
        "home_team": home.name or home_name,
        "away_team": away.name or away_name,
        "home_team_id": home.id,
        "away_team_id": away.id,
        "home_win_pct": round((home_p / total) * 100, 1),
        "draw_pct": round((draw / total) * 100, 1),
        "away_win_pct": round((away_p / total) * 100, 1),
        "model_version": "bq_heuristic_v2",
        "data_source": "bigquery",
        "samples": {
            "home_recent": home_form["matches"],
            "away_recent": away_form["matches"],
            "h2h": h2h_n,
        },
        "entity_resolution": {
            "home_confidence": home.confidence,
            "away_confidence": away.confidence,
        },
    }


def predict_match(query: str) -> dict | None:
    parsed = _parse_matchup(query)
    if not parsed:
        return None
    home_name, away_name = parsed

    try:
        probs = _heuristic_probs(home_name, away_name)
    except Exception:
        probs = None
    if probs:
        return probs

    model = _load_model()
    if model is not None:
        return {
            "home_team": home_name, "away_team": away_name,
            "home_win_pct": 33.3, "draw_pct": 33.3, "away_win_pct": 33.3,
            "model_version": "xgboost_features_missing_fallback",
        }

    return {
        "home_team": home_name, "away_team": away_name,
        "home_win_pct": 33.3, "draw_pct": 33.3, "away_win_pct": 33.3,
        "model_version": "fallback_uniform",
    }


if __name__ == "__main__":
    print(predict_match("Portugal vs Morocco"))
