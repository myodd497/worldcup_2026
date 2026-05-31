"""
Prediction — loads the trained model and returns win/draw/loss probabilities.
Falls back to a uniform prior if no model is deployed yet.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

_MODEL_PATH = Path(__file__).resolve().parents[2] / "bin" / "models_deployed" / "wc2026_predictor.pkl"

_model = None


def _load_model():
    global _model
    if _model is None and _MODEL_PATH.exists():
        _model = joblib.load(_MODEL_PATH)
    return _model


def predict_match(query: str) -> dict | None:
    """
    Parses a query like 'Portugal vs Morocco' and returns probability dict.
    NOTE: extend this to build a real feature vector from team stats.
    """
    parts = [p.strip() for p in query.replace(" vs ", " v ").split(" v ")]
    if len(parts) < 2:
        return None

    home_team, away_team = parts[0], parts[1]
    model = _load_model()

    if model is None:
        # No model deployed yet — return uniform fallback
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_pct": 33.3,
            "draw_pct": 33.3,
            "away_win_pct": 33.3,
            "model_version": "fallback_uniform",
        }

    # TODO: build proper feature vector from home_team / away_team stats
    X = pd.DataFrame([{}])  # placeholder — replace with real features
    proba = model.predict_proba(X)[0]
    # classes: 0=away win, 1=draw, 2=home win
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_pct": round(proba[2] * 100, 1),
        "draw_pct": round(proba[1] * 100, 1),
        "away_win_pct": round(proba[0] * 100, 1),
        "model_version": "xgboost_v1",
    }


if __name__ == "__main__":
    print(predict_match("Portugal vs Morocco"))
