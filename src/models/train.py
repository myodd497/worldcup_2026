"""
Model training — trains XGBoost classifier, logs to MLflow, serialises to
bin/models_deployed/.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.models.feature_engineering import load_fixtures_df, build_features

_MODEL_PATH = Path(__file__).resolve().parents[2] / "bin" / "models_deployed" / "wc2026_predictor.pkl"
_MLFLOW_URI = str(Path(__file__).resolve().parents[2] / "bin" / "mlruns")


def train() -> None:
    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment("wc2026_match_prediction")

    df = load_fixtures_df()
    X, y = build_features(df)

    if X.empty:
        print("No features available yet. Add features to feature_engineering.py first.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "use_label_encoder": False}
        model = XGBClassifier(**params, eval_metric="mlogloss")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "xgboost_model")

        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, _MODEL_PATH)
        print(f"Model saved to {_MODEL_PATH}. Accuracy: {acc:.3f}")


if __name__ == "__main__":
    train()
