"""
Prediction Agent — returns win/draw/loss probabilities for a match.
Loads the trained XGBoost model and augments the output with LLM reasoning.
"""
from __future__ import annotations

from src.models.predict import predict_match
from langchain_openai import ChatOpenAI


_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    """Lazy-initialize the LLM client."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    return _llm


def run_structured(query: str) -> dict:
    probs = predict_match(query)
    if not probs:
        return {
            "answer": f"Unable to generate a prediction for: {query}",
            "confidence_score": 0.25,
            "confidence_reason": "Insufficient match context to build features.",
            "metadata": {"model_version": "none"},
        }

    home = probs["home_team"]
    away = probs["away_team"]
    p_home = probs["home_win_pct"]
    p_draw = probs["draw_pct"]
    p_away = probs["away_win_pct"]

    reasoning_prompt = (
        f"Based on these match probabilities for {home} vs {away}: "
        f"{home} win {p_home:.0f}%, draw {p_draw:.0f}%, {away} win {p_away:.0f}%. "
        f"Give a 2-sentence tactical analysis explaining why. Be concise."
    )
    reasoning = _get_llm().invoke(reasoning_prompt).content.strip()

    answer = (
        f"*Prediction: {home} vs {away}*\n"
        f"🟢 {home} win: {p_home:.0f}%\n"
        f"🟡 Draw: {p_draw:.0f}%\n"
        f"🔴 {away} win: {p_away:.0f}%\n\n"
        f"💡 {reasoning}"
    )

    model_version = probs.get("model_version", "unknown")
    data_source = probs.get("data_source", "unknown")
    if model_version == "fallback_uniform":
        score = 0.4
        reason = "Fallback probabilities used because trained model features are not available."
    elif model_version == "bq_heuristic_v1" and data_source == "bigquery":
        score = 0.85
        reason = "Prediction computed from historical BigQuery data (cache-first, no API call)."
    elif model_version == "bq_heuristic_v1" and data_source == "api_then_bigquery":
        score = 0.7
        reason = "BigQuery cache miss: fetched from API once, stored in BigQuery, then predicted."
    else:
        score = 0.6
        reason = "Prediction generated with limited features; treat as directional guidance."

    return {
        "answer": answer,
        "confidence_score": score,
        "confidence_reason": reason,
        "metadata": {
            "model_version": model_version,
            "data_source": data_source,
            "samples": probs.get("samples", {}),
        },
    }


def run(query: str) -> str:
    return run_structured(query)["answer"]


if __name__ == "__main__":
    print(run("Portugal vs Morocco"))
