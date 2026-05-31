"""
Prediction Agent — returns win/draw/loss probabilities for a match.
Loads the trained XGBoost model and augments the output with LLM reasoning.
"""
from src.models.predict import predict_match
from langchain_openai import ChatOpenAI


_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


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
    reasoning = _llm.invoke(reasoning_prompt).content.strip()

    answer = (
        f"*Prediction: {home} vs {away}*\n"
        f"🟢 {home} win: {p_home:.0f}%\n"
        f"🟡 Draw: {p_draw:.0f}%\n"
        f"🔴 {away} win: {p_away:.0f}%\n\n"
        f"💡 {reasoning}"
    )

    model_version = probs.get("model_version", "unknown")
    if model_version == "fallback_uniform":
        score = 0.4
        reason = "Fallback probabilities used because trained model features are not available."
    else:
        score = 0.8
        reason = "Probabilities produced by trained model and contextualized by LLM."

    return {
        "answer": answer,
        "confidence_score": score,
        "confidence_reason": reason,
        "metadata": {"model_version": model_version},
    }


def run(query: str) -> str:
    return run_structured(query)["answer"]


if __name__ == "__main__":
    print(run("Portugal vs Morocco"))
