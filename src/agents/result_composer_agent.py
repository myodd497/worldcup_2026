"""
Result Composer Agent — converts structured specialist output into a concise,
WhatsApp-friendly response with confidence signalling.
"""
from __future__ import annotations

from typing import Any


def _confidence_line(score: float, label: str) -> str:
    pct = round(score * 100)
    return f"Confidence: {label.upper()} ({pct}%)"


def compose(
    user_message: str,
    intent: str,
    selected_agent: str,
    payload: dict[str, Any],
    confidence_score: float,
    confidence_label: str,
    confidence_reason: str,
) -> str:
    answer = payload.get("answer", "I could not build a response for that request.")
    sections = [answer, "", _confidence_line(confidence_score, confidence_label)]

    if confidence_label == "low":
        sections.append(f"Reason: {confidence_reason}")
        sections.append("Tip: ask with a specific match, teams, or time window for better accuracy.")

    if selected_agent == "prediction" and confidence_score < 0.55:
        sections.append("Prediction caution: use this as directional guidance, not a guaranteed outcome.")

    return "\n".join(sections)
