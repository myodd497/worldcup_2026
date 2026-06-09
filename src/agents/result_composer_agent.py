"""
Result Composer Agent — converts structured specialist output into a concise,
WhatsApp-friendly response with confidence signalling (gold star ratings).
"""
from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.llm_config import create_chat_model


_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    """Lazy-initialize the LLM client."""
    global _llm
    if _llm is None:
        # Cap output to keep replies WhatsApp-friendly and avoid DeepSeek verbosity.
        _llm = create_chat_model("simple", temperature=0, max_tokens=400)
    return _llm


def _confidence_line(score: float, label: str) -> str:
    """Format confidence as 1-5 gold stars + label."""
    num_stars = max(1, min(5, round(score * 5)))
    filled = "⭐" * num_stars
    empty = "☆" * (5 - num_stars)
    pct = round(score * 100)
    return f"{filled}{empty}  *{label.upper()} ({pct}%)*"


def _format_final_answer(
    *,
    user_message: str,
    intent: str,
    selected_agent: str,
    raw_answer: str,
) -> str:
    """Applies a final presentation layer so all agents return polished output."""
    prompt = (
        "You are a response formatter for a football assistant.\n"
        "Rewrite the assistant answer to be clean, user-friendly markdown.\n"
        "Formatting rules:\n"
        "- Keep all factual content from the raw answer; do not invent new facts.\n"
        "- Start with a direct answer in one short sentence.\n"
        "- Use either bullet points or a markdown table when listing data.\n"
        "- Keep links if they already exist.\n"
        "- Keep it concise and readable on chat/mobile.\n"
        "- Do not include confidence labels or internal workflow/debug text.\n"
        "- Preserve uncertainty where present.\n\n"
        f"User message: {user_message}\n"
        f"Intent: {intent}\n"
        f"Selected agent: {selected_agent}\n\n"
        "Raw assistant answer:\n"
        f"{raw_answer}"
    )
    return _get_llm().invoke(prompt).content.strip()


def compose(
    user_message: str,
    intent: str,
    selected_agent: str,
    payload: dict[str, Any],
    confidence_score: float,
    confidence_label: str,
    confidence_reason: str,
) -> str:
    raw_answer = str(payload.get("answer", "I could not build a response for that request."))
    data_source = str((payload.get("metadata", {}) or {}).get("data_source", "unknown"))

    if selected_agent == "bigquery" or data_source == "bigquery":
        answer = raw_answer
    else:
        try:
            answer = _format_final_answer(
                user_message=user_message,
                intent=intent,
                selected_agent=selected_agent,
                raw_answer=raw_answer,
            )
        except Exception:
            answer = raw_answer

    sections = [answer, "", _confidence_line(confidence_score, confidence_label)]

    if confidence_label == "low":
        sections.append(f"💡 *Reason*: {confidence_reason}")
        sections.append("💬 *Tip*: ask with a specific match, teams, or time window for better accuracy.")

    if selected_agent == "prediction" and confidence_score < 0.55:
        sections.append("⚠️ *Prediction caution*: use this as directional guidance, not a guaranteed outcome.")

    return "\n\n".join(sections)
