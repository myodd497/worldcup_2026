"""Planner Agent — decides the best response plan and specialist agents to use.

This agent does not answer the user directly. It analyses the request, the
conversation context, and the available agents, then returns a structured plan.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

AVAILABLE_AGENTS = ["news", "sentiment", "prediction", "bigquery", "chat"]

_PLANNER_PROMPT = """\
You are a planning agent for a football assistant.

Your job:
- Read the user request and recent conversation context.
- Decide the best plan to respond.
- Select one or two specialist agents that should be used.

Available agents:
- news: latest media news, headlines, rumours, transfers, injuries.
- sentiment: fan/social sentiment and public opinion.
- prediction: win/draw/loss probabilities and match forecasts.
- bigquery: ALL structured data — fixtures, results, standings, venues, referees, lineups, historical facts, counts, comparisons, analytics, head-to-head, form, upcoming schedule. Use this for any factual data question.
- chat: generic conversation only.

Rules:
- bigquery is the single source of truth for all structured football data. Always include it for any data question.
- If the user asks for predictions, include prediction and bigquery.
- If the user asks for news or sentiment, include the corresponding specialist.
- Use chat only for generic conversation with no data need.
- Never select more than 2 agents unless truly necessary.
- Return JSON only.

Return schema:
{{
    "agents": ["bigquery"],
    "response_mode": "single|multi",
    "reason": "short explanation",
    "primary_agent": "bigquery"
}}

Conversation context:
{context}

User request:
{message}
""".strip()


def _format_recent_history(messages: list[dict[str, str]], max_messages: int = 8, max_chars: int = 350) -> str:
    if not messages:
        return "None"

    lines: list[str] = []
    for msg in messages[-max_messages:]:
        role = str(msg.get("role", "user")).strip().lower() or "user"
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:max_chars]}")
    return "\n".join(lines) if lines else "None"


def _fallback_plan(query: str) -> dict[str, Any]:
    q = query.lower()
    if any(term in q for term in ("count", "how many", "average", "table", "schema", "list", "show", "days until", "days left", "countdown", "how long until")):
        agents = ["bigquery"]
    elif any(term in q for term in ("prediction", "probability", "win", "draw", "loss")):
        agents = ["prediction", "bigquery"]
    elif any(term in q for term in ("news", "latest", "headline", "rumour", "injury")):
        agents = ["news"]
    elif any(term in q for term in ("sentiment", "social", "fan reaction")):
        agents = ["sentiment"]
    elif any(term in q for term in ("fixture", "match", "result", "standings", "venue", "referee", "lineup")):
        agents = ["bigquery"]
    else:
        agents = ["chat"]

    return {
        "agents": agents,
        "response_mode": "multi" if len(agents) > 1 else "single",
        "reason": "Fallback planner used due to planner parse failure or unavailable model.",
        "primary_agent": agents[0],
    }


def plan_response(
    user_message: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    prompt = _PLANNER_PROMPT.format(
        context=_format_recent_history(conversation_history or []),
        message=user_message,
    )

    try:
        raw = _llm.invoke(prompt).content.strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Planner output was not a JSON object")

        candidates = parsed.get("agents", [])
        agents = [a for a in candidates if isinstance(a, str) and a in AVAILABLE_AGENTS]
        if not agents:
            raise ValueError("Planner returned no valid agents")

        primary_agent = parsed.get("primary_agent")
        if not isinstance(primary_agent, str) or primary_agent not in agents:
            primary_agent = agents[0]

        response_mode = parsed.get("response_mode", "single")
        if response_mode not in {"single", "multi"}:
            response_mode = "multi" if len(agents) > 1 else "single"

        reason = parsed.get("reason", "")
        if not isinstance(reason, str):
            reason = ""

        return {
            "agents": agents[:3],
            "response_mode": response_mode,
            "reason": reason,
            "primary_agent": primary_agent,
        }
    except Exception:
        return _fallback_plan(user_message)
