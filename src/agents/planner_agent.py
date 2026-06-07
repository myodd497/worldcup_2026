"""Planner Agent — single-call router for the football assistant.

Replaces the old two-step (classify_intent + route_request) flow.
One gpt-4o call decides:
  - which specialist(s) to invoke (1-2)
  - the topic of the request (used for memory and confidence)
  - whether the request needs the LLM verifier afterwards

Specialist set after the match_facts removal:
  bigquery | prediction | news | sentiment | rules | chat
"""
from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI


_MODEL = "gpt-4o"  # planning is the routing brain; bad routing = wrong specialist = wrong answer
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=_MODEL, temperature=0, max_retries=6, timeout=60)
    return _llm


AVAILABLE_AGENTS = ["bigquery", "prediction", "news", "sentiment", "rules", "chat"]
DATA_AGENTS = {"bigquery", "prediction"}


_PROMPT = """\
You are the routing brain for a football assistant about the FIFA World Cup 2026 and football history.

Decide which specialist(s) should handle the request. Return ONLY valid JSON.

## Specialists
- bigquery   — ALL structured football data: teams, players, fixtures, results, standings, stats, history, head-to-head, form, schedule, comparisons, analytics. This is the source of truth for any factual question.
- prediction — outcome probabilities for an UPCOMING match the user names explicitly (e.g. "predict Portugal vs Morocco").
- news       — recent media headlines, transfers, injuries.
- sentiment  — fan/social-media reaction.
- rules      — FIFA WC2026 official regulations (cards, eligibility, format, protests, awards…).
- chat       — greetings, generic conversation, or anything that clearly needs no data.

## Rules
- Default to `bigquery` for anything that requires a fact, a number, a name, a comparison, a ranking, or a date.
- Use `prediction` ONLY when the user explicitly asks for a forecast/probability of a specific upcoming match. In that case include BOTH `prediction` and `bigquery`.
- Never select more than 2 agents.
- Set `needs_verifier = true` whenever `bigquery` is selected, OR when the request asks for specific numbers/lists.
- `topic` is one short noun phrase (e.g. "Portugal form", "WC2026 top scorers", "rules: yellow cards").

## Return schema (JSON only, no prose)
{{
  "agents": ["bigquery"],
  "primary_agent": "bigquery",
  "topic": "...",
  "needs_verifier": true,
  "reason": "short explanation"
}}

## Conversation context (earlier turns + remembered entities)
{context}

## User request
{message}
""".strip()


def _fallback_plan(query: str) -> dict[str, Any]:
    q = (query or "").lower()
    if any(t in q for t in ("hello", "hi ", "how are you", "thanks", "thank you")):
        agents = ["chat"]
    elif any(t in q for t in ("rule", "regulation", "format", "yellow card", "red card",
                              "penalty", "extra time", "protest", "disciplinary",
                              "eligibility", "squad", "kit", "doping", "award", "trophy")):
        agents = ["rules"]
    elif any(t in q for t in ("news", "headline", "rumour", "transfer", "injury")):
        agents = ["news"]
    elif any(t in q for t in ("sentiment", "social", "fan reaction", "fans think")):
        agents = ["sentiment"]
    elif any(t in q for t in ("predict", "prediction", "probability", "odds", "who will win", "forecast")):
        agents = ["prediction", "bigquery"]
    else:
        agents = ["bigquery"]
    return {
        "agents": agents,
        "primary_agent": agents[0],
        "topic": "",
        "needs_verifier": "bigquery" in agents,
        "reason": "Fallback planner (LLM unavailable or invalid JSON).",
    }


def plan_response(
    user_message: str,
    conversation_context: str | None = None,
) -> dict[str, Any]:
    prompt = _PROMPT.format(
        context=conversation_context or "None",
        message=user_message,
    )
    try:
        raw = _get_llm().invoke(prompt).content.strip()
        if raw.startswith("```"):
            raw = raw.strip("` \n")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)

        agents = [a for a in parsed.get("agents", []) if a in AVAILABLE_AGENTS]
        if not agents:
            raise ValueError("planner returned no valid agents")
        agents = agents[:2]

        primary = parsed.get("primary_agent")
        if primary not in agents:
            primary = agents[0]

        needs_verifier = bool(parsed.get("needs_verifier", "bigquery" in agents))

        return {
            "agents": agents,
            "primary_agent": primary,
            "topic": str(parsed.get("topic") or ""),
            "needs_verifier": needs_verifier,
            "reason": str(parsed.get("reason") or ""),
        }
    except Exception:
        return _fallback_plan(user_message)
