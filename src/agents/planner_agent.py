"""Planner Agent — single-call router for the football assistant.

One simple-tier LLM call with structured output decides:
  - which specialist(s) to invoke (1-2)
  - the topic of the request (used for memory and confidence)
  - whether the request needs the LLM verifier afterwards

Routing is a constrained classification problem. We save the complex tier for
quality-critical SQL generation and verification.

Specialist set:
  bigquery | prediction | news | sentiment | rules | chat
"""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agents.llm_config import create_chat_model

_llm = None  # ChatOpenAI bound with structured output


AgentName = Literal["bigquery", "prediction", "news", "sentiment", "rules", "chat"]


class _PlanSchema(BaseModel):
    agents: list[AgentName] = Field(..., description="1-2 specialists to invoke, ordered by primacy.")
    primary_agent: AgentName = Field(..., description="The agent whose answer drives the reply.")
    topic: str = Field(default="", description="Short noun phrase summarising the request.")
    needs_verifier: bool = Field(default=False, description="Set true when bigquery is selected or numbers/lists are requested.")
    is_summary: bool = Field(default=False, description="True ONLY for broad 'summarise / overview / season recap / tell me about' style questions that need multiple SQL angles.")
    reason: str = Field(default="", description="One short sentence explaining the routing.")


def _get_llm():
    global _llm
    if _llm is None:
        base = create_chat_model("simple", temperature=0, max_retries=6, timeout=60)
        _llm = base.with_structured_output(_PlanSchema)
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
- Words like "squad", "roster", "lineup", "players", "starting XI" asking WHO is on a team → `bigquery` (player list lives in the warehouse). Route to `rules` ONLY when the question is about the *regulation* itself (e.g. "how many players per squad", "squad submission deadline", "squad size limit").
- Use `prediction` ONLY when the user explicitly asks for a forecast/probability of a specific upcoming match. In that case include BOTH `prediction` and `bigquery`.
- Never select more than 2 agents.
- Set `needs_verifier = true` ONLY when the request asks for specific numbers, rankings, multi-row lists, or comparisons that can be wrong subtly. Simple lookups (next match, last result, single fact) do NOT need the verifier.
- Set `is_summary = true` ONLY for BROAD overview questions about a team/player/competition that span multiple angles (e.g. "summarise FC Porto 2025-2026 season", "give me an overview of Mbappé this year", "how was Portugal's qualification campaign"). Single-fact and ranking questions are NOT summaries.
- When `is_summary = true`, set `needs_verifier = false` (the summary path verifies each sub-question internally).
- `topic` is one short noun phrase (e.g. "Portugal form", "WC2026 top scorers", "rules: yellow cards").

## Return schema (JSON only, no prose)
{{
  "agents": ["bigquery"],
  "primary_agent": "bigquery",
  "topic": "...",
  "needs_verifier": true,
  "is_summary": false,
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
    elif any(t in q for t in ("rule", "regulation", "format", "protest", "disciplinary",
                              "eligibility", "doping", "award", "trophy")):
        agents = ["rules"]
    elif any(t in q for t in ("news", "headline", "rumour", "transfer", "injury")):
        agents = ["news"]
    elif any(t in q for t in ("sentiment", "social", "fan reaction", "fans think")):
        agents = ["sentiment"]
    elif any(t in q for t in ("predict", "prediction", "probability", "odds", "who will win", "forecast")):
        agents = ["prediction", "bigquery"]
    else:
        agents = ["bigquery"]
    is_summary = any(
        t in q for t in (
            "summary", "summarise", "summarize", "overview", "recap",
            "tell me about", "how was", "season ", "campaign",
        )
    ) and "bigquery" in agents
    return {
        "agents": agents,
        "primary_agent": agents[0],
        "topic": "",
        "needs_verifier": ("bigquery" in agents) and not is_summary,
        "is_summary": is_summary,
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
        plan: _PlanSchema = _get_llm().invoke(prompt)
        agents = [a for a in plan.agents if a in AVAILABLE_AGENTS][:2]
        if not agents:
            raise ValueError("planner returned no valid agents")
        primary = plan.primary_agent if plan.primary_agent in agents else agents[0]
        is_summary = bool(plan.is_summary) and "bigquery" in agents
        # Summary path verifies each sub-question internally — skip the global verifier.
        needs_verifier = (
            False if is_summary else bool(plan.needs_verifier or "bigquery" in agents)
        )
        return {
            "agents": agents,
            "primary_agent": primary,
            "topic": plan.topic or "",
            "needs_verifier": needs_verifier,
            "is_summary": is_summary,
            "reason": plan.reason or "",
        }
    except Exception:
        return _fallback_plan(user_message)
