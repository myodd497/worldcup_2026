"""Verifier (LLM-as-judge) — scores whether the proposed answer actually answers
the user's question and is grounded in the retrieved rows.

Returns a verdict the orchestrator uses to:
  - accept and pass to the composer, or
  - send back to the specialist with a structured critique for ONE repair attempt.

Public API:
  - verify(question, answer, sql_executed, row_samples) → Verdict
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.llm_config import create_chat_model


_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        # Verifier emits a tiny JSON verdict. Use simple tier (Flash, no thinking)
        # to keep latency low; override via VERIFIER_LLM_TIER=complex if needed.
        import os
        tier = os.getenv("VERIFIER_LLM_TIER", "simple")
        _llm = create_chat_model(tier, temperature=0, max_retries=6, timeout=60, max_tokens=400)
    return _llm


@dataclass(frozen=True)
class Verdict:
    grounded: bool             # facts in the answer can be traced to the rows
    answers_question: bool     # the answer addresses what was asked
    confidence: float          # 0-1
    issues: tuple[str, ...]    # short bullet list of problems
    repair_hint: str           # actionable hint for the SQL agent (if needs_repair)
    needs_repair: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["issues"] = list(d["issues"])
        return d


_SYSTEM = """\
You are a strict quality reviewer for a football-data assistant.
You receive: the user's question, the SQL the agent ran, sample rows from the result, and the agent's natural-language answer.

Your job:
1. Decide if the answer is GROUNDED — every concrete claim must trace to the rows.
2. Decide if the answer ACTUALLY answers the question (right metric, right scope, right entities).
3. Score overall confidence 0.0-1.0.
4. List concrete issues (e.g. "wrong team_id used", "missing 'last 10' window", "ranks by goals instead of goals+assists").
5. If repair is warranted, give a SHORT actionable hint the SQL agent can act on.

Return ONLY valid JSON in this exact shape:
{
  "grounded": true|false,
  "answers_question": true|false,
  "confidence": 0.0-1.0,
  "issues": ["..."],
  "repair_hint": "...",
  "needs_repair": true|false
}
"""


def verify(
    *,
    question: str,
    answer: str,
    sql_executed: list[str],
    row_samples: list[dict],
    max_row_preview: int = 5,
) -> Verdict:
    rows_preview = json.dumps(row_samples[:max_row_preview], default=str)[:1200]
    sql_block = "\n---\n".join(sql_executed[-2:])[:1200]
    user_block = (
        f"Question: {question}\n\n"
        f"SQL executed (last calls):\n```sql\n{sql_block}\n```\n\n"
        f"Sample rows (first {max_row_preview}):\n{rows_preview}\n\n"
        f"Agent answer:\n{answer}\n"
    )
    try:
        raw = _get_llm().invoke([
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_block},
        ]).content.strip()
        # Strip optional ```json fences
        if raw.startswith("```"):
            raw = raw.strip("` \n")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
    except Exception as exc:
        return Verdict(
            grounded=True, answers_question=True, confidence=0.5,
            issues=(f"verifier failed: {exc}",),
            repair_hint="",
            needs_repair=False,
        )

    return Verdict(
        grounded=bool(parsed.get("grounded", True)),
        answers_question=bool(parsed.get("answers_question", True)),
        confidence=float(parsed.get("confidence", 0.5)),
        issues=tuple(str(x) for x in parsed.get("issues", []) or []),
        repair_hint=str(parsed.get("repair_hint", "") or ""),
        needs_repair=bool(parsed.get("needs_repair", False)),
    )
