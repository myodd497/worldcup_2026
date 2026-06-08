"""Self-consistency check for ranking-style questions.

Many wrong answers in text-to-SQL systems are silently wrong: the SQL runs,
returns rows, the verifier sees a confident-looking answer, but the JOIN path
or aggregation was subtly off. For *ranking* questions ("top N", "best",
"worst", "most", "leader") we can dramatically reduce that failure rate by
generating a second, independently-phrased SQL and requiring agreement on the
top entity.

This module exposes one function:
    run_self_consistency_check(question, primary_answer, conversation_context)
        → dict { consistent: bool, second_top: str | None, second_answer: str }

It only runs when `is_ranking_question(question)` returns True.
"""
from __future__ import annotations

import re
from typing import Any

_RANKING_PATTERNS = (
    r"\btop\s+\d+\b",
    r"\bbest\b",
    r"\bworst\b",
    r"\bmost\b",
    r"\bhighest\b",
    r"\blowest\b",
    r"\bleader(s|ship)?\b",
    r"\branking\b",
    r"\bwho (has|leads|tops)\b",
    r"\bwhich (team|player) (has|leads|tops|scored)\b",
)
_RANKING_RE = re.compile("|".join(_RANKING_PATTERNS), re.IGNORECASE)


def is_ranking_question(question: str) -> bool:
    return bool(_RANKING_RE.search(question or ""))


def _extract_top_entity(answer: str) -> str | None:
    """Best-effort extraction of the top entity name from a markdown answer.

    Looks for the first bullet/numbered list item or the first bolded name.
    """
    if not answer:
        return None
    # Bold name in first line
    m = re.search(r"\*\*([^*]{2,60})\*\*", answer)
    if m:
        return _clean(m.group(1))
    # First markdown list item
    for line in answer.splitlines():
        m = re.match(r"^\s*(?:\d+\.|[-*])\s*(.+?)(?:\s+[-—:|]|\s*$)", line)
        if m:
            return _clean(m.group(1))
    return None


def _clean(name: str) -> str:
    name = re.sub(r"[`*_]+", "", name).strip()
    name = re.sub(r"\s+[—-]\s+.*$", "", name)  # strip trailing "— 12 goals"
    return name.strip().lower()


def run_self_consistency_check(
    question: str,
    primary_answer: str,
    conversation_context: str | None = None,
) -> dict[str, Any]:
    """Run a second BQ pass with an alternative-path instruction; compare top entity."""
    from src.agents.bigquery_agent import run_structured as run_bq

    alt_prompt = (
        f"{question}\n\n"
        "[INTERNAL SELF-CONSISTENCY CHECK — write SQL that takes a DIFFERENT join path "
        "or computes the metric in a different way than the most obvious approach. "
        "If a mart was the natural choice, derive from facts. If facts were natural, "
        "use the mart. The goal is to independently verify the top result.]"
    )
    try:
        result = run_bq(alt_prompt, conversation_context=conversation_context)
    except Exception as exc:
        return {"consistent": True, "second_top": None, "second_answer": "", "error": str(exc)}

    second_answer = str(result.get("answer", ""))
    primary_top = _extract_top_entity(primary_answer)
    second_top = _extract_top_entity(second_answer)

    # If we couldn't extract either, don't claim inconsistency.
    if not primary_top or not second_top:
        return {"consistent": True, "second_top": second_top, "second_answer": second_answer}

    # Loose match: substring either way (handles "Lionel Messi" vs "Messi").
    consistent = primary_top in second_top or second_top in primary_top
    return {
        "consistent": consistent,
        "primary_top": primary_top,
        "second_top": second_top,
        "second_answer": second_answer,
    }
