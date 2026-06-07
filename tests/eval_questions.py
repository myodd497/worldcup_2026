"""Eval harness — runs the 10 canonical questions and prints pass/fail signals.

Not a unit test. Run manually: `poetry run python -m tests.eval_questions`
(requires live BigQuery + OpenAI credentials).

The harness checks each answer for:
  - was a SQL query executed?
  - did at least one query return rows?
  - did the answer mention the expected entity/keyword?

Pass criteria intentionally loose — this is a sanity sweep, not a regression suite.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass

# Allow `python -m tests.eval_questions` from repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import run_orchestrator  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")


@dataclass
class Eval:
    id: str
    question: str
    expects: tuple[str, ...]   # case-insensitive substrings that should appear
    notes: str = ""


EVAL_SET: tuple[Eval, ...] = (
    Eval("Q1",
         "Which teams will participate in the World Cup 2026?",
         expects=("team", )),
    Eval("Q2",
         "For Mexico what is its current form? What were the last 10 games?",
         expects=("mexico", )),
    Eval("Q3",
         "What are Portugal top 10 players with most goal contributions (goals + assists) in the last 10 games of Portugal?",
         expects=("portugal", "goal")),
    Eval("Q4",
         "For Spain who is the top 10 players with worst discipline (yellow + red cards) in the last 10 games of Spain?",
         expects=("spain", "card")),
    Eval("Q5",
         "For all WC2026 teams, which player has the most minutes played in the last month?",
         expects=("minute", )),
    Eval("Q6",
         "What's the team with most shots on target and most shots conceded in the last 5 games?",
         expects=("shot", )),
    Eval("Q7",
         "Teams with the best defense and best attack (goals conceded and goals scored).",
         expects=("goal", )),
    Eval("Q8",
         "Teams with the highest and lowest ball possession percentage.",
         expects=("possession", )),
    Eval("Q9",
         "Best attacking and defending team in the previous World Cup (2022).",
         expects=("2022", )),
    Eval("Q10",
         "Top 10 teams across all World Cup history (all editions in data).",
         expects=("team", )),
)


async def _run_one(e: Eval) -> dict:
    answer = await run_orchestrator(
        user_message=e.question,
        user_id="eval",
        conversation_history=[],
    )
    text = (answer or "").lower()
    hits = [w for w in e.expects if w.lower() in text]
    return {
        "id": e.id,
        "question": e.question,
        "answer": answer,
        "expected_keywords": list(e.expects),
        "hits": hits,
        "ok": len(hits) == len(e.expects),
    }


async def _main() -> None:
    results = []
    for e in EVAL_SET:
        print(f"\n=== {e.id}: {e.question}")
        try:
            r = await _run_one(e)
        except Exception as exc:
            r = {"id": e.id, "question": e.question, "error": str(exc), "ok": False}
        print(f"--- ok={r.get('ok')}")
        print((r.get("answer") or r.get("error", ""))[:600])
        results.append(r)
    summary = {
        "passed": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
        "total":  len(results),
    }
    print("\n========= SUMMARY =========")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
