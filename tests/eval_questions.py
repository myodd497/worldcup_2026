"""Eval harness — golden questions scored by the LLM verifier.

Usage:
    poetry run python -m tests.eval_questions               # human-readable
    poetry run python -m tests.eval_questions --json out.json
    poetry run python -m tests.eval_questions --min-pass-rate 0.7

Exit codes:
    0 — passed the threshold
    1 — failed the threshold (or any question errored)

Designed to run in CI:
  - Every question runs through `run_orchestrator`.
  - Each answer is scored by `verifier_agent.verify` using the orchestrator's
    own SQL trace (so we measure groundedness, not just keyword presence).
  - A question PASSES iff: orchestrator returned a non-empty answer AND verifier
    says grounded AND answers_question AND confidence >= 0.6 AND all expected
    keywords appear (case-insensitive).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.orchestrator import run_orchestrator_full  # noqa: E402
from src.agents.verifier_agent import verify           # noqa: E402
from src.agents.workflow_logger import reset_tracker   # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")


@dataclass
class Eval:
    id: str
    question: str
    expects: tuple[str, ...]
    notes: str = ""


EVAL_SET: tuple[Eval, ...] = (
    Eval("Q1",  "Which teams will participate in the World Cup 2026?", ("team",)),
    Eval("Q2",  "For Mexico what is its current form? What were the last 10 games?", ("mexico",)),
    Eval("Q3",  "What are Portugal top 10 players with most goal contributions (goals + assists) in the last 10 games of Portugal?", ("portugal", "goal")),
    Eval("Q4",  "For Spain who is the top 10 players with worst discipline (yellow + red cards) in the last 10 games of Spain?", ("spain", "card")),
    Eval("Q5",  "For all WC2026 teams, which player has the most minutes played in the last month?", ("minute",)),
    Eval("Q6",  "What's the team with most shots on target and most shots conceded in the last 5 games?", ("shot",)),
    Eval("Q7",  "Teams with the best defense and best attack (goals conceded and goals scored).", ("goal",)),
    Eval("Q8",  "Teams with the highest and lowest ball possession percentage.", ("possession",)),
    Eval("Q9",  "Best attacking and defending team in the previous World Cup (2022).", ("2022",)),
    Eval("Q10", "Top 10 teams across all World Cup history (all editions in data).", ("team",)),
)

_MIN_CONFIDENCE = 0.6


async def _run_one(e: Eval) -> dict:
    reset_tracker()
    try:
        state = await run_orchestrator_full(
            user_message=e.question,
            user_id="eval",
            conversation_history=[],
        )
    except Exception as exc:
        return {"id": e.id, "question": e.question, "error": str(exc), "ok": False}

    answer = state.get("final_reply") or ""
    text = answer.lower()
    keyword_hits = [w for w in e.expects if w.lower() in text]
    keyword_ok = len(keyword_hits) == len(e.expects)

    primary_payload = state.get("agent_payload") or {}
    meta = primary_payload.get("metadata") or {}
    sql_executed = list(meta.get("sql_executed") or [])
    row_samples = list(meta.get("row_samples") or [])

    verdict = verify(
        question=e.question,
        answer=answer,
        sql_executed=sql_executed,
        row_samples=row_samples,
    )

    ok = bool(
        answer
        and keyword_ok
        and verdict.grounded
        and verdict.answers_question
        and verdict.confidence >= _MIN_CONFIDENCE
    )

    return {
        "id": e.id,
        "question": e.question,
        "answer": answer,
        "expected_keywords": list(e.expects),
        "keyword_hits": keyword_hits,
        "keyword_ok": keyword_ok,
        "selected_agent": state.get("selected_agent"),
        "selected_agents": state.get("selected_agents"),
        "confidence_score": state.get("confidence_score"),
        "sql_executed": sql_executed,
        "verifier": asdict(verdict),
        "ok": ok,
    }


async def _main(min_pass_rate: float, json_path: str | None) -> int:
    results: list[dict] = []
    for e in EVAL_SET:
        print(f"\n=== {e.id}: {e.question}", flush=True)
        r = await _run_one(e)
        ok = r.get("ok")
        print(f"--- ok={ok} verifier_conf={(r.get('verifier') or {}).get('confidence')}", flush=True)
        print((r.get("answer") or r.get("error", ""))[:600], flush=True)
        results.append(r)

    passed = sum(1 for r in results if r.get("ok"))
    total = len(results)
    pass_rate = passed / total if total else 0.0
    summary = {
        "passed": passed,
        "failed": total - passed,
        "total": total,
        "pass_rate": round(pass_rate, 3),
        "min_pass_rate": min_pass_rate,
    }
    print("\n========= SUMMARY =========")
    print(json.dumps(summary, indent=2))

    if json_path:
        with open(json_path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, default=str)

    return 0 if pass_rate >= min_pass_rate else 1


def _cli() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--min-pass-rate", type=float, default=float(os.getenv("EVAL_MIN_PASS_RATE", "0.7")))
    p.add_argument("--json", dest="json_path", default=os.getenv("EVAL_OUTPUT_JSON"))
    args = p.parse_args()
    return asyncio.run(_main(args.min_pass_rate, args.json_path))


if __name__ == "__main__":
    sys.exit(_cli())
