"""BigQuery Agent — answers football questions over the gold datamodel.

Architecture (state-of-the-art):
  1. resolve_entity (deterministic team/player name → id)
  2. search_schema  (retrieves only the 3-5 relevant tables)
  3. few_shots      (retrieved Q→SQL examples — not hardcoded recipes)
    4. SQL generator  (complex LLM tier, function-calling)
  5. run_sql        (validate → dry-run + cost guard → execute)
  6. structured repair loop on failure (max 2 attempts with the actual error)
  7. compose grounded markdown answer

All BigQuery access is mediated through `src.tools.datamodel_tools` which
enforces read-only + allow-listed-table guardrails.

Public API (preserved):
  - run(query) -> str
  - run_structured(query) -> dict[answer, confidence_score, confidence_reason, metadata]
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.llm_config import create_chat_model
from src.agents.sql_few_shots import format_few_shots
from src.data.datamodel.schema_retriever import search_schema_tool
from src.tools.datamodel_tools import (
    data_freshness_tool,
    describe_table_tool,
    run_sql_tool,
    sample_table_tool,
)
from src.tools.entity_resolver import resolve_player_tool, resolve_team_tool

logger = logging.getLogger(__name__)

_MAX_TOOL_TURNS = 10
_MAX_SQL_ATTEMPTS = 3           # initial + 2 repairs


_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "resolve_team",
            "description": (
                "Resolve a team name (e.g. 'Portugal', 'USA', 'South Korea') to its "
                "canonical team_id. Returns JSON with id, name, confidence, and "
                "alternatives. ALWAYS use this BEFORE writing SQL that filters by team."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_player",
            "description": (
                "Resolve a player name to its canonical player_id. "
                "Returns JSON with id, name, confidence, and alternatives."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": (
                "Get the full schema for a single table. Call when the retrieved "
                "schema block is missing a column you need."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_table",
            "description": "Return the first N rows of a table to inspect actual values (max 20).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name":  {"type": "string"},
                    "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_freshness",
            "description": (
                "Return how recently the data warehouse was refreshed by the ETL. "
                "Call this FIRST whenever the question asks about today/yesterday/this week/live/recent data."
                " If the last successful ETL run is older than 6 hours, you MUST caveat the answer with the staleness."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a read-only BigQuery SELECT/WITH statement. Pre-flighted via "
                "BigQuery dry-run (catches column errors AND enforces a 1 GB cost cap). "
                "Bare table names are auto-qualified. "
                "Returns rows + columns or a clear error message you can act on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT or WITH query. No semicolons."},
                },
                "required": ["sql"],
            },
        },
    },
]


def _dispatch_tool(name: str, args: dict[str, Any]) -> str:
    try:
        if name == "resolve_team":
            return resolve_team_tool(args["name"])
        if name == "resolve_player":
            return resolve_player_tool(args["name"])
        if name == "describe_table":
            return describe_table_tool(args["name"])
        if name == "sample_table":
            return sample_table_tool(args["name"], limit=args.get("limit", 5))
        if name == "data_freshness":
            return data_freshness_tool()
        if name == "run_sql":
            result = run_sql_tool(args["sql"])
            return json.dumps(result, default=str)
        return f"Error: unknown tool '{name}'"
    except Exception as exc:
        return f"Error executing tool '{name}': {exc}"


def _extract_sql_metrics(name: str, raw_result: str) -> dict[str, Any]:
    if name != "run_sql":
        return {}
    try:
        parsed = json.loads(raw_result)
    except Exception:
        return {}
    return {
        "row_count": int(parsed.get("row_count", 0) or 0),
        "error":     parsed.get("error"),
        "bytes_billed_estimate": int(parsed.get("bytes_billed_estimate", 0) or 0),
    }


_SYSTEM_PROMPT = """\
You are a senior BigQuery analyst for a World Cup 2026 football assistant.
Every fact in your final answer MUST be grounded in rows you retrieved.

## Workflow (follow in order)
1. If the question references a team or player by NAME, FIRST call `resolve_team` / `resolve_player` to get the canonical id. NEVER write `LOWER(name) LIKE '%...%'`.
2. If the question is time-sensitive (today, yesterday, this week, live, recent, latest), FIRST call `data_freshness` and caveat the answer when the warehouse is stale.
3. Use the retrieved schema block + example queries below — they are pre-selected for this question. Read each table's **Usage hint** carefully; it contains domain rules you MUST follow.
4. Write ONE `run_sql` query. Prefer marts when one fits; fall back to facts only when no mart matches.
5. If `run_sql` returns an error, READ it carefully and try ONE corrected query (max 2 repairs).
6. Once you have the rows, write a concise markdown answer. Do not include the SQL.

## Universal rules
- Always include a LIMIT for lists (default 25). Aggregations don't need it.
- Snake_case columns. Primary keys are `<entity>_id`.
- If filtering by an entity id, the column lives on the fact/mart itself — use it directly. JOIN a dim only for human-readable names.
- If the retrieved tables clearly cannot answer the question, say so plainly — do not fabricate.

All domain-specific rules (enums, magic ids, partition keys, sorted pair keys, NULL handling) live in the per-table **Usage hint** sections below.
"""


def _build_user_prompt(
    question: str,
    conversation_context: str | None,
    repair: dict[str, Any] | None = None,
) -> str:
    schema_block = search_schema_tool(question, top_k=5)
    few_shot_block = format_few_shots(question, k=3)

    parts: list[str] = []
    if conversation_context and conversation_context != "None":
        parts.append("## Conversation context\n" + conversation_context)
    parts.append(schema_block)
    parts.append(few_shot_block)
    if repair:
        prior_sql = "\n---\n".join(repair.get("prior_sql", []))[:2000]
        parts.append(
            "## Reviewer feedback on the previous attempt (MUST address)\n"
            f"- Issues: {'; '.join(repair.get('issues', [])) or '(none)'}\n"
            f"- Hint: {repair.get('hint', '')}\n"
            f"- Previous answer (do NOT repeat verbatim):\n{repair.get('prior_answer', '')[:600]}\n"
            f"- SQL previously tried:\n```sql\n{prior_sql}\n```\n"
            "Write a NEW query that fixes the issues above. Do not repeat the same SQL."
        )
    parts.append("## User question\n" + question)
    return "\n\n".join(parts)


def _make_llm() -> ChatOpenAI:
    return create_chat_model("complex", temperature=0, max_retries=6, timeout=60, tools=True).bind_tools(_TOOLS_SCHEMA)


def _run_agent(
    question: str,
    conversation_context: str | None = None,
    repair: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    llm = _make_llm()
    messages: list[Any] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_prompt(question, conversation_context, repair=repair)},
    ]
    trace: list[dict[str, Any]] = []
    sql_attempts = 0

    for turn in range(_MAX_TOOL_TURNS):
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            return (ai_msg.content or "").strip(), trace

        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if name == "run_sql":
                sql_attempts += 1
                if sql_attempts > _MAX_SQL_ATTEMPTS:
                    result = json.dumps({
                        "error": "Max SQL attempts reached. Stop and answer with what you have.",
                        "row_count": 0,
                    })
                    trace.append({"turn": turn, "tool": name, "args": args, "result_preview": result[:500]})
                    messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": result})
                    continue

            result = _dispatch_tool(name, args or {})
            metrics = _extract_sql_metrics(name, result)
            trace.append({
                "turn": turn,
                "tool": name,
                "args": args,
                "result_preview": result[:500] if isinstance(result, str) else str(result)[:500],
                **metrics,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": result,
            })

    return (
        "I ran out of reasoning turns before producing an answer. Please rephrase the question.",
        trace,
    )


def _confidence(trace: list[dict[str, Any]]) -> tuple[float, str]:
    sql_calls = [t for t in trace if t["tool"] == "run_sql"]
    if not sql_calls:
        return 0.3, "Agent answered without querying BigQuery."
    last_error = next((c.get("error") for c in reversed(sql_calls) if c.get("error")), None)
    if last_error and all(c.get("error") for c in sql_calls):
        return 0.3, f"All SQL queries failed. Last error: {last_error}"
    total_rows = sum(int(c.get("row_count") or 0) for c in sql_calls)
    if total_rows == 0:
        return 0.5, "Queries ran but returned zero rows."
    return 0.85, f"Answer grounded in {len(sql_calls)} SQL call(s), {total_rows} row(s) total."


def run_structured(
    query: str,
    conversation_context: str | None = None,
    repair: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        answer, trace = _run_agent(query, conversation_context=conversation_context, repair=repair)
    except Exception as exc:
        logger.exception("bigquery_agent failed")
        return {
            "answer": f"I could not retrieve a warehouse answer.\nReason: {exc}",
            "confidence_score": 0.2,
            "confidence_reason": f"BigQuery agent failed: {exc}",
            "metadata": {"data_source": "bigquery", "error": str(exc)},
        }

    score, reason = _confidence(trace)
    sql_executed: list[str] = []
    row_samples: list[dict] = []
    for step in trace:
        if step["tool"] == "run_sql":
            args = step.get("args") or {}
            sql_executed.append(str(args.get("sql", "")))
            try:
                parsed = json.loads(step.get("result_preview", "{}"))
                rows = parsed.get("rows") or []
                row_samples.extend(rows[:5])
            except Exception:
                pass

    return {
        "answer": answer,
        "confidence_score": score,
        "confidence_reason": reason,
        "metadata": {
            "data_source": "bigquery",
            "tool_calls": [{"tool": t["tool"], "args": t["args"]} for t in trace],
            "sql_executed": sql_executed,
            "row_samples": row_samples[:10],
        },
    }


def run(query: str) -> str:
    return run_structured(query)["answer"]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    q = " ".join(sys.argv[1:]) or "Which teams will participate in the World Cup 2026?"
    out = run_structured(q)
    print(out["answer"])
    print("\n---")
    print(f"confidence={out['confidence_score']:.2f} — {out['confidence_reason']}")
    print(f"sql calls: {len(out['metadata'].get('sql_executed', []))}")
