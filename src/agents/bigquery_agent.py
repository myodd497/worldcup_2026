"""BigQuery Agent — answers football questions over the gold datamodel.

Uses OpenAI function-calling to let the LLM autonomously:
  1. Inspect the catalog of available tables (marts/facts/dims)
  2. Describe / sample any table it wants more detail on
  3. Write and execute SELECT SQL against agent-visible tables
  4. Compose a grounded markdown answer

All BigQuery access is mediated through `src.tools.datamodel_tools` which
enforces read-only + allow-listed-table guardrails.

Public API (preserved for backwards compatibility):
  - run(query) -> str
  - run_structured(query) -> dict[answer, confidence_score, confidence_reason, metadata]
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.data.datamodel.catalog import format_catalog_for_llm
from src.tools.datamodel_tools import (
    describe_table_tool,
    list_tables_tool,
    run_sql_tool,
    sample_table_tool,
)

logger = logging.getLogger(__name__)

_MODEL_NAME = "gpt-4o-mini"
_MAX_TOOL_TURNS = 8


# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions (OpenAI function-calling schema)
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": (
                "List available tables in the World Cup datamodel. "
                "Optional 'layer' filter: 'mart', 'fact', or 'dim'. "
                "Returns markdown summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "layer": {"type": "string", "enum": ["mart", "fact", "dim"]}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": (
                "Get the full schema (columns, types, descriptions) and usage hint "
                "for a single table. Call this before writing SQL against a table you "
                "are unsure about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact table name."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_table",
            "description": "Return the first N rows of a table to inspect actual values. Max 20 rows.",
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
            "name": "run_sql",
            "description": (
                "Execute a read-only BigQuery SELECT/WITH statement against agent-visible "
                "tables. Table references may be bare names (e.g. `mart_team_form`) which "
                "will be auto-qualified. Returns rows as a list of dicts."
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
    """Route a function-call to the corresponding tool. Always returns a string."""
    try:
        if name == "list_tables":
            return list_tables_tool(layer=args.get("layer"))
        if name == "describe_table":
            return describe_table_tool(args["name"])
        if name == "sample_table":
            return sample_table_tool(args["name"], limit=args.get("limit", 5))
        if name == "run_sql":
            result = run_sql_tool(args["sql"])
            return json.dumps(result, default=str)
        return f"Error: unknown tool '{name}'"
    except Exception as exc:
        return f"Error executing tool '{name}': {exc}"


def _extract_sql_metrics(name: str, raw_result: str) -> dict[str, Any]:
    """Pull row_count + error from a run_sql tool result (JSON-encoded)."""
    if name != "run_sql":
        return {}
    try:
        parsed = json.loads(raw_result)
    except Exception:
        return {}
    return {
        "row_count": int(parsed.get("row_count", 0) or 0),
        "error":     parsed.get("error"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────


def _system_prompt() -> str:
    return (
        "You are a BigQuery analyst for a World Cup 2026 assistant.\n"
        "You answer football questions by querying the gold datamodel and grounding every fact in the rows you retrieve.\n"
        "\n"
        "## Workflow\n"
        "1. Read the table catalog below.\n"
        "2. Prefer MARTS — they are pre-aggregated and agent-shaped. Only fall back to FACTS when no mart fits.\n"
        "3. Use DIMS (especially `dim_team`) to resolve team names to team_id when needed.\n"
        "4. If unsure about a column, call `describe_table` or `sample_table` BEFORE writing SQL.\n"
        "5. Call `run_sql` with a single SELECT/WITH. Always include a LIMIT for lists (default 25).\n"
        "6. If a query fails, read the error and try once more with a corrected query.\n"
        "7. Once you have the data, write a concise markdown answer. Do not show SQL in the answer.\n"
        "\n"
        "## Conventions\n"
        "- All column names are snake_case. Primary keys are `<entity>_id`.\n"
        "- World Cup 2026: competition_id=1, season_year=2026. Use `is_wc2026_participant` on `dim_team`.\n"
        "- `result` columns use 'W'/'D'/'L'. `match_status` enum: SCHEDULED / LIVE / FINISHED / POSTPONED / CANCELLED / ABANDONED.\n"
        "- `mart_head_to_head` uses sorted pair: team_lo_id = LEAST(a,b), team_hi_id = GREATEST(a,b).\n"
        "- Suffixes: `_count` (integers), `_pct` (0-100 percentages), `_minutes` (durations), `_at` (timestamps), `_date` (dates).\n"
        "\n"
        "## Tables\n"
        f"{format_catalog_for_llm()}\n"
        "\n"
        "Be concise. Ground every claim in retrieved rows."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=_MODEL_NAME, temperature=0).bind_tools(_TOOLS_SCHEMA)


def _run_agent(query: str) -> tuple[str, list[dict[str, Any]]]:
    """Execute the function-calling loop. Returns (final_answer, trace)."""
    llm = _make_llm()
    messages: list[Any] = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user",   "content": query},
    ]
    trace: list[dict[str, Any]] = []

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


# ─────────────────────────────────────────────────────────────────────────────
# Confidence
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_structured(query: str) -> dict[str, Any]:
    try:
        answer, trace = _run_agent(query)
    except Exception as exc:
        logger.exception("bigquery_agent failed")
        return {
            "answer": f"I could not retrieve a warehouse answer.\nReason: {exc}",
            "confidence_score": 0.2,
            "confidence_reason": f"BigQuery agent failed: {exc}",
            "metadata": {"data_source": "bigquery", "error": str(exc)},
        }

    score, reason = _confidence(trace)
    tables_used: list[str] = []
    sql_executed: list[str] = []
    for step in trace:
        if step["tool"] == "run_sql":
            args = step.get("args") or {}
            sql_executed.append(str(args.get("sql", "")))
    return {
        "answer": answer,
        "confidence_score": score,
        "confidence_reason": reason,
        "metadata": {
            "data_source": "bigquery",
            "tool_calls": [{"tool": t["tool"], "args": t["args"]} for t in trace],
            "sql_executed": sql_executed,
        },
    }


def run(query: str) -> str:
    return run_structured(query)["answer"]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    q = " ".join(sys.argv[1:]) or "What is Portugal's next World Cup match?"
    out = run_structured(q)
    print(out["answer"])
    print("\n---")
    print(f"confidence={out['confidence_score']:.2f} — {out['confidence_reason']}")
    print(f"sql calls: {len(out['metadata'].get('sql_executed', []))}")
