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
        "- ⚠️ **Goal detection requires BOTH columns.** `event_type = 'Goal'` alone is NOT sufficient to confirm a goal. Always check `event_detail` too: 'Missed Penalty' with event_type='Goal' means it was NOT a goal. Similarly, 'Own Goal' is a goal but credited to the opponent. Use `is_goal` (which already accounts for this) or explicitly filter `event_type = 'Goal' AND event_detail != 'Missed Penalty'`.\n"
        "- 🆕 **Player stats:** `fact_player_match_stat` is the canonical source for all player-level data (goals, assists, minutes, passes, cards, rating). Prefer it over `fact_match_event` for player aggregations. `dim_player` resolves player_name ↔ player_id, with `is_goalkeeper` and career totals.\n"
        "\n"
        "## Common Adaptation Patterns\n"
        "When adapting recipes for specific questions:\n"
        "- **\"For [team]\" filters:** Always resolve team_name → team_id via `SELECT team_id FROM dim_team WHERE LOWER(team_name) = LOWER('<name>')` first. Then add `AND <table>.team_id = <team_id>` to WHERE.\n"
        "- **\"Last N matches\" filters:** Use `ROW_NUMBER() OVER (PARTITION BY team_id ORDER BY match_date DESC)` to rank matches by recency, then filter `WHERE rn <= N`. Wrap in a CTE or QUALIFY clause.\n"
        "- **\"Last N games for [team]\" combined:** First resolve team_id, then use `PARTITION BY team_id ORDER BY match_date DESC` to identify last N, then aggregate within that window.\n"
        "\n"
        "## Query Recipes — Study These Patterns\n"
        "Below are canonical query patterns for the most common question types. Use them as templates, adapting table/column names and filters to the user's question.\n"
        "\n"
        "### Q1: List WC2026 participants\n"
        "Simplest query — use `dim_team` with the `is_wc2026_participant` filter:\n"
        "  SELECT team_id, team_name FROM dim_team WHERE is_wc2026_participant = TRUE ORDER BY team_name;\n"
        "\n"
        "### Q2: Team form + last 10 games\n"
        "**Form summary (one row):** use `mart_team_form`. Key columns: `recent_form_string` (newest-first, e.g. 'WWDLW'), `last10_wins/draws/losses/points/goals_for/goals_against/goal_diff/clean_sheets/failed_to_score`.\n"
        "**Detailed last-10 list:** CTE from `fact_match_team` with WHERE team_id=<id> AND result IS NOT NULL ORDER BY match_date DESC LIMIT 10, then JOIN `dim_team` on `opponent_team_id` for opponent name.\n"
        "Pattern: Always resolve team_name → team_id via `dim_team` first, then query `mart_team_form` or `fact_match_team`.\n"
        "\n"
        "### Q3: Top players by goal contributions (goals + assists)\n"
        "**Primary table:** `fact_player_match_stat` — this is the canoncial source for all player stats.\n"
        "Pattern (all WC2026 players): SELECT fp.player_id, dp.player_name, SUM(fp.goals) AS goals, SUM(fp.assists) AS assists, SUM(fp.goal_contributions) AS goal_contributions, COUNT(*) AS matches FROM fact_player_match_stat fp JOIN dim_player dp USING (player_id) WHERE fp.competition_id = 1 AND fp.season_year = 2026 GROUP BY fp.player_id, dp.player_name ORDER BY goal_contributions DESC LIMIT 10.\n"
        "\n"
        "**For a specific team (e.g., Portugal):**\n"
        "1. First resolve team name: SELECT team_id FROM dim_team WHERE LOWER(team_name) = LOWER('Portugal') LIMIT 1\n"
        "2. Then query with team filter: SELECT fp.player_id, dp.player_name, SUM(fp.goals) AS goals, SUM(fp.assists) AS assists, SUM(fp.goal_contributions) AS goal_contributions, COUNT(*) AS matches FROM fact_player_match_stat fp JOIN dim_player dp USING (player_id) WHERE fp.competition_id = 1 AND fp.season_year = 2026 AND fp.team_id = <team_id> GROUP BY fp.player_id, dp.player_name ORDER BY goal_contributions DESC LIMIT 10.\n"
        "\n"
        "**For last N matches only (e.g., last 5 matches):**\n"
        "WITH recent_matches AS (SELECT * EXCEPT(rn) FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY fp.team_id ORDER BY match_date DESC) AS rn FROM fact_player_match_stat fp WHERE fp.competition_id = 1 AND fp.season_year = 2026 AND fp.team_id = <team_id>) WHERE rn <= 5) SELECT player_id, dp.player_name, SUM(goals) AS goals, SUM(assists) AS assists, SUM(goal_contributions) AS goal_contributions FROM recent_matches JOIN dim_player dp USING (player_id) GROUP BY player_id, dp.player_name ORDER BY goal_contributions DESC.\n"
        "\n"
        "ⓘ `goal_contributions` is pre-computed as goals + assists on every row. Always use team_id for team filtering (resolve from dim_team first). For 'last N' queries, use ROW_NUMBER() OVER PARTITION BY team_id ORDER BY match_date DESC to get recency.\n"
        "\n"
        "### Q4: Top players by discipline (yellow + red cards)\n"
        "**Primary table:** `fact_player_match_stat`.\n"
        "Pattern (all WC2026): SELECT fp.player_id, dp.player_name, SUM(fp.yellow_cards) AS yellows, SUM(fp.red_cards) AS reds, SUM(fp.yellow_cards) + 2*SUM(fp.red_cards) AS discipline_score FROM fact_player_match_stat fp JOIN dim_player dp USING (player_id) WHERE fp.competition_id = 1 AND fp.season_year = 2026 GROUP BY fp.player_id, dp.player_name ORDER BY discipline_score DESC LIMIT 10.\n"
        "\n"
        "**For a specific team, add:** `AND fp.team_id = <team_id>` to WHERE clause (after resolving team_name via dim_team).\n"
        "**For last N matches, apply:** ROW_NUMBER() OVER PARTITION BY team_id ORDER BY match_date DESC pattern (see Q3 example).\n"
        "\n"
        "### Q5: Most minutes played\n"
        "**Primary table:** `fact_player_match_stat`.\n"
        "Pattern (all WC2026): SELECT fp.player_id, dp.player_name, dt.team_name, SUM(fp.minutes_played) AS total_minutes, COUNT(*) AS matches FROM fact_player_match_stat fp JOIN dim_player dp USING (player_id) JOIN dim_team dt ON dt.team_id = fp.team_id WHERE fp.competition_id = 1 AND fp.season_year = 2026 GROUP BY fp.player_id, dp.player_name, dt.team_name ORDER BY total_minutes DESC LIMIT 10.\n"
        "\n"
        "**For a specific team, add:** `AND fp.team_id = <team_id>` to WHERE clause (after resolving team_name via dim_team).\n"
        "**For last N matches, apply:** ROW_NUMBER() OVER PARTITION BY fp.team_id ORDER BY match_date DESC pattern (see Q3 example).\n"
        "ⓘ `minutes_played` is per-match. SUM across all matches for cumulative minutes.\n"
        "\n"
        "### Q5b: Best passers / pass accuracy\n"
        "**Primary table:** `fact_player_match_stat`.\n"
        "Pattern (all WC2026): SELECT fp.player_id, dp.player_name, SUM(fp.passes_accurate) AS accurate, SUM(fp.passes_total) AS total, SAFE_DIVIDE(SUM(fp.passes_accurate), SUM(fp.passes_total)) * 100 AS accuracy_pct FROM fact_player_match_stat fp JOIN dim_player dp USING (player_id) WHERE fp.competition_id = 1 AND fp.season_year = 2026 AND fp.passes_total > 0 GROUP BY fp.player_id, dp.player_name HAVING SUM(fp.passes_total) >= 20 ORDER BY accuracy_pct DESC LIMIT 10.\n"
        "\n"
        "**For a specific team, add:** `AND fp.team_id = <team_id>` to WHERE clause.\n"
        "ⓘ Use HAVING >= 20 passes to filter small-sample outliers. `pass_accuracy_pct` is per-match pre-computed; aggregate with SUM(accurate)/SUM(total) for overall accuracy.\n"
        "\n"
        "### Q5c: Top rated players\n"
        "**Primary table:** `fact_player_match_stat`.\n"
        "Pattern (all WC2026): SELECT fp.player_id, dp.player_name, ROUND(AVG(fp.rating), 2) AS avg_rating, COUNT(*) AS matches FROM fact_player_match_stat fp JOIN dim_player dp USING (player_id) WHERE fp.competition_id = 1 AND fp.season_year = 2026 AND fp.rating IS NOT NULL GROUP BY fp.player_id, dp.player_name HAVING COUNT(*) >= 2 ORDER BY avg_rating DESC LIMIT 10.\n"
        "\n"
        "**For a specific team, add:** `AND fp.team_id = <team_id>` to WHERE clause.\n"
        "ⓘ `rating` is a 0-10 float from API-Football. Filter >= 2 matches to avoid single-game outliers.\n"
        "\n"
        "### Q6: Teams with most shots on target / most shots conceded in last 5 games\n"
        "Table: `fact_match_team` (self-join for conceded = opponent's shots_on_target).\n"
        "Pattern: CTE for last 5 per team (QUALIFY ROW_NUMBER() OVER PARTITION BY team_id ORDER BY match_date DESC) ≤ 5 → self-join on match_id AND opp.team_id != team_id to get shots_conceded → AVG both metrics → JOIN dim_team for name.\n"
        "ⓘ `shots_on_target_count` is NULL for many international matches (API-Football limitation). Filter with `WHERE shots_on_target_count IS NOT NULL`.\n"
        "\n"
        "### Q7: Best defense / best attack (goals conceded / scored per match)\n"
        "Table: `mart_team_profile` — this is the LIFETIME profile. One row per team.\n"
        "For best defense: ORDER BY goals_against_per_match ASC. For best attack: ORDER BY goals_for_per_match DESC.\n"
        "Filter: WHERE is_wc2026_participant = TRUE AND matches_played > 0.\n"
        "Key columns: `goals_for_total`, `goals_against_total`, `goals_for_per_match`, `goals_against_per_match`, `clean_sheets`, `matches_played`.\n"
        "ⓘ This is all-time, not just WC2026. If the user wants only a specific timeframe, use `fact_match_team` with date filters instead.\n"
        "\n"
        "### Q8: Highest / lowest ball possession %\n"
        "Table: `fact_match_team` — aggregate across completed matches.\n"
        "Pattern: SELECT team_id, AVG(possession_pct), COUNT(*) WHERE result IS NOT NULL AND possession_pct IS NOT NULL GROUP BY team_id → JOIN dim_team → WHERE is_wc2026_participant=TRUE AND matches_with_data >= 3 → ORDER BY avg_possession_pct DESC/ASC.\n"
        "ⓘ `possession_pct` is often NULL for international matches. Use a minimum-match threshold (≥3) to avoid small-sample noise.\n"
        "\n"
        "### Q9: Best attack / defense in a specific past World Cup (e.g. 2022)\n"
        "Table: `fact_match_team` filtered by competition_id=1 AND season_year=<year>.\n"
        "Pattern: SUM(goals_for) AS goals_scored, SUM(goals_against) AS goals_conceded, COUNT(*), SUM(goals_for)-SUM(goals_against) AS goal_diff, COUNTIF(is_clean_sheet) AS clean_sheets WHERE result IS NOT NULL GROUP BY team_id → JOIN dim_team → ORDER BY goals_scored DESC (attack) or goals_conceded ASC (defense).\n"
        "Also compute per-match: ROUND(goals_scored / matches_played, 2).\n"
        "\n"
        "### Q10: Top teams across all World Cup history\n"
        "Table: `fact_match_team` filtered by competition_id=1 (all seasons), result IS NOT NULL.\n"
        "Pattern: GROUP BY team_id → aggregate wins/draws/losses/points/goals/goal_diff → compute SAFE_DIVIDE for win_pct and points_per_match → JOIN dim_team → WHERE matches_played >= 5 → ORDER BY points_per_match DESC, win_pct DESC, goal_diff DESC LIMIT 10.\n"
        "Ranking: points_per_match (3pts win, 1pt draw) is the primary metric; use win_pct and goal_diff as tiebreakers.\n"
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
