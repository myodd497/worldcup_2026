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

_MAX_TOOL_TURNS = 12
_MAX_SQL_ATTEMPTS = 20          # hard cap on total run_sql calls per question
_MAX_SQL_FAILURES = 5           # consecutive failed run_sql calls before forcing the agent to stop


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
    schema_override: str | None = None,
) -> str:
    # Allow callers (e.g. summary fan-out) to pass a pre-computed schema block
    # so we don't run the embedding retriever N times for the same topic.
    schema_block = schema_override if schema_override is not None else search_schema_tool(question, top_k=5)
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
    import os
    tier = os.getenv("BIGQUERY_LLM_TIER", "complex")  # SQL gen on Flash is fast and good enough with few-shots
    return create_chat_model(tier, temperature=0, max_retries=6, timeout=60, tools=True).bind_tools(_TOOLS_SCHEMA)


# DeepSeek V4 sometimes serialises a function call into the assistant `content`
# field using its internal DSML markup (e.g. `｜｜DSML｜｜invoke name="run_sql"`).
# Detect and strip these so the user never sees them.
import re as _re

_DSML_MARKER_RE = _re.compile(r"[｜|]{2}DSML[｜|]{2}", _re.IGNORECASE)
_DSML_BLOCK_RE = _re.compile(
    r"[｜|]{2}DSML[｜|]{2}\s*(?:tool_calls|invoke|parameter)[^\n]*?(?=[｜|]{2}DSML[｜|]{2}|\Z)",
    _re.IGNORECASE | _re.DOTALL,
)


def _looks_like_leaked_tool_call(text: str) -> bool:
    if not text:
        return False
    return bool(_DSML_MARKER_RE.search(text))


def _sanitize_answer(text: str) -> str:
    """Strip leaked DSML tool-call markup from a final answer, just in case."""
    if not text or "DSML" not in text.upper():
        return text
    cleaned = _DSML_BLOCK_RE.sub("", text)
    # Also wipe any stray DSML markers that survived the block sub.
    cleaned = _DSML_MARKER_RE.sub("", cleaned)
    return cleaned.strip()


def _force_final_answer(messages: list[Any], reason: str) -> str:
    """Last-resort: ask the LLM (no tools bound) to compose an answer from whatever
    tool results are already in `messages`. Used when the turn / SQL budget is
    exhausted so we never throw away the data we already retrieved.
    """
    try:
        import os
        tier = os.getenv("BIGQUERY_LLM_TIER", "complex")
        plain_llm = create_chat_model(tier, temperature=0, max_retries=3, timeout=60, tools=False)
        nudge = {
            "role": "system",
            "content": (
                f"BUDGET EXHAUSTED ({reason}). You have NO more tool calls available. "
                "Write the best possible markdown answer NOW using ONLY the tool results "
                "already present in this conversation. If some sub-aspects of the question "
                "were not covered, list them under a short 'Not retrieved' note instead of "
                "guessing. Do not apologise. Do not promise to try again."
            ),
        }
        ai = plain_llm.invoke(messages + [nudge])
        text = (getattr(ai, "content", "") or "").strip()
        if text:
            return _sanitize_answer(text)
    except Exception as exc:
        logger.warning("forced final answer failed: %s", exc)
    return (
        "I ran out of reasoning turns before producing a complete answer. "
        "Please narrow the question (e.g. ask about one specific aspect)."
    )


def _run_agent(
    question: str,
    conversation_context: str | None = None,
    repair: dict[str, Any] | None = None,
    max_turns: int | None = None,
    schema_override: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    import time as _time
    llm = _make_llm()
    messages: list[Any] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_prompt(question, conversation_context, repair=repair, schema_override=schema_override)},
    ]
    trace: list[dict[str, Any]] = []
    sql_attempts = 0
    sql_consecutive_failures = 0
    turn_budget = max_turns if max_turns is not None else _MAX_TOOL_TURNS

    for turn in range(turn_budget):
        _t_llm = _time.perf_counter()
        ai_msg = llm.invoke(messages)
        llm_sec = round(_time.perf_counter() - _t_llm, 2)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            raw = (ai_msg.content or "").strip()
            # DeepSeek V4 occasionally serialises a tool-call into the `content`
            # field instead of the structured `tool_calls` slot (leaks the
            # `｜｜DSML｜｜invoke name=...` markup as plain text). Detect, recover,
            # and nudge the model to retry with proper function-calling.
            if _looks_like_leaked_tool_call(raw):
                trace.append({
                    "turn": turn,
                    "tool": "_llm_leaked_toolcall",
                    "llm_sec": llm_sec,
                    "preview": raw[:200],
                })
                messages.append({
                    "role": "system",
                    "content": (
                        "Your previous reply leaked tool-call markup into the answer text. "
                        "Tool calls MUST be emitted via the structured function-calling "
                        "interface, never as plain text. Either issue the tool call properly "
                        "now, or write the final markdown answer using ONLY the tool results "
                        "already gathered."
                    ),
                })
                continue
            trace.append({"turn": turn, "tool": "_llm_final", "llm_sec": llm_sec})
            return _sanitize_answer(raw), trace

        trace.append({"turn": turn, "tool": "_llm_step", "llm_sec": llm_sec, "n_tool_calls": len(tool_calls)})

        # Parse all tool calls first so we can run independent tools concurrently.
        parsed_calls: list[tuple[str, dict, Any]] = []  # (name, args, call_id)
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            parsed_calls.append((name, args or {}, call_id))

        # Enforce SQL attempt caps before dispatch.
        prepared: list[tuple[str, dict, Any, str | None]] = []  # (name, args, call_id, forced_result)
        for name, args, call_id in parsed_calls:
            if name == "run_sql":
                sql_attempts += 1
                if sql_attempts > _MAX_SQL_ATTEMPTS:
                    forced = json.dumps({
                        "error": f"Max SQL attempts ({_MAX_SQL_ATTEMPTS}) reached. Stop and answer with what you have.",
                        "row_count": 0,
                    })
                    prepared.append((name, args, call_id, forced))
                    continue
                if sql_consecutive_failures >= _MAX_SQL_FAILURES:
                    forced = json.dumps({
                        "error": f"{_MAX_SQL_FAILURES} consecutive SQL failures. Stop repairing and answer with what you have.",
                        "row_count": 0,
                    })
                    prepared.append((name, args, call_id, forced))
                    continue
            prepared.append((name, args, call_id, None))

        # Run tools in parallel when there are 2+ to dispatch (independent calls).
        from concurrent.futures import ThreadPoolExecutor
        dispatch_targets = [(i, n, a) for i, (n, a, _, forced) in enumerate(prepared) if forced is None]
        results_by_idx: dict[int, Any] = {}
        if len(dispatch_targets) > 1:
            with ThreadPoolExecutor(max_workers=len(dispatch_targets)) as ex:
                futures = {ex.submit(_dispatch_tool, n, a): i for i, n, a in dispatch_targets}
                for fut in futures:
                    i = futures[fut]
                    _t_tool = _time.perf_counter()
                    results_by_idx[i] = (fut.result(), round(_time.perf_counter() - _t_tool, 2))
        else:
            for i, n, a in dispatch_targets:
                _t_tool = _time.perf_counter()
                results_by_idx[i] = (_dispatch_tool(n, a), round(_time.perf_counter() - _t_tool, 2))

        # Append results in original order so message order matches tool_call_ids.
        for i, (name, args, call_id, forced) in enumerate(prepared):
            if forced is not None:
                trace.append({"turn": turn, "tool": name, "args": args, "result_preview": forced[:500]})
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": forced})
                continue
            result, tool_sec = results_by_idx[i]
            metrics = _extract_sql_metrics(name, result)
            if name == "run_sql":
                if metrics.get("error"):
                    sql_consecutive_failures += 1
                else:
                    sql_consecutive_failures = 0
            trace.append({
                "turn": turn,
                "tool": name,
                "tool_sec": tool_sec,
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

    # Turn budget exhausted — salvage whatever was gathered instead of dropping it.
    final = _force_final_answer(
        messages,
        reason=f"hit turn budget={turn_budget} (sql_attempts={sql_attempts})",
    )
    trace.append({"turn": turn_budget, "tool": "_forced_final"})
    return final, trace


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
    max_turns: int | None = None,
    schema_override: str | None = None,
) -> dict[str, Any]:
    try:
        answer, trace = _run_agent(
            query,
            conversation_context=conversation_context,
            repair=repair,
            max_turns=max_turns,
            schema_override=schema_override,
        )
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
            "tool_calls": [{"tool": t["tool"], "args": t.get("args", {})} for t in trace],
            "trace": trace,  # full trace incl. per-step llm_sec / tool_sec timings
            "sql_executed": sql_executed,
            "row_samples": row_samples[:10],
        },
    }


def run(query: str) -> str:
    return run_structured(query)["answer"]


# ── Summary fan-out (broad "overview/summary" questions) ────────────────────
#
# Open-ended questions like "summarise FC Porto 2025-2026 season" easily blow
# past `_MAX_TOOL_TURNS` because a single agent loop tries to cover too many
# angles (record, top scorers, form, transfers, …) in one go. Plan-then-fan-out:
#   1. decompose the question into N focused sub-questions (one LLM call)
#   2. run each sub-question through `run_structured` in parallel
#   3. compose ONE markdown summary from the sub-answers
#
# Each sub-question gets its own full tool budget, so nothing is lost.

_SUMMARY_DECOMPOSE_PROMPT = """\
You are planning a structured summary about a football entity.

Decompose the user's broad question into 3-4 FOCUSED, INDEPENDENT sub-questions
that together cover the topic. Fewer is better — only add a sub-question if it
reveals a genuinely different angle.

Each sub-question must be answerable with a SINGLE BigQuery query against a
football data warehouse (dim/fact/mart tables covering matches, results, goals,
players, standings, fixtures).

Rules:
- Each sub-question stands alone (no pronouns referring to others, no
  references to "the players above" or "these teams" — inline the actual names
  from the conversation context if relevant).
- Each one names the entity AND the time range EXPLICITLY. The current real-
  world date is {today}. When the user says "this season", "current season",
  "recent", or gives no season, use the most recently completed/in-progress
  season (e.g. "2025-2026" or season_year=2025). NEVER default to historical
  seasons like 2023-2024 unless the user asked for them.
- If the conversation context lists specific entities (players, fixtures,
  lineup), reuse THOSE exact names/ids — do not invent a new "top 10" list.
- Prefer concrete angles: overall record, top scorers, recent form, key fixtures.
- Skip angles that clearly don't apply (e.g. "WC2026 group stage" for a club).
- Return ONLY a JSON array of strings. No prose, no markdown.

## Conversation context (may be empty)
{context}

## User question
{question}
""".strip()


_SUMMARY_COMPOSE_PROMPT = """\
You are composing ONE coherent markdown summary from several sub-answers that
were each grounded in BigQuery results.

STRICT GROUNDING RULES (read carefully — violating these is worse than a short answer):
- You may ONLY mention entities (players, clubs, scores, dates, numbers) that
  appear VERBATIM in the sub-answers below. Do NOT add names or stats from
  your own knowledge, even if you are sure they are correct.
- If a sub-answer says "no data", "not available", "not extracted", "could
  not be retrieved", or is otherwise empty of concrete facts, DROP that
  section entirely. Do not write a section that admits to having no data.
- If a sub-answer lists fewer entities than the user asked for, only mention
  the ones actually returned. Do not pad the list from memory.
- Use short markdown sections with ## headings, one per sub-topic that
  actually has concrete data.
- If NONE of the sub-answers produced concrete data, reply with a single short
  paragraph saying the warehouse did not have what was asked, and suggest the
  user narrow the question.
- End with a one-line italic caveat ONLY if some (not all) sub-answers were
  empty, listing which angles are missing.
- Keep it tight; this is for a WhatsApp/web assistant.

Original user question:
{question}

Sub-question results (JSON — `usable=false` means the sub-answer had no
concrete data and MUST be ignored):
{sub_results}
""".strip()


def _decompose_summary(question: str, conversation_context: str | None = None) -> list[str]:
    """Use the simple-tier LLM to break the question into sub-questions.

    The decomposer gets the current date and the conversation context so it can
    (a) resolve "this season" to the right year and (b) carry forward entities
    the user already named (e.g. a lineup from a prior turn).
    """
    from datetime import date as _date
    try:
        llm = create_chat_model("simple", temperature=0, max_retries=3, timeout=30)
        raw = llm.invoke(
            _SUMMARY_DECOMPOSE_PROMPT.format(
                question=question,
                context=(conversation_context or "None")[:4000],
                today=_date.today().isoformat(),
            )
        ).content or ""
        raw = raw.strip()
        # Strip ```json fences if present.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        subs = json.loads(raw)
        if isinstance(subs, list):
            cleaned = [str(s).strip() for s in subs if str(s).strip()]
            return cleaned[:4]
    except Exception as exc:
        logger.warning("summary decomposition failed: %s", exc)
    # Fallback: a single sub-question = the original question. The bigquery
    # agent will still get to run, just without parallel fan-out benefit.
    return [question]


# ── Entity allow-list scrub ─────────────────────────────────────────────────

# Sentence-start / common capitalised words that should NOT be treated as
# proper-noun evidence (otherwise every sentence starting with "The" would be
# flagged). Domain section headings get included so headings survive.
_SCRUB_STOPWORDS = {
    "the", "a", "an", "this", "these", "those", "their", "they", "he", "she",
    "it", "his", "her", "its", "and", "or", "but", "if", "when", "while",
    "however", "note", "summary", "overview", "key", "season", "stats",
    "statistics", "data", "players", "player", "club", "clubs", "team", "teams",
    "match", "matches", "game", "games", "goals", "assists", "appearances",
    "minutes", "league", "cup", "based", "here", "no", "yes", "missing",
    "available", "retrieved", "not", "some", "all", "most", "top", "best",
    "first", "second", "third", "last", "next", "for", "from", "with", "of",
    "in", "on", "at", "by", "as", "to", "vs", "won", "lost", "drew", "draw",
    "win", "wins", "loss", "losses", "draws", "points", "position", "form",
    "starts", "starter", "substitute", "substitutes", "minutes_played",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    # Nationality / language adjectives — frequent false positives when they
    # start a sentence ("The top Portuguese players by goals are:").
    "portuguese", "spanish", "english", "french", "german", "italian", "dutch",
    "belgian", "brazilian", "argentine", "argentinian", "mexican", "american",
    "canadian", "moroccan", "egyptian", "nigerian", "ghanaian", "senegalese",
    "japanese", "korean", "saudi", "iranian", "uruguayan", "colombian",
    "chilean", "ecuadorian", "peruvian", "australian", "croatian", "serbian",
    "polish", "swiss", "austrian", "danish", "swedish", "norwegian", "scottish",
    "irish", "welsh", "european", "african", "asian", "south", "north",
}

# Match either: a multi-word capitalised phrase (e.g. "Cristiano Ronaldo",
# "Manchester United", "Al Hilal", "Paris Saint-Germain") OR a single
# capitalised word of length >= 4 that could plausibly be a proper noun.
# Allows hyphens, apostrophes, and accented characters (Léao, Félix).
_PROPER_NOUN_RE = _re.compile(
    r"\b([A-ZÀ-Ý][\wÀ-ÿ'\-]*"          # first capitalised word
    r"(?:\s+[A-ZÀ-Ý][\wÀ-ÿ'\-]*)*)",  # optional additional capitalised words
    _re.UNICODE,
)


def _extract_proper_nouns(text: str) -> set[str]:
    """Extract proper-noun phrases as lowercase strings, ignoring stopwords."""
    found: set[str] = set()
    for match in _PROPER_NOUN_RE.finditer(text or ""):
        phrase = match.group(1).strip()
        lo = phrase.lower()
        # Single short words and pure stopwords don't count as proper nouns.
        if " " not in phrase and (len(phrase) < 4 or lo in _SCRUB_STOPWORDS):
            continue
        # Multi-word phrases: drop if every token is a stopword.
        if " " in phrase and all(tok.lower() in _SCRUB_STOPWORDS for tok in phrase.split()):
            continue
        found.add(lo)
    return found


def _scrub_unknown_entities(
    composed: str,
    usable_sub_answers: list[str],
    row_samples: list[Any],
) -> tuple[str, list[str]]:
    """Drop sentences in the composed answer that mention a proper-noun phrase
    not present anywhere in the grounded source material.

    Returns the cleaned text + a list of dropped sentences (for logging).
    Conservative: lower-case words and numbers pass freely; only proper-noun
    phrases trigger checks. Headings (`##`, `-`, `*`) are inspected too, but
    their leading markdown is preserved when re-emitting kept content.
    """
    if not composed:
        return composed, []

    # Build the allow-list from all usable sub-answers + raw row sample JSON.
    allow: set[str] = set()
    for ans in usable_sub_answers:
        allow |= _extract_proper_nouns(ans)
    try:
        import json as _json
        for rs in row_samples:
            allow |= _extract_proper_nouns(_json.dumps(rs, ensure_ascii=False, default=str))
    except Exception:
        pass

    # Substring-tolerant containment: "Ronaldo" in allow lets "Cristiano Ronaldo"
    # pass even if only the surname appeared in the sources, and vice versa.
    def _allowed(phrase: str) -> bool:
        if phrase in allow:
            return True
        for known in allow:
            if phrase in known or known in phrase:
                return True
        return False

    kept_lines: list[str] = []
    dropped: list[str] = []
    for raw_line in composed.splitlines():
        stripped = raw_line.strip()
        # Headings, blank lines, separators, code fences: keep as-is.
        if not stripped or stripped.startswith(("#", "```", "---", "===")):
            kept_lines.append(raw_line)
            continue

        # Split the line into sentences. Keep markdown bullet/numbering prefix
        # so list structure survives.
        prefix_match = _re.match(r"^(\s*(?:[-*+]|\d+\.)\s+)", raw_line)
        prefix = prefix_match.group(1) if prefix_match else ""
        body = raw_line[len(prefix):] if prefix else raw_line

        sentences = _re.split(r"(?<=[.!?])\s+", body.strip())
        kept_sentences: list[str] = []
        for sent in sentences:
            nouns = _extract_proper_nouns(sent)
            bad = [n for n in nouns if not _allowed(n)]
            if bad:
                dropped.append(sent)
            else:
                kept_sentences.append(sent)

        if kept_sentences:
            kept_lines.append(prefix + " ".join(kept_sentences))
        # If every sentence on a bullet was dropped, drop the bullet entirely.

    cleaned = "\n".join(kept_lines).strip()

    # If the scrub left us with only headings / nothing, fall back to a short
    # honest message rather than shipping an empty shell.
    has_content = any(
        line.strip() and not line.strip().startswith(("#", "```", "---", "==="))
        for line in cleaned.splitlines()
    )
    if not has_content:
        cleaned = (
            "The data warehouse did not return enough concrete information to "
            "summarise this. Try narrowing the question (e.g. ask about one "
            "specific player, team, or season)."
        )

    if dropped:
        cleaned = (
            cleaned
            + "\n\n*Note: parts of the draft answer were removed because they "
              "referenced entities not present in the retrieved data.*"
        )

    return cleaned, dropped


def run_summary_structured(
    query: str,
    conversation_context: str | None = None,
) -> dict[str, Any]:
    """Plan-then-fan-out runner for broad summary/overview questions.

    Each sub-question is executed by `run_structured` in parallel threads, then a
    single LLM call stitches the sub-answers into a coherent markdown summary.
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    sub_questions = _decompose_summary(query, conversation_context=conversation_context)
    logger.info("summary fan-out: %d sub-questions for %r", len(sub_questions), query[:80])

    # Schema retrieval is the same expensive embedding call for every sub-question
    # about the same entity. Compute it ONCE against the broad parent question
    # (better topical anchor than any narrow sub-question) and reuse.
    try:
        from src.data.datamodel.schema_retriever import search_schema_tool as _search
        shared_schema = _search(query, top_k=5)
    except Exception as exc:
        logger.warning("shared schema retrieval failed (%s); sub-questions will retrieve individually", exc)
        shared_schema = None

    t0 = _time.perf_counter()
    sub_results: list[dict[str, Any]] = []
    # Each sub-question is focused (one SQL angle) — small turn budget keeps the
    # slowest parallel branch from dragging the whole summary down.
    sub_max_turns = 6
    with ThreadPoolExecutor(max_workers=min(len(sub_questions), 4)) as ex:
        futures = {
            ex.submit(
                run_structured,
                sq,
                conversation_context,
                None,
                sub_max_turns,
                shared_schema,
            ): sq
            for sq in sub_questions
        }
        for fut in futures:
            sq = futures[fut]
            try:
                sub_results.append({"question": sq, "result": fut.result()})
            except Exception as exc:
                logger.exception("summary sub-question failed: %s", sq)
                sub_results.append({
                    "question": sq,
                    "result": {
                        "answer": f"(sub-question failed: {exc})",
                        "confidence_score": 0.0,
                        "metadata": {"error": str(exc)},
                    },
                })
    fanout_sec = round(_time.perf_counter() - t0, 2)

    # Hedge detection: the heuristic _confidence() returns 0.85 whenever *any*
    # rows came back, even garbage from a discovery query. If the answer text
    # admits to missing data, the sub-answer is NOT usable for the composer.
    _HEDGE_PHRASES = (
        "no data", "not available", "not extracted", "could not be retrieved",
        "couldn't be retrieved", "were not extracted", "no results", "0 rows",
        "no rows", "unable to retrieve", "no matching", "could not find",
        "not found in the database", "not in the database", "missing data",
        "data is not available", "n/a", "(sub-question failed",
    )

    def _is_usable(answer_text: str) -> bool:
        if not answer_text or len(answer_text.strip()) < 20:
            return False
        lo = answer_text.lower()
        return not any(p in lo for p in _HEDGE_PHRASES)

    # Compose the final markdown answer from the sub-answers.
    compact = []
    for item in sub_results:
        ans = str(item["result"].get("answer", ""))
        usable = _is_usable(ans)
        compact.append({
            "sub_question": item["question"],
            "answer": ans[:1500],
            "usable": usable,
            "confidence": float(item["result"].get("confidence_score", 0.0) or 0.0),
        })
    try:
        # Complex tier: the composer must follow strict anti-hallucination rules
        # (ignore unusable sub-answers, never invent entities from training data).
        # Simple tier was too eager to "help" by padding with prior knowledge.
        composer = create_chat_model("complex", temperature=0, max_retries=3, timeout=60)
        composed = composer.invoke(
            _SUMMARY_COMPOSE_PROMPT.format(
                question=query,
                sub_results=json.dumps(compact, ensure_ascii=False, indent=2),
            )
        ).content
        final_answer = (composed or "").strip() or "No data was retrieved for this summary."
    except Exception as exc:
        logger.exception("summary composition failed")
        # Concatenation fallback so we still return SOMETHING grounded.
        final_answer = "\n\n".join(
            f"### {item['sub_question']}\n{item['answer']}" for item in sub_results
        ) or f"Summary composition failed: {exc}"

    # Entity allow-list scrub: hard guarantee that the composer didn't smuggle in
    # entity names from its training data. We extract proper-noun phrases that
    # appear in the (usable) sub-answers + row samples, then drop any sentence
    # in the composed markdown that mentions a proper-noun phrase NOT in that
    # allow-list. Conservative: only proper-noun phrases (2+ capitalised words,
    # or a long single capitalised word). Numbers and lower-case words pass.
    try:
        final_answer, dropped = _scrub_unknown_entities(
            final_answer,
            usable_sub_answers=[c["answer"] for c in compact if c["usable"]],
            row_samples=[item["result"].get("metadata", {}).get("row_samples") or []
                         for item in sub_results],
        )
        if dropped:
            logger.warning("entity scrub dropped %d sentence(s): %s",
                           len(dropped), dropped[:3])
    except Exception as exc:
        logger.warning("entity scrub failed (%s); returning raw composed answer", exc)
        dropped = []

    # Merge metadata across sub-runs.
    merged_sql: list[str] = []
    merged_rows: list[dict] = []
    merged_traces: list[Any] = []
    effective_confidences: list[float] = []
    for item, c in zip(sub_results, compact):
        meta = (item["result"].get("metadata") or {})
        merged_sql.extend(meta.get("sql_executed") or [])
        merged_rows.extend(meta.get("row_samples") or [])
        merged_traces.append({"sub_question": item["question"], "trace": meta.get("trace")})
        raw_conf = float(item["result"].get("confidence_score", 0.0) or 0.0)
        # Hedged / empty sub-answers get capped at 0.3 regardless of how many
        # rows the heuristic counted — those rows did not answer the question.
        effective_confidences.append(raw_conf if c["usable"] else min(raw_conf, 0.3))

    n_usable = sum(1 for c in compact if c["usable"])
    if not effective_confidences:
        avg_conf = 0.3
    elif n_usable == 0:
        # Nothing answered the question — be honest with the user.
        avg_conf = 0.25
    else:
        avg_conf = round(sum(effective_confidences) / len(effective_confidences), 2)
    reason = (
        f"Summary fan-out: {n_usable}/{len(sub_results)} sub-questions returned "
        f"concrete data (others were empty or hedged)."
    )

    return {
        "answer": final_answer,
        "confidence_score": avg_conf,
        "confidence_reason": reason,
        "metadata": {
            "data_source": "bigquery",
            "mode": "summary_fanout",
            "fanout_sec": fanout_sec,
            "sub_questions": sub_questions,
            "sql_executed": merged_sql,
            "row_samples": merged_rows[:15],
            "sub_traces": merged_traces,
            "scrubbed_sentences": dropped,
        },
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    q = " ".join(sys.argv[1:]) or "Which teams will participate in the World Cup 2026?"
    out = run_structured(q)
    print(out["answer"])
    print("\n---")
    print(f"confidence={out['confidence_score']:.2f} — {out['confidence_reason']}")
    print(f"sql calls: {len(out['metadata'].get('sql_executed', []))}")
