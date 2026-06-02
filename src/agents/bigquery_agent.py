"""BigQuery Agent — multi-step planning over the World Cup warehouse.

Pipeline:
1. Load schema metadata (columns + descriptions) for `fact_*`, `dim_*`, `v_*`.
2. Load business contract (app_allowed objects + business_concept) from
   `v_data_contract_tables` with a static fallback.
3. Extract entities from the user question (team names, season, fixture intent).
4. Resolve team names -> team_id via `dim_team`.
5. Plan which tables to use (LLM-guided, restricted to app_allowed prefixes).
6. Generate one or more SQL queries with full schema + resolved IDs in context.
7. Execute queries, repair invalid SQL once using BigQuery error feedback.
8. Compose a final answer that merges all query results.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from langchain_openai import ChatOpenAI

from src.tools.bigquery_tools import run_query


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_ALLOWED_PREFIXES = ("fact_", "dim_", "v_")
_SCHEMA_CACHE_TTL_SECONDS = 600
_MAX_QUERIES = 4
_FALLBACK_DATASET = "worldcup2026"


# ─────────────────────────────────────────────────────────────────────────────
# Static catalog (used as fallback if v_data_contract_tables is unreachable)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WarehouseObject:
    name: str
    kind: str
    usage: str
    business_concept: str
    description: str
    key_columns: tuple[str, ...] = ()
    notes: str = ""


WAREHOUSE_OBJECTS: tuple[WarehouseObject, ...] = (
    WarehouseObject("fact_fixture", "canonical table", "app_allowed", "fixture",
                    "One row per fixture with home/away team IDs, score, venue, referee.",
                    ("fixture_id",)),
    WarehouseObject("fact_team_fixture", "canonical table", "app_allowed", "team fixture participation",
                    "Team-centric match history with result, goals_scored, goals_conceded, was_home, opponent_id.",
                    ("team_id", "fixture_id")),
    WarehouseObject("fact_fixture_event", "canonical table", "app_allowed", "fixture event",
                    "Match events: goals, cards, substitutions, VAR with player/team and minute.",
                    ("event_key",)),
    WarehouseObject("fact_fixture_team_stat", "canonical table", "app_allowed", "fixture team statistic",
                    "Per-team match stats: possession, shots, fouls (long format).",
                    ("fixture_id", "team_id", "stat_type")),
    WarehouseObject("dim_team", "canonical dimension", "app_allowed", "team",
                    "Team master. Resolve human team names to team_id here.",
                    ("team_id",)),
    WarehouseObject("dim_competition", "canonical dimension", "app_allowed", "competition",
                    "Competition master.", ("competition_id",)),
    WarehouseObject("dim_venue", "canonical dimension", "app_allowed", "venue",
                    "Venue master.", ("venue_id",)),
    WarehouseObject("dim_referee", "canonical dimension", "app_allowed", "referee",
                    "Referee master.", ("referee_id",)),
    WarehouseObject("v_team_recent_form", "gold view", "app_allowed", "team recent form",
                    "Pre-aggregated last-5 form (W/D/L, goals).", ("team_id",)),
    WarehouseObject("v_head_to_head", "gold view", "app_allowed", "head to head summary",
                    "Pairwise h2h summary; team_a_id = LEAST, team_b_id = GREATEST.",
                    ("team_a_id", "team_b_id")),
    WarehouseObject("v_next_fixtures", "gold view", "app_allowed", "upcoming fixture list",
                    "Upcoming fixtures filtered for unplayed status.", ("fixture_id",)),
    WarehouseObject("v_match_card", "gold view", "app_allowed", "match card",
                    "Match composite with form + standings context.", ("fixture_id",)),
    WarehouseObject("v_prediction_features", "gold view", "app_allowed", "prediction feature set",
                    "Last-10 PPM and goal-diff per match features.", ("fixture_id",)),
    WarehouseObject("v_data_contract_tables", "gold view", "app_allowed", "table usage contract",
                    "Inventory of warehouse objects and their app_usage.", ("object_name",)),
)

APP_ALLOWED_OBJECTS = {obj.name for obj in WAREHOUSE_OBJECTS if obj.usage == "app_allowed"}


# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────


def _project() -> str:
    return os.environ.get("BIGQUERY_PROJECT_ID", "").strip()


def _dataset() -> str:
    return os.environ.get("BIGQUERY_DATASET_ID", _FALLBACK_DATASET).strip() or _FALLBACK_DATASET


def _fq(table: str) -> str:
    return f"`{_project()}.{_dataset()}.{table}`"


def _is_allowed_table(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _ALLOWED_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# Schema metadata loader (cached)
# ─────────────────────────────────────────────────────────────────────────────


_schema_cache: dict[str, Any] = {"loaded_at": 0.0, "data": {}}


def _load_column_metadata() -> dict[str, list[dict[str, str]]]:
    """{table_name: [{column_name, data_type, is_nullable, description}, ...]}.

    Restricted to app_allowed fact/dim/v objects. Cached for TTL seconds.
    """
    now = time.time()
    if _schema_cache["data"] and (now - _schema_cache["loaded_at"]) < _SCHEMA_CACHE_TTL_SECONDS:
        return _schema_cache["data"]

    project = _project()
    dataset = _dataset()
    if not project:
        return {}

    allowed = [name for name in sorted(APP_ALLOWED_OBJECTS) if _is_allowed_table(name)]
    quoted = ", ".join(f"'{name}'" for name in allowed)
    sql = f"""
    SELECT table_name, column_name, data_type, is_nullable, IFNULL(description, '') AS description
    FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name IN ({quoted})
    ORDER BY table_name, ordinal_position
    """
    try:
        df = run_query(sql)
    except Exception:
        return _schema_cache["data"]

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in df.itertuples(index=False):
        grouped.setdefault(str(row.table_name), []).append({
            "column_name": str(row.column_name),
            "data_type": str(row.data_type),
            "is_nullable": str(row.is_nullable),
            "description": str(getattr(row, "description", "") or "").strip(),
        })

    _schema_cache["data"] = grouped
    _schema_cache["loaded_at"] = now
    return grouped


def _format_schema(tables: list[str]) -> str:
    schema = _load_column_metadata()
    if not schema:
        return "(no live schema metadata available)"

    lines: list[str] = []
    for table in tables:
        cols = schema.get(table, [])
        if not cols:
            continue
        lines.append(f"- {table}:")
        for col in cols:
            desc = f" — {col['description']}" if col["description"] else ""
            null = "NULL" if col["is_nullable"].upper() == "YES" else "NOT NULL"
            lines.append(f"    • {col['column_name']} {col['data_type']} {null}{desc}")
    return "\n".join(lines) if lines else "(no schema rows)"


# ─────────────────────────────────────────────────────────────────────────────
# Business contract loader (with static fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _load_contract() -> list[dict[str, str]]:
    """Returns list of {object_name, object_layer, app_usage, business_concept}.

    Filtered to app_allowed fact/dim/v objects only.
    """
    project = _project()
    dataset = _dataset()
    if project:
        sql = f"""
        SELECT object_name, object_layer, app_usage, business_concept
        FROM `{project}.{dataset}.v_data_contract_tables`
        WHERE app_usage = 'app_allowed'
        """
        try:
            df = run_query(sql)
            rows = [
                {
                    "object_name": str(r.object_name),
                    "object_layer": str(r.object_layer),
                    "app_usage": str(r.app_usage),
                    "business_concept": str(r.business_concept),
                }
                for r in df.itertuples(index=False)
                if _is_allowed_table(str(r.object_name))
            ]
            if rows:
                return rows
        except Exception:
            pass

    # Fallback to static catalog
    return [
        {
            "object_name": obj.name,
            "object_layer": obj.kind,
            "app_usage": obj.usage,
            "business_concept": obj.business_concept,
        }
        for obj in WAREHOUSE_OBJECTS
        if obj.usage == "app_allowed" and _is_allowed_table(obj.name)
    ]


def _format_contract(contract: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- {row['object_name']} [{row['object_layer']}] — {row['business_concept']}"
        for row in contract
    )


# ─────────────────────────────────────────────────────────────────────────────
# Robust JSON parsing helper
# ─────────────────────────────────────────────────────────────────────────────


def _parse_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Empty LLM output")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    preview = text.replace("\n", " ")[:240]
    raise ValueError(f"Could not parse JSON: {preview}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Entity extraction
# ─────────────────────────────────────────────────────────────────────────────


def _extract_entities(query: str) -> dict[str, Any]:
    """Identify teams, season, and intent flags from the user question."""
    prompt = (
        "Extract structured entities from a football question for a World Cup 2026 assistant.\n"
        "Return JSON only with these keys:\n"
        "- teams: list of full team names mentioned (e.g., [\"Portugal\", \"Morocco\"])\n"
        "- season: integer year if mentioned, else null\n"
        "- is_head_to_head: true if the question compares two specific teams\n"
        "- is_specific_match: true if it asks about one specific fixture (last result, lineup, stats of a match)\n"
        "- needs_recent_form: true if it asks about form, streak, last matches\n"
        "- needs_upcoming: true if it asks about next/upcoming fixtures\n"
        "- needs_match_stats: true if it asks about possession, shots, fouls or other match statistics\n"
        "- needs_events: true if it asks about goals, cards, substitutions, scorers\n\n"
        f"Question: {query}\n\n"
        "JSON only."
    )
    try:
        parsed = _parse_json(_llm.invoke(prompt).content)
    except Exception:
        parsed = {}

    return {
        "teams": [str(t).strip() for t in parsed.get("teams", []) if str(t).strip()],
        "season": parsed.get("season"),
        "is_head_to_head": bool(parsed.get("is_head_to_head", False)),
        "is_specific_match": bool(parsed.get("is_specific_match", False)),
        "needs_recent_form": bool(parsed.get("needs_recent_form", False)),
        "needs_upcoming": bool(parsed.get("needs_upcoming", False)),
        "needs_match_stats": bool(parsed.get("needs_match_stats", False)),
        "needs_events": bool(parsed.get("needs_events", False)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Team resolution via dim_team
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_team_ids(team_names: list[str]) -> dict[str, int]:
    """Maps each input team name to its team_id via dim_team (fuzzy LIKE match)."""
    if not team_names or not _project():
        return {}

    safe_names = [name.replace("'", "''") for name in team_names]
    clauses = " OR ".join(
        f"LOWER(team_name) LIKE '%{name.lower()}%'" for name in safe_names
    )
    sql = f"""
    SELECT team_id, team_name
    FROM {_fq('dim_team')}
    WHERE {clauses}
    """
    try:
        df = run_query(sql)
    except Exception:
        return {}

    resolved: dict[str, int] = {}
    for original in team_names:
        needle = original.lower()
        match = df[df["team_name"].str.lower().str.contains(needle, na=False, regex=False)]
        if not match.empty:
            resolved[original] = int(match.iloc[0]["team_id"])
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Table selection
# ─────────────────────────────────────────────────────────────────────────────


def _heuristic_table_selection(entities: dict[str, Any]) -> list[str]:
    tables: list[str] = []
    if entities.get("needs_upcoming"):
        tables.append("v_next_fixtures")
    if entities.get("needs_recent_form"):
        tables.append("v_team_recent_form")
    if entities.get("is_head_to_head"):
        tables.append("v_head_to_head")
    if entities.get("is_specific_match"):
        tables.append("fact_fixture")
    if entities.get("needs_match_stats"):
        tables.append("fact_fixture_team_stat")
    if entities.get("needs_events"):
        tables.append("fact_fixture_event")
    if not tables:
        tables.append("fact_fixture")
    return tables


def _select_tables(query: str, entities: dict[str, Any], contract: list[dict[str, str]]) -> list[str]:
    """LLM-guided table selection, restricted to app_allowed fact/dim/v objects."""
    catalog_text = _format_contract(contract)
    valid_names = {row["object_name"] for row in contract}

    prompt = (
        "You are selecting BigQuery tables to answer a football question.\n"
        "Pick the smallest set of objects that together answer the question.\n"
        "Rules:\n"
        "- Choose only from the catalog below.\n"
        "- Prefer gold views (v_*) when they pre-aggregate the needed concept.\n"
        "- Include dim_team only if you need team names beyond what facts already denormalize.\n"
        "- Return JSON only: {\"tables\": [\"...\", \"...\"], \"reason\": \"...\"}\n\n"
        f"Catalog (app_allowed only):\n{catalog_text}\n\n"
        f"Question: {query}\n"
        f"Entities: {json.dumps(entities)}"
    )
    try:
        parsed = _parse_json(_llm.invoke(prompt).content)
        raw_tables = [str(t).strip() for t in parsed.get("tables", []) if str(t).strip()]
    except Exception:
        raw_tables = []

    selected = [t for t in raw_tables if t in valid_names and _is_allowed_table(t)]
    if not selected:
        selected = _heuristic_table_selection(entities)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: SQL generation
# ─────────────────────────────────────────────────────────────────────────────


def _generate_sql_plan(
    query: str,
    entities: dict[str, Any],
    resolved_teams: dict[str, int],
    tables: list[str],
) -> list[dict[str, str]]:
    """Returns list of {name, purpose, sql} dicts."""
    schema_text = _format_schema(tables)
    fq_examples = ", ".join(_fq(t) for t in tables)
    team_lookup = json.dumps(resolved_teams) if resolved_teams else "{}"

    prompt = (
        "You are a senior BigQuery analyst for a World Cup football assistant.\n"
        "Generate one or more SELECT queries that, together, fully answer the user question.\n"
        "Rules:\n"
        f"- Use ONLY these fully-qualified table references: {fq_examples}.\n"
        "- Every table reference must be in backticks with the form `project.dataset.table` as shown above.\n"
        "- Use the resolved team IDs when filtering by team. Do NOT use string LIKE on team_name when an ID is known.\n"
        "- Use only columns that appear in the schema below.\n"
        "- For head-to-head with v_head_to_head, remember team_a_id = LEAST(id1,id2), team_b_id = GREATEST(id1,id2).\n"
        "- For fact_fixture filtered by both teams, use (home_team_id=A AND away_team_id=B) OR (home_team_id=B AND away_team_id=A).\n"
        "- Prefer played matches (home_goals IS NOT NULL) when the user asks about results.\n"
        "- Add LIMIT to keep results compact (default LIMIT 25 for lists).\n"
        "- Each query must be a single read-only SELECT/WITH; no DDL/DML, no semicolons inside.\n"
        "- Return JSON only with shape: {\"queries\": [{\"name\":\"...\", \"purpose\":\"...\", \"sql\":\"...\"}]}\n"
        f"- Produce at most {_MAX_QUERIES} queries; usually 1-2 is best.\n\n"
        f"Question: {query}\n"
        f"Entities: {json.dumps(entities)}\n"
        f"Resolved team IDs (name -> team_id): {team_lookup}\n\n"
        f"Selected tables and their schemas:\n{schema_text}"
    )
    raw = _llm.invoke(prompt).content
    try:
        parsed = _parse_json(raw)
    except ValueError:
        repair = _llm.invoke(
            "Reformat the following into strict JSON with shape "
            "{\"queries\":[{\"name\":\"...\",\"purpose\":\"...\",\"sql\":\"...\"}]}. JSON only.\n\n"
            f"{raw}"
        ).content
        parsed = _parse_json(repair)

    queries = parsed.get("queries") or []
    cleaned: list[dict[str, str]] = []
    for i, q in enumerate(queries[:_MAX_QUERIES]):
        if not isinstance(q, dict):
            continue
        sql = str(q.get("sql", "")).strip()
        if not sql:
            continue
        cleaned.append({
            "name": str(q.get("name") or f"query_{i+1}"),
            "purpose": str(q.get("purpose") or ""),
            "sql": sql,
        })
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: SQL validation, normalization, and execution with one repair retry
# ─────────────────────────────────────────────────────────────────────────────


_FORBIDDEN_TOKENS = (
    " insert ", " update ", " delete ", " drop ", " create ",
    " alter ", " truncate ", " merge ", " grant ", " revoke ",
)


def _normalize_sql(sql: str) -> str:
    project, dataset = _project(), _dataset()
    if not project:
        return sql
    prefix = f"{project}.{dataset}"
    out = sql
    out = re.sub(r"`project[\.:]dataset\.([A-Za-z_][A-Za-z0-9_]*)`",
                 rf"`{prefix}.\1`", out, flags=re.IGNORECASE)
    out = re.sub(r"\bproject[\.:]dataset\.([A-Za-z_][A-Za-z0-9_]*)\b",
                 rf"`{prefix}.\1`", out, flags=re.IGNORECASE)
    out = out.replace("<project>", project)
    return out


def _validate_sql(sql: str) -> str:
    cleaned = _normalize_sql(str(sql or "")).strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("Empty SQL")
    if ";" in cleaned:
        raise ValueError("Multiple statements not allowed")
    if not re.match(r"(?is)^(with|select)\b", cleaned):
        raise ValueError("Only SELECT/WITH allowed")
    lowered = f" {cleaned.lower()} "
    if any(tok in lowered for tok in _FORBIDDEN_TOKENS):
        raise ValueError("Only read-only SQL allowed")
    for table in re.findall(r"`[^`]+\.([A-Za-z0-9_]+)`", cleaned):
        if not _is_allowed_table(table):
            raise ValueError(f"Table '{table}' is not in the app_allowed catalog")
    return cleaned


def _repair_sql(
    query: str,
    failed_sql: str,
    error_text: str,
    tables: list[str],
) -> str:
    schema_text = _format_schema(tables)
    fq_examples = ", ".join(_fq(t) for t in tables)
    repair_prompt = (
        "Fix a failed BigQuery SELECT query using the schema and error below.\n"
        "Rules:\n"
        f"- Use ONLY these fully-qualified tables: {fq_examples}.\n"
        "- Use only columns in the schema (correct any wrong column names).\n"
        "- Keep one read-only SELECT/WITH.\n"
        "- Preserve the original analytical intent.\n"
        "- Return JSON only: {\"sql\": \"...\"}\n\n"
        f"User question: {query}\n"
        f"Failed SQL: {failed_sql}\n"
        f"BigQuery error: {error_text}\n\n"
        f"Schemas:\n{schema_text}"
    )
    parsed = _parse_json(_llm.invoke(repair_prompt).content)
    return _validate_sql(str(parsed.get("sql", "")))


def _execute_query(
    user_query: str,
    sql: str,
    tables: list[str],
) -> tuple[pd.DataFrame, str, str | None]:
    """Returns (dataframe, sql_used, repair_note_or_None)."""
    try:
        validated = _validate_sql(sql)
        df = run_query(validated)
        return df, validated, None
    except Exception as first_exc:
        try:
            repaired_sql = _repair_sql(user_query, sql, str(first_exc), tables)
            df = run_query(repaired_sql)
            return df, repaired_sql, f"Auto-repaired after error: {first_exc}"
        except Exception as second_exc:
            raise RuntimeError(
                f"SQL failed twice. Initial error: {first_exc}. Repair error: {second_exc}"
            ) from second_exc


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Composition
# ─────────────────────────────────────────────────────────────────────────────


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "_No rows returned._"
    preview = df.head(max_rows).copy()
    try:
        return preview.to_markdown(index=False)
    except Exception:
        headers = list(preview.columns)
        lines = [" | ".join(map(str, headers)), " | ".join(["---"] * len(headers))]
        for row in preview.itertuples(index=False):
            lines.append(" | ".join("" if v is None else str(v).replace("\n", " ") for v in row))
        return "\n".join(lines)


def _compose_answer(
    user_query: str,
    executions: list[dict[str, Any]],
) -> str:
    sections: list[str] = []
    for item in executions:
        name = item["name"]
        purpose = item["purpose"]
        df: pd.DataFrame = item["df"]
        sections.append(
            f"### {name} — {purpose}\n"
            f"Rows: {len(df)}\n\n"
            f"{_df_to_markdown(df)}"
        )
    data_block = "\n\n".join(sections) if sections else "_No data retrieved._"

    prompt = (
        "You are answering a football question using only the BigQuery results below.\n"
        "Rules:\n"
        "- Ground every factual statement in the provided rows; do not invent numbers.\n"
        "- Use concise markdown with short sections and bullet points.\n"
        "- If a result is empty, say so explicitly for that aspect.\n"
        "- Do not include the SQL text; summarize what the data shows.\n\n"
        f"User question: {user_query}\n\n"
        f"Query results:\n{data_block}"
    )
    return _llm.invoke(prompt).content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Confidence
# ─────────────────────────────────────────────────────────────────────────────


def _confidence(executions: list[dict[str, Any]]) -> tuple[float, str]:
    if not executions:
        return 0.2, "No queries executed."

    total_rows = sum(len(item["df"]) for item in executions)
    had_repair = any(item.get("repair_note") for item in executions)

    if total_rows == 0:
        return 0.35, "Queries ran but returned no rows."
    if had_repair:
        return 0.7, "Queries succeeded after one SQL auto-repair."
    if total_rows >= 5:
        return 0.85, "Multiple matching rows from canonical warehouse objects."
    return 0.8, "Focused result set from canonical warehouse objects."


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_structured(query: str) -> dict[str, Any]:
    debug: dict[str, Any] = {}
    try:
        entities = _extract_entities(query)
        debug["entities"] = entities

        resolved_teams = _resolve_team_ids(entities["teams"])
        debug["resolved_teams"] = resolved_teams

        contract = _load_contract()
        tables = _select_tables(query, entities, contract)
        debug["selected_tables"] = tables

        plan = _generate_sql_plan(query, entities, resolved_teams, tables)
        if not plan:
            raise ValueError("Planner produced no queries")
        debug["plan"] = [{"name": p["name"], "purpose": p["purpose"]} for p in plan]

        executions: list[dict[str, Any]] = []
        for step in plan:
            df, used_sql, repair_note = _execute_query(query, step["sql"], tables)
            executions.append({
                "name": step["name"],
                "purpose": step["purpose"],
                "sql": used_sql,
                "df": df,
                "repair_note": repair_note,
            })

        answer = _compose_answer(query, executions)
        confidence_score, confidence_reason = _confidence(executions)

        return {
            "answer": answer,
            "confidence_score": confidence_score,
            "confidence_reason": confidence_reason,
            "metadata": {
                "data_source": "bigquery",
                "entities": entities,
                "resolved_teams": resolved_teams,
                "tables_used": tables,
                "queries": [
                    {
                        "name": item["name"],
                        "purpose": item["purpose"],
                        "sql": item["sql"],
                        "row_count": int(len(item["df"])),
                        "repair_note": item["repair_note"],
                    }
                    for item in executions
                ],
            },
        }
    except Exception as exc:
        return {
            "answer": (
                f"I could not retrieve a warehouse answer for your question.\n"
                f"Reason: {exc}"
            ),
            "confidence_score": 0.2,
            "confidence_reason": f"BigQuery pipeline failed: {exc}",
            "metadata": {
                "data_source": "bigquery",
                "error": str(exc),
                "debug": debug,
            },
        }


def run(query: str) -> str:
    return run_structured(query)["answer"]


if __name__ == "__main__":
    print(run("What was Portugal vs Morocco last result and match stats?"))
