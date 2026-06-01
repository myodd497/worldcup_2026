"""BigQuery Agent — plans and runs SQL over the warehouse using schema-aware metadata.

The agent prefers canonical `fact_` / `dim_` tables and `v_` gold views, while
keeping source-only tables available for explicit warehouse-introspection tasks.
It returns structured output that is later polished by the shared final composer.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from langchain_openai import ChatOpenAI

from src.tools.bigquery_tools import run_query


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


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
    WarehouseObject(
        name="fixtures_historical",
        kind="source table",
        usage="source_only",
        business_concept="raw fixture snapshot",
        description="Historical fixture rows ingested from API-Football.",
        key_columns=("fixture_id", "season", "date", "home_team", "away_team"),
        notes="Source-only; app features should not query this directly.",
    ),
    WarehouseObject(
        name="team_match_history",
        kind="source table",
        usage="source_only",
        business_concept="raw team-match history",
        description="Team-centric historical matches across competitions.",
        key_columns=("team_id", "fixture_id", "match_date", "result"),
        notes="Source-only; canonical team fixture facts are built from it.",
    ),
    WarehouseObject(
        name="fixture_events",
        kind="source table",
        usage="source_only",
        business_concept="raw fixture events",
        description="Goals, cards, substitutions, and VAR events.",
        key_columns=("fixture_id", "team_id", "event_type", "event_detail"),
        notes="Source-only; use canonical event facts for app queries.",
    ),
    WarehouseObject(
        name="fixture_stats",
        kind="source table",
        usage="source_only",
        business_concept="raw fixture team stats",
        description="Long-format team statistics for each fixture.",
        key_columns=("fixture_id", "team_id", "stat_type"),
        notes="Source-only; use fact_fixture_team_stat for app queries.",
    ),
    WarehouseObject(
        name="standings",
        kind="source table",
        usage="source_only",
        business_concept="raw standings",
        description="Group standings snapshots.",
        key_columns=("season", "team_id", "group_name"),
        notes="Source-only; use canonical tables/views in app logic.",
    ),
    WarehouseObject(
        name="team_stats",
        kind="source table",
        usage="source_only",
        business_concept="raw team season stats",
        description="Season-level team aggregates from API-Football.",
        key_columns=("team_id", "season"),
        notes="Source-only; use canonical tables/views in app logic.",
    ),
    WarehouseObject(
        name="fact_fixture",
        kind="canonical table",
        usage="app_allowed",
        business_concept="fixture",
        description="One row per fixture with competition, venue, referee, and score columns.",
        key_columns=("fixture_id",),
    ),
    WarehouseObject(
        name="fact_team_fixture",
        kind="canonical table",
        usage="app_allowed",
        business_concept="team fixture participation",
        description="One row per team and fixture pair for team-centric history questions.",
        key_columns=("team_id", "fixture_id"),
    ),
    WarehouseObject(
        name="fact_fixture_event",
        kind="canonical table",
        usage="app_allowed",
        business_concept="fixture event",
        description="Canonical fixture-event timeline.",
        key_columns=("event_key",),
    ),
    WarehouseObject(
        name="fact_fixture_team_stat",
        kind="canonical table",
        usage="app_allowed",
        business_concept="fixture team statistic",
        description="One row per fixture, team, and stat type.",
        key_columns=("fixture_id", "team_id", "stat_type"),
    ),
    WarehouseObject(
        name="dim_team",
        kind="canonical dimension",
        usage="app_allowed",
        business_concept="team",
        description="Master team dimension.",
        key_columns=("team_id",),
    ),
    WarehouseObject(
        name="dim_competition",
        kind="canonical dimension",
        usage="app_allowed",
        business_concept="competition",
        description="Master competition dimension.",
        key_columns=("competition_id",),
    ),
    WarehouseObject(
        name="dim_venue",
        kind="canonical dimension",
        usage="app_allowed",
        business_concept="venue",
        description="Master venue dimension.",
        key_columns=("venue_id",),
    ),
    WarehouseObject(
        name="dim_referee",
        kind="canonical dimension",
        usage="app_allowed",
        business_concept="referee",
        description="Master referee dimension.",
        key_columns=("referee_id",),
    ),
    WarehouseObject(
        name="v_team_recent_form",
        kind="gold view",
        usage="app_allowed",
        business_concept="team recent form",
        description="Recent form summary by team.",
        key_columns=("team_id",),
    ),
    WarehouseObject(
        name="v_head_to_head",
        kind="gold view",
        usage="app_allowed",
        business_concept="head to head summary",
        description="Head-to-head aggregates by team pair.",
        key_columns=("team_a_id", "team_b_id"),
    ),
    WarehouseObject(
        name="v_next_fixtures",
        kind="gold view",
        usage="app_allowed",
        business_concept="upcoming fixture list",
        description="Upcoming fixtures ordered by kickoff time.",
        key_columns=("fixture_id",),
    ),
    WarehouseObject(
        name="v_match_card",
        kind="gold view",
        usage="app_allowed",
        business_concept="match card",
        description="Composite match card with form and standings context.",
        key_columns=("fixture_id",),
    ),
    WarehouseObject(
        name="v_prediction_features",
        kind="gold view",
        usage="app_allowed",
        business_concept="prediction feature set",
        description="Feature view for prediction use cases.",
        key_columns=("fixture_id",),
    ),
    WarehouseObject(
        name="v_dq_uniqueness_checks",
        kind="gold view",
        usage="app_allowed",
        business_concept="uniqueness quality checks",
        description="Data quality summary for uniqueness constraints.",
        key_columns=("table_name",),
    ),
    WarehouseObject(
        name="v_data_contract_tables",
        kind="gold view",
        usage="app_allowed",
        business_concept="table usage contract",
        description="Inventory of source-only, canonical, and gold objects.",
        key_columns=("object_name",),
    ),
    WarehouseObject(
        name="etl_run_status",
        kind="ops table",
        usage="app_allowed",
        business_concept="ETL run status",
        description="Operational log of ETL runs, status, duration, and errors.",
        key_columns=("run_id", "started_at"),
    ),
)

APP_ALLOWED_OBJECTS = {
    obj.name for obj in WAREHOUSE_OBJECTS if obj.usage == "app_allowed"
}

ANALYTICAL_KEYWORDS = {
    "count",
    "how many",
    "average",
    "avg",
    "sum",
    "top",
    "rank",
    "compare",
    "comparison",
    "trend",
    "list",
    "show",
    "table",
    "tables",
    "schema",
    "column",
    "columns",
    "rows",
    "most",
    "least",
    "unique",
    "distinct",
    "next fixtures",
    "recent form",
    "head to head",
    "prediction features",
}


def _catalog_summary() -> str:
    lines: list[str] = []
    for obj in WAREHOUSE_OBJECTS:
        key_cols = ", ".join(obj.key_columns) if obj.key_columns else "n/a"
        lines.append(
            f"- {obj.name} | {obj.kind} | {obj.usage} | {obj.business_concept} | keys: {key_cols}"
        )
        lines.append(f"  - {obj.description}")
        if obj.notes:
            lines.append(f"  - {obj.notes}")
    return "\n".join(lines)


def _runtime_column_metadata() -> str:
    dataset = os.environ.get("BIGQUERY_DATASET_ID", "").strip()
    project = os.environ.get("BIGQUERY_PROJECT_ID", "").strip()
    if not dataset or not project:
        return ""

    object_names = sorted(APP_ALLOWED_OBJECTS)
    quoted = ", ".join([f"'{name}'" for name in object_names])
    sql = f"""
    SELECT table_name, column_name, data_type, is_nullable, IFNULL(description, '') AS description
    FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name IN ({quoted})
    ORDER BY table_name, ordinal_position
    """
    try:
        df = run_query(sql)
    except Exception:
        return ""

    if df.empty:
        return ""

    grouped: dict[str, list[str]] = {}
    for row in df.itertuples(index=False):
        desc = str(getattr(row, "description", "") or "").strip()
        nullability = "nullable" if str(getattr(row, "is_nullable", "YES")).upper() == "YES" else "required"
        col_text = f"{row.column_name}:{row.data_type} ({nullability})"
        if desc:
            col_text += f" - {desc}"
        grouped.setdefault(str(row.table_name), []).append(col_text)

    lines: list[str] = ["Live column metadata from BigQuery:"]
    for table_name in sorted(grouped):
        lines.append(f"- {table_name}: {', '.join(grouped[table_name])}")
    return "\n".join(lines)


def _keyword_seed(query: str) -> str:
    q = query.lower()
    if any(term in q for term in ("what tables", "available tables", "table list", "schema")):
        return "catalog"
    if any(term in q for term in ("next game", "upcoming", "next match", "today", "fixture")):
        return "fixtures"
    if any(term in q for term in ("head to head", "vs", "versus")):
        return "h2h"
    if any(term in q for term in ("recent form", "form")):
        return "form"
    if any(term in q for term in ("prediction", "probability", "win", "draw", "loss")):
        return "prediction"
    if any(term in q for term in ("venue", "referee")):
        return "match_card"
    if any(term in q for term in ANALYTICAL_KEYWORDS):
        return "analytics"
    return "generic"


def _heuristic_sql(query: str) -> str:
    seed = _keyword_seed(query)
    dataset = os.environ["BIGQUERY_DATASET_ID"]
    project = os.environ["BIGQUERY_PROJECT_ID"]

    if seed == "catalog":
        return f"SELECT * FROM `{project}.{dataset}.v_data_contract_tables` ORDER BY object_layer, object_name"
    if seed == "fixtures":
        return f"SELECT * FROM `{project}.{dataset}.v_next_fixtures` ORDER BY fixture_datetime ASC LIMIT 10"
    if seed == "h2h":
        return f"SELECT * FROM `{project}.{dataset}.v_head_to_head` ORDER BY last_meeting_datetime DESC LIMIT 10"
    if seed == "form":
        return f"SELECT * FROM `{project}.{dataset}.v_team_recent_form` ORDER BY matches_used DESC LIMIT 20"
    if seed == "prediction":
        return f"SELECT * FROM `{project}.{dataset}.v_prediction_features` ORDER BY fixture_datetime ASC LIMIT 10"
    if seed == "match_card":
        return f"SELECT * FROM `{project}.{dataset}.v_match_card` ORDER BY fixture_datetime ASC LIMIT 10"

    return f"""
    SELECT *
    FROM `{project}.{dataset}.fact_fixture`
    ORDER BY fixture_datetime DESC
    LIMIT 10
    """.strip()


def _plan_sql(query: str) -> dict[str, Any]:
    catalog = _catalog_summary()
    live_metadata = _runtime_column_metadata()

    prompt = (
        "You are a BigQuery analyst for a World Cup football assistant.\n"
        "Translate the user request into a single safe BigQuery SELECT query.\n"
        "Rules:\n"
        "- Use only the warehouse objects listed below.\n"
        "- Prefer app_allowed canonical tables and gold views.\n"
        "- Use source-only tables only if the user explicitly asks about raw ingestion or warehouse inventory.\n"
        "- Return JSON only with keys: sql, tables_used, explanation, answer_style.\n"
        "- Use one query statement only. No DDL/DML. No comments.\n"
        "- If the user asks for available tables or schema, query v_data_contract_tables or INFORMATION_SCHEMA.\n"
        "- If the user asks about fixtures, use fact_fixture or v_next_fixtures.\n"
        "- If the user asks for form, head-to-head, or prediction inputs, prefer the matching gold view.\n"
        "- If a query needs team/competition/venue/referee metadata, join the relevant dim tables.\n\n"
        "- If the question is temporal or relative to today, use CURRENT_DATE('UTC') and date arithmetic instead of a hardcoded countdown branch.\n"
        f"Warehouse catalog:\n{catalog}\n\n"
        f"{live_metadata}\n\n"
        f"User request: {query}"
    )
    raw = _llm.invoke(prompt).content.strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Planner output was not a JSON object")
    return parsed


def _validate_sql(sql: str) -> str:
    cleaned = str(sql or "").strip()
    if not cleaned:
        raise ValueError("Empty SQL generated")

    cleaned = cleaned.rstrip(";").strip()
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed")

    if not re.match(r"(?is)^(with|select)\b", cleaned):
        raise ValueError("Only SELECT/WITH queries are allowed")

    forbidden = (
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " create ",
        " alter ",
        " truncate ",
        " merge ",
        " grant ",
        " revoke ",
    )
    lowered = f" {cleaned.lower()} "
    if any(token in lowered for token in forbidden):
        raise ValueError("Only read-only SQL is allowed")

    return cleaned


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("\n", " ").strip()


def _format_dataframe(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows returned."

    preview = df.head(10).copy()
    try:
        return preview.to_markdown(index=False)
    except Exception:
        headers = list(preview.columns)
        lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
        for row in preview.itertuples(index=False):
            lines.append(" | ".join(_format_value(cell) for cell in row))
        return "\n".join(lines)


def _fallback_answer(query: str, reason: str) -> dict[str, Any]:
    sql = _heuristic_sql(query)
    try:
        df = run_query(sql)
    except Exception as exc:
        return {
            "answer": f"I could not run a warehouse query for your request. {reason} Error: {exc}",
            "confidence_score": 0.2,
            "confidence_reason": f"{reason} Heuristic fallback also failed.",
            "metadata": {
                "data_source": "bigquery",
                "sql": sql,
                "tables_used": [],
                "row_count": 0,
                "execution_mode": "fallback_failed",
            },
        }

    tables_used = _extract_tables_from_sql(sql)
    return {
        "answer": _render_answer(query=query, sql=sql, df=df, tables_used=tables_used, answer_style="fallback"),
        "confidence_score": 0.55,
        "confidence_reason": f"{reason} Used a heuristic warehouse query fallback.",
        "metadata": {
            "data_source": "bigquery",
            "sql": sql,
            "tables_used": tables_used,
            "row_count": int(len(df)),
            "execution_mode": "fallback",
        },
    }


def _extract_tables_from_sql(sql: str) -> list[str]:
    matches = re.findall(r"`[^`]+\.([A-Za-z0-9_]+)`", sql)
    seen: list[str] = []
    for table in matches:
        if table not in seen:
            seen.append(table)
    return seen


def _render_countdown_answer(df: pd.DataFrame) -> str:
    if df.empty:
        return "I could not calculate the countdown."

    row = df.iloc[0]
    days_left = row.get("days_until_start")
    start_date = row.get("world_cup_start_date")
    today_utc = row.get("today_utc")
    return (
        f"The FIFA Men's World Cup 2026 starts on {start_date}.\n"
        f"Today (UTC): {today_utc}\n"
        f"Days left: {int(days_left)}"
    )


def _render_answer(
    *,
    query: str,
    sql: str,
    df: pd.DataFrame,
    tables_used: list[str],
    answer_style: str,
) -> str:
    table_label = ", ".join(tables_used) if tables_used else "warehouse"
    rows = len(df)
    preview = _format_dataframe(df)

    lines = [
        f"I queried BigQuery using {table_label}.",
        f"SQL used: `{sql}`",
        f"Rows returned: {rows}",
        "",
    ]

    if answer_style == "catalog":
        lines.append("Available warehouse objects:")
    elif answer_style in {"analytics", "table"}:
        lines.append("Result summary:")
    else:
        lines.append("Result preview:")

    lines.append(preview)
    return "\n".join(lines)


def _confidence_from_result(df: pd.DataFrame, tables_used: list[str]) -> tuple[float, str]:
    if df.empty:
        return 0.3, "BigQuery query ran successfully but returned no rows."
    if any(table in {"v_data_contract_tables", "INFORMATION_SCHEMA"} for table in tables_used):
        return 0.85, "Direct warehouse metadata lookup succeeded."
    if len(df) <= 5:
        return 0.8, "Small, focused result set from canonical warehouse objects."
    return 0.75, "Warehouse query succeeded with a broader result set."


def run_structured(query: str) -> dict[str, Any]:
    try:
        spec = _plan_sql(query)
        sql = _validate_sql(str(spec.get("sql", "")))
        tables_used = list(spec.get("tables_used", []) or _extract_tables_from_sql(sql))
        answer_style = str(spec.get("answer_style", "analytics")).strip().lower() or "analytics"
        df = run_query(sql)

        answer = _render_answer(
            query=query,
            sql=sql,
            df=df,
            tables_used=tables_used,
            answer_style=answer_style,
        )
        confidence_score, confidence_reason = _confidence_from_result(df, tables_used)
        return {
            "answer": answer,
            "confidence_score": confidence_score,
            "confidence_reason": confidence_reason,
            "metadata": {
                "data_source": "bigquery",
                "sql": sql,
                "tables_used": tables_used,
                "row_count": int(len(df)),
                "answer_style": answer_style,
            },
        }
    except Exception as exc:
        return _fallback_answer(query=query, reason=f"BigQuery SQL planning failed: {exc}")


def run(query: str) -> str:
    return run_structured(query)["answer"]


if __name__ == "__main__":
    print(run("what tables do we have in BigQuery?"))
