"""Datamodel tools — read-only BigQuery primitives exposed to the agent.

Four tools, designed to be LLM-callable:
  - list_tables(filter)        → names + 1-line descriptions
  - describe_table(name)       → full schema + usage notes for one table
  - sample_table(name, limit)  → first N rows for shape inspection
  - run_sql(sql)               → execute SELECT/WITH against allowed tables

Guardrails:
  - SELECT/WITH only (no DDL/DML)
  - Single statement (no semicolons)
  - Every backticked table reference must resolve to an agent-visible table
  - LIMIT auto-applied when missing (configurable)
"""
from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from src.data.datamodel.catalog import (
    agent_visible_table_names,
    format_catalog_for_llm,
    format_table_detail_for_llm,
    fqn,
    get_table,
    list_tables,
)
from src.tools.bigquery_tools import _client, run_query

logger = logging.getLogger(__name__)


_FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "create",
    "alter", "truncate", "merge", "grant", "revoke", "call",
)

_DEFAULT_SAMPLE_LIMIT = 5
_DEFAULT_MAX_ROWS = 500
_MAX_BYTES_BILLED = 1_000_000_000  # 1 GB safety cap


# ─────────────────────────────────────────────────────────────────────────────
# Tool: list_tables
# ─────────────────────────────────────────────────────────────────────────────


def list_tables_tool(layer: str | None = None) -> str:
    """Return a markdown summary of available tables. Optional layer filter: 'mart'|'fact'|'dim'."""
    if layer:
        layer = layer.lower().strip()
        if layer not in {"mart", "fact", "dim"}:
            return f"Invalid layer '{layer}'. Use one of: mart, fact, dim, or omit."
        tables = list_tables(agent_visible=True, layer=layer)
        if not tables:
            return f"No agent-visible tables in layer '{layer}'."
        return "\n".join(f"- `{t.name}` — {t.description}" for t in tables)
    return format_catalog_for_llm()


# ─────────────────────────────────────────────────────────────────────────────
# Tool: describe_table
# ─────────────────────────────────────────────────────────────────────────────


def describe_table_tool(name: str) -> str:
    """Return full schema (columns, types, descriptions) and usage hint for a table."""
    try:
        return format_table_detail_for_llm(name)
    except KeyError as exc:
        return f"Error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool: sample_table
# ─────────────────────────────────────────────────────────────────────────────


def sample_table_tool(name: str, limit: int = _DEFAULT_SAMPLE_LIMIT) -> str:
    """Return first N rows of a table as a markdown table (max 20 rows)."""
    try:
        spec = get_table(name)
    except KeyError as exc:
        return f"Error: {exc}"
    if not spec.agent_visible:
        return f"Error: table '{name}' is not agent-visible."

    limit = max(1, min(int(limit or _DEFAULT_SAMPLE_LIMIT), 20))
    sql = f"SELECT * FROM {fqn(name)} LIMIT {limit}"
    try:
        df = run_query(sql)
    except Exception as exc:
        return f"Error executing sample query: {exc}"
    return _df_to_markdown(df, max_rows=limit)


# ─────────────────────────────────────────────────────────────────────────────
# Tool: run_sql
# ─────────────────────────────────────────────────────────────────────────────


def run_sql_tool(sql: str, max_rows: int = _DEFAULT_MAX_ROWS) -> dict[str, Any]:
    """Execute a read-only SQL query. Returns dict with rows (list[dict]), row_count, error.

    Pre-flight: validate syntax/whitelist, then BigQuery dry-run to catch column
    errors AND estimate bytes-scanned. If bytes_billed_estimate exceeds the cap,
    refuse the query rather than run it.
    """
    try:
        validated = validate_sql(sql)
    except ValueError as exc:
        return {"error": str(exc), "rows": [], "row_count": 0, "sql": sql, "bytes_billed_estimate": 0}

    # Dry-run to surface column/typing errors cheaply and to enforce a cost cap.
    try:
        bytes_estimate = dry_run_sql(validated)
    except Exception as exc:
        return {
            "error": f"dry-run failed: {exc}",
            "rows": [], "row_count": 0,
            "sql": validated,
            "bytes_billed_estimate": 0,
        }
    if bytes_estimate > _MAX_BYTES_BILLED:
        return {
            "error": (
                f"Query would scan {bytes_estimate / 1e9:.2f} GB which exceeds the "
                f"{_MAX_BYTES_BILLED / 1e9:.2f} GB cap. Add a more selective WHERE "
                f"clause (e.g. team_id, competition_id, date range) and try again."
            ),
            "rows": [], "row_count": 0,
            "sql": validated,
            "bytes_billed_estimate": bytes_estimate,
        }

    try:
        df = run_query(validated)
    except Exception as exc:
        return {
            "error": str(exc),
            "rows": [], "row_count": 0,
            "sql": validated,
            "bytes_billed_estimate": bytes_estimate,
        }

    if len(df) > max_rows:
        df = df.head(max_rows)

    return {
        "error": None,
        "sql": validated,
        "row_count": int(len(df)),
        "rows": df.to_dict(orient="records"),
        "columns": list(df.columns),
        "bytes_billed_estimate": bytes_estimate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool: data_freshness
# ─────────────────────────────────────────────────────────────────────────────

_FRESHNESS_CACHE: dict[str, Any] = {"ts": 0.0, "payload": None}
_FRESHNESS_TTL_SECONDS = 300  # 5 min — avoid hammering BQ from every chat turn


def data_freshness_tool() -> str:
    """Return JSON with hours since the last successful ETL run + stale flag."""
    import json as _json
    import time
    from datetime import datetime, timezone

    now = time.time()
    cached = _FRESHNESS_CACHE.get("payload")
    if cached and (now - float(_FRESHNESS_CACHE.get("ts", 0))) < _FRESHNESS_TTL_SECONDS:
        return cached

    import os as _os
    project = _os.environ.get("BIGQUERY_PROJECT_ID")
    dataset = _os.environ.get("BIGQUERY_DATASET_ID")
    if not project or not dataset:
        return _json.dumps({"error": "BIGQUERY_PROJECT_ID/DATASET_ID not set", "is_stale": True})

    sql = (
        f"SELECT MAX(finished_at) AS last_success_ts "
        f"FROM `{project}.{dataset}.etl_run_status` "
        f"WHERE status = 'SUCCESS'"
    )
    try:
        df = run_query(sql)
        last_ts = df["last_success_ts"].iloc[0] if not df.empty else None
        if last_ts is None or pd.isna(last_ts):
            payload = _json.dumps({"last_success_ts": None, "hours_since_last_success": None, "is_stale": True})
        else:
            ts = last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
            payload = _json.dumps({
                "last_success_ts": ts.isoformat(),
                "hours_since_last_success": round(hours, 1),
                "is_stale": hours > 6.0,
            })
    except Exception as exc:
        payload = _json.dumps({"error": str(exc), "is_stale": True})

    _FRESHNESS_CACHE["ts"] = now
    _FRESHNESS_CACHE["payload"] = payload
    return payload


def dry_run_sql(sql: str) -> int:
    """BigQuery dry-run. Returns total_bytes_processed; raises on syntax/column errors."""
    from google.cloud import bigquery
    project = __import__("os").environ["BIGQUERY_PROJECT_ID"]
    dataset = __import__("os").environ.get("BIGQUERY_DATASET_ID", "worldcup2026")
    client = _client()
    job_config = bigquery.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
        default_dataset=f"{project}.{dataset}",
    )
    job = client.query(sql, job_config=job_config)
    return int(job.total_bytes_processed or 0)


# ─────────────────────────────────────────────────────────────────────────────
# SQL validation
# ─────────────────────────────────────────────────────────────────────────────


_TABLE_REF_RE = re.compile(r"`[^`]+\.([A-Za-z0-9_]+)`")
_BARE_TABLE_REF_RE = re.compile(
    r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)\b|\bJOIN\s+([A-Za-z_][A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> str:
    """Validate + normalize SQL. Returns cleaned SQL or raises ValueError."""
    if not sql or not isinstance(sql, str):
        raise ValueError("SQL is empty")
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("SQL is empty after trimming")
    if ";" in cleaned:
        raise ValueError("Multiple statements are not allowed")
    if not re.match(r"(?is)^(with|select)\b", cleaned):
        raise ValueError("Only SELECT/WITH queries are allowed")

    # Forbidden keywords (word-boundary match to avoid false positives like 'created_at')
    lowered = cleaned.lower()
    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", lowered):
            raise ValueError(f"Forbidden keyword '{kw}' in SQL")

    # Auto-qualify bare table references (FROM table → FROM `project.dataset.table`)
    allowed = agent_visible_table_names()
    cleaned = _auto_qualify(cleaned, allowed)

    # Every backticked table reference must be agent-visible
    found = set(_TABLE_REF_RE.findall(cleaned))
    if not found:
        raise ValueError(
            "No qualified table references found. "
            "Use the form `project.dataset.<table>` or just the table name."
        )
    illegal = found - allowed
    if illegal:
        raise ValueError(
            f"Table(s) not agent-visible: {sorted(illegal)}. "
            f"Allowed: {sorted(allowed)}"
        )

    return cleaned


def _auto_qualify(sql: str, allowed: set[str]) -> str:
    """Rewrite bare `FROM tablename` / `JOIN tablename` to fully-qualified backticked form."""
    def _replace(m: re.Match) -> str:
        full = m.group(0)
        name = m.group(1) or m.group(2)
        if name in allowed:
            kw = "FROM" if m.group(1) else "JOIN"
            return f"{kw} {fqn(name)}"
        return full
    return _BARE_TABLE_REF_RE.sub(_replace, sql)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "_(no rows)_"
    preview = df.head(max_rows)
    try:
        return preview.to_markdown(index=False)
    except Exception:
        headers = list(preview.columns)
        lines = [" | ".join(map(str, headers)), " | ".join(["---"] * len(headers))]
        for row in preview.itertuples(index=False):
            lines.append(" | ".join("" if v is None else str(v).replace("\n", " ") for v in row))
        return "\n".join(lines)
