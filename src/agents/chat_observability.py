"""BigQuery chat observability logger.

Writes one row per interaction with user question, final answer, execution metrics,
and SQL/tool telemetry extracted from orchestrator state.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from src.tools.bigquery_tools import _client

logger = logging.getLogger(__name__)

_TABLE_CACHE: set[str] = set()


def _enabled() -> bool:
    return os.getenv("CHAT_BQ_LOGGING_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _table_name() -> str:
    return os.getenv("CHAT_BQ_LOG_TABLE", "chat_observability")


def _table_fqn() -> str:
    return (
        f"{os.environ['BIGQUERY_PROJECT_ID']}."
        f"{os.environ['BIGQUERY_DATASET_ID']}.{_table_name()}"
    )


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=True)
    except Exception:
        return "{}"


def _resolve_model_names() -> list[str]:
    provider = (os.getenv("LLM_PROVIDER") or "deepseek").strip().lower()
    if provider == "openai":
        return [
            os.getenv("OPENAI_SIMPLE_MODEL", "gpt-4o-mini"),
            os.getenv("OPENAI_COMPLEX_MODEL", "gpt-4o"),
        ]
    return [
        os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
        os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro"),
        os.getenv(
            "DEEPSEEK_TOOL_MODEL",
            os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
        ),
    ]


def _ensure_table() -> None:
    fqn = _table_fqn()
    if fqn in _TABLE_CACHE:
        return

    client = _client()
    try:
        client.get_table(fqn)
        _TABLE_CACHE.add(fqn)
        return
    except NotFound:
        pass

    schema = [
        bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("timestamp_utc", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("request_ts_utc", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("response_ts_utc", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("latency_ms", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("user_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("conversation_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("question_text", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("final_answer_text", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("selected_agents", "STRING", mode="REPEATED"),
        bigquery.SchemaField("primary_agent", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("agent_path", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("models_used", "STRING", mode="REPEATED"),
        bigquery.SchemaField("token_input", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("token_output", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("token_total", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("estimated_cost_usd", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("error_flag", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("fallback_flag", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("cache_hit", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("tool_runs", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("sql_runs", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("sql_bytes_read", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("sql_gb_read", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("sql_queries", "STRING", mode="REPEATED"),
        bigquery.SchemaField("sql_query_metrics_json", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("tool_runs_by_agent_json", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("workflow_steps_json", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("verifier_verdict_json", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("error_message", "STRING", mode="NULLABLE"),
    ]

    table = bigquery.Table(fqn, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="timestamp_utc",
    )
    client.create_table(table)
    _TABLE_CACHE.add(fqn)


def _extract_metrics(state: dict[str, Any]) -> dict[str, Any]:
    outputs = state.get("agent_outputs") or {}
    tool_runs = 0
    sql_runs = 0
    sql_bytes_read = 0
    sql_queries: list[str] = []
    sql_query_metrics: list[dict[str, Any]] = []
    tool_runs_by_agent: dict[str, int] = {}
    error_flag = False
    error_messages: list[str] = []

    for agent_name, payload in outputs.items():
        metadata = (payload or {}).get("metadata") or {}
        trace = list(metadata.get("trace") or [])
        sqls = [str(q) for q in (metadata.get("sql_executed") or []) if q]
        sql_queries.extend(sqls)

        agent_tool_runs = 0
        for step in trace:
            tool = str(step.get("tool") or "")
            if not tool or tool.startswith("_"):
                continue
            tool_runs += 1
            agent_tool_runs += 1
            if tool == "run_sql":
                sql_runs += 1
                bytes_est = int(step.get("bytes_billed_estimate") or 0)
                sql_bytes_read += bytes_est
                sql_query_metrics.append(
                    {
                        "agent": agent_name,
                        "bytes_billed_estimate": bytes_est,
                        "row_count": int(step.get("row_count") or 0),
                        "error": step.get("error"),
                    }
                )
                if step.get("error"):
                    error_flag = True
                    error_messages.append(str(step.get("error")))

        tool_runs_by_agent[agent_name] = agent_tool_runs

        payload_meta_error = metadata.get("error")
        if payload_meta_error:
            error_flag = True
            error_messages.append(str(payload_meta_error))

    return {
        "tool_runs": tool_runs,
        "sql_runs": sql_runs,
        "sql_bytes_read": sql_bytes_read,
        "sql_gb_read": round(sql_bytes_read / 1_000_000_000, 6),
        "sql_queries": sql_queries,
        "sql_query_metrics": sql_query_metrics,
        "tool_runs_by_agent": tool_runs_by_agent,
        "error_flag": error_flag,
        "error_message": " | ".join(error_messages)[:4000] if error_messages else None,
    }


def log_chat_interaction(
    *,
    user_id: str,
    conversation_id: str,
    question_text: str,
    final_answer_text: str,
    state: dict[str, Any] | None,
    workflow_steps: list[dict[str, Any]] | None,
    usage_summary: dict[str, Any] | None,
    request_ts: datetime,
    response_ts: datetime,
    cache_hit: bool = False,
) -> None:
    """Persist one interaction row into BigQuery. Never raises to caller."""
    if not _enabled():
        return

    try:
        _ensure_table()

        st = state or {}
        usage = usage_summary or {}
        metrics = _extract_metrics(st)
        selected_agents = [str(a) for a in (st.get("selected_agents") or [])]
        primary_agent = str(st.get("selected_agent") or "") or None
        verifier_used = bool(st.get("verifier_verdict"))

        path_parts = ["plan", "execute"]
        if verifier_used:
            path_parts.append("verify")
        path_parts.append("compose")
        agent_path = " -> ".join(path_parts)
        if selected_agents:
            agent_path = f"{agent_path} [{', '.join(selected_agents)}]"

        fallback_flag = False
        for step in workflow_steps or []:
            if step.get("node") == "plan":
                reason = str(((step.get("output") or {}).get("reason") or "")).lower()
                if "fallback" in reason:
                    fallback_flag = True
                    break

        latency_ms = max(0, int((response_ts - request_ts).total_seconds() * 1000))

        row = {
            "event_id": str(uuid.uuid4()),
            "timestamp_utc": _now_utc().isoformat(),
            "request_ts_utc": request_ts.isoformat(),
            "response_ts_utc": response_ts.isoformat(),
            "latency_ms": latency_ms,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "question_text": question_text,
            "final_answer_text": final_answer_text,
            "selected_agents": selected_agents,
            "primary_agent": primary_agent,
            "agent_path": agent_path,
            "models_used": _resolve_model_names(),
            "token_input": int(usage.get("token_input", 0) or 0),
            "token_output": int(usage.get("token_output", 0) or 0),
            "token_total": int(usage.get("token_total", 0) or 0),
            "estimated_cost_usd": float(usage.get("estimated_cost_usd", 0.0) or 0.0),
            "error_flag": bool(metrics["error_flag"]),
            "fallback_flag": fallback_flag,
            "cache_hit": bool(cache_hit),
            "tool_runs": int(metrics["tool_runs"]),
            "sql_runs": int(metrics["sql_runs"]),
            "sql_bytes_read": int(metrics["sql_bytes_read"]),
            "sql_gb_read": float(metrics["sql_gb_read"]),
            "sql_queries": metrics["sql_queries"],
            "sql_query_metrics_json": _safe_json(metrics["sql_query_metrics"]),
            "tool_runs_by_agent_json": _safe_json(metrics["tool_runs_by_agent"]),
            "workflow_steps_json": _safe_json(workflow_steps or []),
            "verifier_verdict_json": _safe_json(st.get("verifier_verdict") or {}),
            "error_message": metrics["error_message"],
        }

        errors = _client().insert_rows_json(_table_fqn(), [row])
        if errors:
            logger.warning("chat observability insert errors: %s", errors)
    except Exception:
        logger.exception("failed to write chat observability row")
