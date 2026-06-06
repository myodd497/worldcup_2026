"""Startup ETL orchestration.

Thin wrapper around `src.data.datamodel.build_datamodel.run()` that adds:
  - process-level singleton guard (`_HAS_RUN`)
  - `RUN_FULL_ETL_ON_STARTUP` env toggle
  - best-effort status row write to `etl_run_status` BQ table

The actual pipeline (raw → dim → fact → mart) lives in the datamodel package.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_HAS_RUN = False


def _status_table_fqn() -> str:
    project = os.environ["BIGQUERY_PROJECT_ID"]
    dataset = os.environ["BIGQUERY_DATASET_ID"]
    return f"`{project}.{dataset}.etl_run_status`"


def _record_status(
    *,
    run_id: str,
    trigger: str,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    duration_s: float,
    error_message: str | None = None,
) -> None:
    """Best-effort write of ETL execution status for ops visibility."""
    try:
        from src.tools import bigquery_tools as _bq_tools

        client = _bq_tools._client()
        table_ref = _status_table_fqn()
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_ref} (
            run_id STRING,
            trigger STRING,
            status STRING,
            started_at TIMESTAMP,
            finished_at TIMESTAMP,
            duration_s FLOAT64,
            error_message STRING,
            created_at TIMESTAMP
        )
        """
        client.query(create_sql).result()

        error_value = "NULL"
        if error_message:
            safe_error = error_message.replace("\\", "\\\\").replace("'", "''")
            error_value = f"'{safe_error[:4000]}'"

        insert_sql = f"""
        INSERT INTO {table_ref}
            (run_id, trigger, status, started_at, finished_at, duration_s, error_message, created_at)
        VALUES
            ('{run_id}', '{trigger}', '{status}', TIMESTAMP('{started_at.isoformat()}'),
             TIMESTAMP('{finished_at.isoformat()}'), {float(duration_s)}, {error_value}, CURRENT_TIMESTAMP())
        """
        client.query(insert_sql).result()
    except Exception as exc:  # pragma: no cover - best effort diagnostics only
        logger.warning("Could not write ETL status row: %s", exc)


def _is_enabled() -> bool:
    value = os.getenv("RUN_FULL_ETL_ON_STARTUP", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def run_full_etl_once(trigger: str, force: bool = False) -> dict[str, object]:
    """Runs the datamodel build once per process.

    Delegates to `src.data.datamodel.build_datamodel.run()` which executes the
    raw → dim → fact → mart pipeline.
    """
    global _HAS_RUN
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    if not force and not _is_enabled():
        logger.info("Startup ETL skipped (RUN_FULL_ETL_ON_STARTUP disabled).")
        finished_at = datetime.now(timezone.utc)
        _record_status(
            run_id=run_id, trigger=trigger, status="SKIPPED_DISABLED",
            started_at=started_at, finished_at=finished_at,
            duration_s=round((finished_at - started_at).total_seconds(), 2),
        )
        return {"ran": False, "skipped": True, "reason": "disabled"}

    with _LOCK:
        if _HAS_RUN and not force:
            logger.info("Startup ETL already executed in this process. Skipping.")
            finished_at = datetime.now(timezone.utc)
            _record_status(
                run_id=run_id, trigger=trigger, status="SKIPPED_ALREADY_RAN",
                started_at=started_at, finished_at=finished_at,
                duration_s=round((finished_at - started_at).total_seconds(), 2),
            )
            return {"ran": False, "skipped": True, "reason": "already_ran"}

        start = time.time()
        logger.info("Starting datamodel build (trigger=%s)", trigger)

        try:
            # Lazy import keeps app boot resilient if BQ deps are missing in some envs.
            from src.data.datamodel.build_datamodel import run as run_pipeline

            result = run_pipeline()

            _HAS_RUN = True
            duration_s = round(time.time() - start, 2)
            finished_at = datetime.now(timezone.utc)
            api_usage = result.get("api_usage") or {}
            _record_status(
                run_id=run_id, trigger=trigger, status="SUCCESS",
                started_at=started_at, finished_at=finished_at,
                duration_s=duration_s,
            )
            logger.info(
                "Datamodel build completed in %ss (api_calls=%s, remaining=%s)",
                duration_s,
                api_usage.get("total_calls"),
                api_usage.get("requests_remaining"),
            )
            return {
                "ran": True,
                "skipped": False,
                "duration_s": duration_s,
                "api_usage": api_usage,
                "steps": result.get("steps", []),
            }
        except Exception as exc:
            duration_s = round(time.time() - start, 2)
            finished_at = datetime.now(timezone.utc)
            logger.exception("Datamodel build failed after %ss", duration_s)
            _record_status(
                run_id=run_id, trigger=trigger, status="FAILED",
                started_at=started_at, finished_at=finished_at,
                duration_s=duration_s, error_message=str(exc),
            )
            raise
