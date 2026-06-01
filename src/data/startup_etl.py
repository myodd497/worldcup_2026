"""Startup ETL orchestration.

Runs the full ETL pipeline once per app process initialization.
Intended for app boot (FastAPI startup / Streamlit init), not per user chat.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone

from src.tools.api_usage_tracker import get_api_usage_snapshot, reset_api_usage

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


def _load_etl_runners() -> tuple:
    """Lazily imports ETL modules to avoid import-time crashes in app bootstrap."""
    try:
        from src.data.build_semantic_model import run as run_semantic_model
        from src.data.ingest_enriched import run_enriched_ingestion
        from src.data.ingest_fixture_stats import run_fixture_stats_ingestion
        from src.data.ingest_historical import run_ingestion as run_historical_ingestion
        from src.data.ingest_team_history import run_team_history_ingestion
    except Exception as exc:  # pragma: no cover - defensive for deploy/runtime env issues
        raise RuntimeError(
            "Unable to import ETL modules. Ensure runtime dependencies are installed "
            "(notably google-cloud-bigquery and related packages) and that src is on PYTHONPATH. "
            f"Underlying error: {exc}"
        )

    return (
        run_historical_ingestion,
        run_enriched_ingestion,
        run_team_history_ingestion,
        run_fixture_stats_ingestion,
        run_semantic_model,
    )


def _is_enabled() -> bool:
    value = os.getenv("RUN_FULL_ETL_ON_STARTUP", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def run_full_etl_once(trigger: str, force: bool = False) -> dict[str, object]:
    """Runs full ETL once per process.

    Order matters because later stages depend on earlier tables:
      1) historical fixtures
      2) enriched tables (includes team_stats)
      3) team match history (depends on team_stats)
      4) fixture stats (depends on team_match_history)
      5) semantic model build (facts/dims/views)
    """
    global _HAS_RUN
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    if not force and not _is_enabled():
        logger.info("Startup ETL skipped (RUN_FULL_ETL_ON_STARTUP disabled).")
        finished_at = datetime.now(timezone.utc)
        _record_status(
            run_id=run_id,
            trigger=trigger,
            status="SKIPPED_DISABLED",
            started_at=started_at,
            finished_at=finished_at,
            duration_s=round((finished_at - started_at).total_seconds(), 2),
        )
        return {"ran": False, "skipped": True, "reason": "disabled"}

    with _LOCK:
        if _HAS_RUN and not force:
            logger.info("Startup ETL already executed in this process. Skipping.")
            finished_at = datetime.now(timezone.utc)
            _record_status(
                run_id=run_id,
                trigger=trigger,
                status="SKIPPED_ALREADY_RAN",
                started_at=started_at,
                finished_at=finished_at,
                duration_s=round((finished_at - started_at).total_seconds(), 2),
            )
            return {"ran": False, "skipped": True, "reason": "already_ran"}

        start = time.time()
        reset_api_usage()
        logger.info("Starting full ETL pipeline (trigger=%s)", trigger)

        try:
            (
                run_historical_ingestion,
                run_enriched_ingestion,
                run_team_history_ingestion,
                run_fixture_stats_ingestion,
                run_semantic_model,
            ) = _load_etl_runners()

            run_historical_ingestion()
            run_enriched_ingestion()
            run_team_history_ingestion()
            run_fixture_stats_ingestion()
            run_semantic_model()

            _HAS_RUN = True
            duration_s = round(time.time() - start, 2)
            finished_at = datetime.now(timezone.utc)
            api_usage = get_api_usage_snapshot()
            _record_status(
                run_id=run_id,
                trigger=trigger,
                status="SUCCESS",
                started_at=started_at,
                finished_at=finished_at,
                duration_s=duration_s,
            )
            logger.info(
                "Full ETL pipeline completed in %ss (api_calls=%s, remaining=%s)",
                duration_s,
                api_usage.get("total_calls"),
                api_usage.get("requests_remaining"),
            )
            return {
                "ran": True,
                "skipped": False,
                "duration_s": duration_s,
                "api_usage": api_usage,
            }
        except Exception as exc:
            duration_s = round(time.time() - start, 2)
            finished_at = datetime.now(timezone.utc)
            api_usage = get_api_usage_snapshot()
            logger.exception(
                "Full ETL pipeline failed after %ss (api_calls=%s, remaining=%s)",
                duration_s,
                api_usage.get("total_calls"),
                api_usage.get("requests_remaining"),
            )
            _record_status(
                run_id=run_id,
                trigger=trigger,
                status="FAILED",
                started_at=started_at,
                finished_at=finished_at,
                duration_s=duration_s,
                error_message=str(exc),
            )
            raise
