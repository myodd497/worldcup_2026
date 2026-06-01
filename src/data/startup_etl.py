"""Startup ETL orchestration.

Runs the full ETL pipeline once per app process initialization.
Intended for app boot (FastAPI startup / Streamlit init), not per user chat.
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_HAS_RUN = False


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
    value = os.getenv("RUN_FULL_ETL_ON_STARTUP", "true").strip().lower()
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

    if not force and not _is_enabled():
        logger.info("Startup ETL skipped (RUN_FULL_ETL_ON_STARTUP disabled).")
        return {"ran": False, "skipped": True, "reason": "disabled"}

    with _LOCK:
        if _HAS_RUN and not force:
            logger.info("Startup ETL already executed in this process. Skipping.")
            return {"ran": False, "skipped": True, "reason": "already_ran"}

        start = time.time()
        logger.info("Starting full ETL pipeline (trigger=%s)", trigger)

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
        logger.info("Full ETL pipeline completed in %ss", duration_s)
        return {"ran": True, "skipped": False, "duration_s": duration_s}
