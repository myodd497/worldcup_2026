"""build_datamodel — orchestrator for the full datamodel pipeline.

Runs in strict dependency order:

    RAW  (API + legacy backfill)
      raw_fixtures
      raw_fixture_events
      raw_fixture_statistics
      raw_standings
        │
    DIM  (zero-API)
      dim_team, dim_competition, dim_venue, dim_date
        │
    FACT (zero-API)
      fact_match            (must run first)
      fact_match_team
      fact_match_event
      fact_standings_snapshot
        │
    MART (zero-API)
      mart_team_profile, mart_team_form, mart_head_to_head,
      mart_match_history, mart_match_upcoming, mart_tournament_state

Since-date scoping:
  Raw steps for events/stats/standings are scoped to a sliding window that
  starts from the day after the last successful RAW layer completion (read
  from the etl_run_status metadata table).  If no prior success is on record,
  the window defaults to 14 days.  This guarantees that a single daily
  failure is automatically recovered the next day without burning API quota
  on unlimited historical backfill.

Error resilience:
  Every per-step call is wrapped in try/except so one misbehaving module
  cannot kill the entire pipeline (DIM / FACT / MART layers are independent).

Run modes:
  python -m src.data.datamodel.build_datamodel              # full pipeline
  python -m src.data.datamodel.build_datamodel --skip-raw   # dim+fact+mart only
  python -m src.data.datamodel.build_datamodel --only-marts # marts only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from src.tools.api_usage_tracker import get_api_usage_snapshot, reset_api_usage
from src.tools import bigquery_tools as _bq_tools

from src.data.datamodel import (
    raw_fixtures,
    raw_fixture_events,
    raw_fixture_statistics,
    raw_standings,
    raw_player_stats,
    dim_team,
    dim_competition,
    dim_venue,
    dim_date,
    dim_player,
    fact_match,
    fact_match_team,
    fact_match_event,
    fact_standings_snapshot,
    fact_player_match_stat,
    mart_team_profile,
    mart_team_form,
    mart_head_to_head,
    mart_match_history,
    mart_match_upcoming,
    mart_tournament_state,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Since-date helper — self-healing sliding window
# ---------------------------------------------------------------------------

_SINCE_FALLBACK_DAYS = 14  # generous default when no prior success is recorded


def _etl_status_fqn() -> str:
    project = os.environ["BIGQUERY_PROJECT_ID"]
    dataset = os.environ["BIGQUERY_DATASET_ID"]
    return f"`{project}.{dataset}.etl_run_status`"


def _compute_since_date() -> str:
    """Return a 'YYYY-MM-DD' ISO date that covers the gap since the last
    successful RAW-layer completion recorded in etl_run_status.

    If no prior success row exists, falls back to _SINCE_FALLBACK_DAYS ago
    so the first run backfills a reasonable window without hitting the full
    historical dataset.
    """
    try:
        sql = f"""
        SELECT MAX(DATE(finished_at)) AS last_success_date
        FROM {_etl_status_fqn()}
        WHERE status = 'SUCCESS'
          AND trigger IN ('github_action', 'fastapi_startup')
        """
        df = _bq_tools.run_query(sql)
        if not df.empty and df["last_success_date"].iloc[0] is not None:
            last_dt = df["last_success_date"].iloc[0]
            # +1 day so we re-process the day after the last success
            # (covers partial failures in the same day)
            since = last_dt + timedelta(days=1)
            logger.info(
                "_compute_since_date: last success date=%s → since=%s", last_dt, since
            )
            return since.isoformat()
    except Exception as exc:
        logger.warning("Could not query etl_run_status for since_date: %s", exc)

    fallback = date.today() - timedelta(days=_SINCE_FALLBACK_DAYS)
    logger.info("_compute_since_date: no prior success found, using fallback %s", fallback)
    return fallback.isoformat()


# ---------------------------------------------------------------------------
# Step lists (RAW steps wrapped with scoping lambdas)
# ---------------------------------------------------------------------------

def _build_raw_steps(
    since_date: str,
    skip_team_fetch: bool = True,
) -> list[tuple[str, Callable[[], dict]]]:
    """Return RAW step list with scoping lambdas.

    * raw_fixtures: skip per-team fetch to avoid burning 300+ API calls
      (the league-level fetch already covers all fixtures).
    * raw_fixture_events: scoped to since_date so only recent missing
      events are backfilled.
    * raw_fixture_statistics: same scoping.
    * raw_standings: unchanged (already idempotent, deduped by snapshot_date).
    """
    return [
        (
            "raw_fixtures",
            lambda: raw_fixtures.run(
                skip_team_fetch=skip_team_fetch,
                since_date=since_date,
            ),
        ),
        (
            "raw_fixture_events",
            lambda: raw_fixture_events.run(since_date=since_date),
        ),
        (
            "raw_fixture_statistics",
            lambda: raw_fixture_statistics.run(since_date=since_date),
        ),
        (
            "raw_player_stats",
            lambda: raw_player_stats.run(since_date=since_date),
        ),
        ("raw_standings", raw_standings.run),
    ]


# Legacy unscoped steps (used only for --skip-raw or --only-marts)
DIM_STEPS: list[tuple[str, Callable[[], dict]]] = [
    ("dim_team",        dim_team.run),
    ("dim_competition", dim_competition.run),
    ("dim_venue",       dim_venue.run),
    ("dim_date",        dim_date.run),
    ("dim_player",      dim_player.run),
]

FACT_STEPS: list[tuple[str, Callable[[], dict]]] = [
    # fact_match MUST be first — other facts reference match_id keys.
    ("fact_match",              fact_match.run),
    ("fact_match_team",         fact_match_team.run),
    ("fact_match_event",        fact_match_event.run),
    ("fact_standings_snapshot", fact_standings_snapshot.run),
    ("fact_player_match_stat",  fact_player_match_stat.run),
]

MART_STEPS: list[tuple[str, Callable[[], dict]]] = [
    ("mart_team_profile",     mart_team_profile.run),
    ("mart_team_form",        mart_team_form.run),
    ("mart_head_to_head",     mart_head_to_head.run),
    ("mart_match_history",    mart_match_history.run),
    ("mart_match_upcoming",   mart_match_upcoming.run),
    ("mart_tournament_state", mart_tournament_state.run),
]


# ---------------------------------------------------------------------------
# Step runner with per-step error resilience
# ---------------------------------------------------------------------------

def _run_steps(label: str, steps: list[tuple[str, Callable[[], dict]]]) -> list[dict]:
    """Execute a list of named callables.

    Each step is wrapped in try/except so a failure in one step does not
    prevent downstream steps from executing (one bad raw_event API call
    shouldn't kill the entire FACT/MART build).
    """
    logger.info("=" * 60)
    logger.info("%s LAYER — %d steps", label, len(steps))
    logger.info("=" * 60)
    results: list[dict] = []
    for name, fn in steps:
        start = time.time()
        logger.info("→ %s …", name)
        try:
            result = fn() or {}
            dur = round(time.time() - start, 2)
            rows = result.get("rows")
            api_calls = result.get("api_calls")
            ingest_info = result.get("ingest") or {}
            logger.info(
                "✓ %s done in %ss (rows=%s, api_calls=%s, ingest=%s)",
                name, dur, rows, api_calls, ingest_info,
            )
            results.append({"step": name, "duration_s": dur, **result})
        except Exception as exc:
            dur = round(time.time() - start, 2)
            logger.exception("✗ %s FAILED after %ss: %s", name, dur, exc)
            results.append({
                "step": name,
                "duration_s": dur,
                "error": str(exc),
                "error_type": type(exc).__name__,
            })
    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(*, skip_raw: bool = False, only_marts: bool = False) -> dict:
    """Execute the pipeline. Returns a dict with per-step results and API usage.

    When running with RAW (default), the events/stats sub-steps are scoped by
    `_compute_since_date()` to avoid burning API quota on unlimited history.
    """
    reset_api_usage()
    overall_start = time.time()
    out: dict = {"steps": []}

    if only_marts:
        out["steps"].extend(_run_steps("MART", MART_STEPS))
    else:
        if not skip_raw:
            since_date = _compute_since_date()
            logger.info("RAW since_date scoping = %s", since_date)
            raw_steps = _build_raw_steps(since_date=since_date, skip_team_fetch=True)
            out["steps"].extend(_run_steps("RAW", raw_steps))
        else:
            logger.info("Skipping RAW layer (--skip-raw)")
        out["steps"].extend(_run_steps("DIM",  DIM_STEPS))
        out["steps"].extend(_run_steps("FACT", FACT_STEPS))
        out["steps"].extend(_run_steps("MART", MART_STEPS))

    out["duration_s"] = round(time.time() - overall_start, 2)
    out["api_usage"]  = get_api_usage_snapshot()
    return out


# ---------------------------------------------------------------------------
# etl_run_status writer (used by both CLI and FastAPI startup paths)
# ---------------------------------------------------------------------------

def _detect_trigger() -> str:
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        return "github_action"
    return "manual"


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
    """Best-effort write of a status row to etl_run_status. Never raises."""
    try:
        from google.cloud import bigquery

        client = _bq_tools._client()
        table_ref = _etl_status_fqn()
        client.query(
            f"""
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
        ).result()

        error_value = "NULL"
        if error_message:
            safe = error_message.replace("\\", "\\\\").replace("'", "''")
            error_value = f"'{safe[:4000]}'"

        client.query(
            f"""
            INSERT INTO {table_ref}
                (run_id, trigger, status, started_at, finished_at, duration_s, error_message, created_at)
            VALUES
                ('{run_id}', '{trigger}', '{status}',
                 TIMESTAMP('{started_at.isoformat()}'),
                 TIMESTAMP('{finished_at.isoformat()}'),
                 {float(duration_s)}, {error_value}, CURRENT_TIMESTAMP())
            """
        ).result()
    except Exception as exc:  # pragma: no cover - observability only
        logger.warning("Could not write ETL status row: %s", exc)


def _failed_steps(result: dict) -> list[str]:
    return [s["step"] for s in result.get("steps", []) if s.get("error")]


def _print_summary(result: dict) -> None:
    print("=" * 60)
    print("DATAMODEL BUILD SUMMARY")
    print("=" * 60)
    for s in result.get("steps", []):
        err = s.get("error")
        status = f"  ERROR: {err}" if err else f"  rows={s.get('rows')!s:<10}"
        print(f"  {s['step']:<28} {status} {s.get('duration_s')}s")
    print(f"Total duration: {result.get('duration_s')}s")
    usage = result.get("api_usage") or {}
    print("-" * 60)
    print("API Usage")
    print(f"  total_calls       = {usage.get('total_calls')}")
    print(f"  requests_remaining= {usage.get('requests_remaining')}")
    print(f"  requests_limit    = {usage.get('requests_limit')}")
    print(f"  calls_by_endpoint = {usage.get('calls_by_endpoint')}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the World Cup 2026 datamodel.")
    parser.add_argument("--skip-raw",   action="store_true", help="Skip RAW layer (no API calls).")
    parser.add_argument("--only-marts", action="store_true", help="Rebuild marts only.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    run_id = str(uuid.uuid4())
    trigger = _detect_trigger()
    started_at = datetime.now(timezone.utc)
    sha = os.getenv("GITHUB_SHA", "local")
    logger.info("build_datamodel start run_id=%s trigger=%s sha=%s", run_id, trigger, sha[:12])

    status = "SUCCESS"
    error_message: str | None = None
    result: dict = {"steps": []}
    try:
        result = run(skip_raw=args.skip_raw, only_marts=args.only_marts)
        failed = _failed_steps(result)
        if failed:
            status = "PARTIAL_FAILURE"
            error_message = f"Failed steps: {', '.join(failed)}"
    except Exception as exc:
        status = "FAILED"
        error_message = f"{type(exc).__name__}: {exc}"
        logger.exception("build_datamodel crashed")
    finally:
        finished_at = datetime.now(timezone.utc)
        duration_s = round((finished_at - started_at).total_seconds(), 2)
        _record_status(
            run_id=run_id, trigger=trigger, status=status,
            started_at=started_at, finished_at=finished_at,
            duration_s=duration_s, error_message=error_message,
        )
        _print_summary(result)
        print(f"run_id={run_id} trigger={trigger} status={status}")

    if status != "SUCCESS":
        sys.exit(1)


if __name__ == "__main__":
    main()
