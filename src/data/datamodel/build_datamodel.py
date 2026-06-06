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

Run modes:
  python -m src.data.datamodel.build_datamodel              # full pipeline
  python -m src.data.datamodel.build_datamodel --skip-raw   # dim+fact+mart only
  python -m src.data.datamodel.build_datamodel --only-marts # marts only
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Callable

from src.tools.api_usage_tracker import get_api_usage_snapshot, reset_api_usage

from src.data.datamodel import (
    raw_fixtures,
    raw_fixture_events,
    raw_fixture_statistics,
    raw_standings,
    dim_team,
    dim_competition,
    dim_venue,
    dim_date,
    fact_match,
    fact_match_team,
    fact_match_event,
    fact_standings_snapshot,
    mart_team_profile,
    mart_team_form,
    mart_head_to_head,
    mart_match_history,
    mart_match_upcoming,
    mart_tournament_state,
)

logger = logging.getLogger(__name__)


# Step name → callable returning dict (matches each module's `run()` contract).
RAW_STEPS: list[tuple[str, Callable[[], dict]]] = [
    ("raw_fixtures",            raw_fixtures.run),
    ("raw_fixture_events",      raw_fixture_events.run),
    ("raw_fixture_statistics",  raw_fixture_statistics.run),
    ("raw_standings",           raw_standings.run),
]

DIM_STEPS: list[tuple[str, Callable[[], dict]]] = [
    ("dim_team",        dim_team.run),
    ("dim_competition", dim_competition.run),
    ("dim_venue",       dim_venue.run),
    ("dim_date",        dim_date.run),
]

FACT_STEPS: list[tuple[str, Callable[[], dict]]] = [
    # fact_match MUST be first — other facts reference match_id keys.
    ("fact_match",              fact_match.run),
    ("fact_match_team",         fact_match_team.run),
    ("fact_match_event",        fact_match_event.run),
    ("fact_standings_snapshot", fact_standings_snapshot.run),
]

MART_STEPS: list[tuple[str, Callable[[], dict]]] = [
    ("mart_team_profile",     mart_team_profile.run),
    ("mart_team_form",        mart_team_form.run),
    ("mart_head_to_head",     mart_head_to_head.run),
    ("mart_match_history",    mart_match_history.run),
    ("mart_match_upcoming",   mart_match_upcoming.run),
    ("mart_tournament_state", mart_tournament_state.run),
]


def _run_steps(label: str, steps: list[tuple[str, Callable[[], dict]]]) -> list[dict]:
    logger.info("=" * 60)
    logger.info("%s LAYER — %d steps", label, len(steps))
    logger.info("=" * 60)
    results: list[dict] = []
    for name, fn in steps:
        start = time.time()
        logger.info("→ %s …", name)
        result = fn() or {}
        dur = round(time.time() - start, 2)
        rows = result.get("rows")
        logger.info("✓ %s done in %ss (rows=%s)", name, dur, rows)
        results.append({"step": name, "duration_s": dur, **result})
    return results


def run(*, skip_raw: bool = False, only_marts: bool = False) -> dict:
    """Execute the pipeline. Returns a dict with per-step results and API usage."""
    reset_api_usage()
    overall_start = time.time()
    out: dict = {"steps": []}

    if only_marts:
        out["steps"].extend(_run_steps("MART", MART_STEPS))
    else:
        if not skip_raw:
            out["steps"].extend(_run_steps("RAW",  RAW_STEPS))
        else:
            logger.info("Skipping RAW layer (--skip-raw)")
        out["steps"].extend(_run_steps("DIM",  DIM_STEPS))
        out["steps"].extend(_run_steps("FACT", FACT_STEPS))
        out["steps"].extend(_run_steps("MART", MART_STEPS))

    out["duration_s"] = round(time.time() - overall_start, 2)
    out["api_usage"]  = get_api_usage_snapshot()
    return out


def _print_summary(result: dict) -> None:
    print("=" * 60)
    print("DATAMODEL BUILD SUMMARY")
    print("=" * 60)
    for s in result.get("steps", []):
        print(f"  {s['step']:<28} rows={s.get('rows')!s:<10} {s.get('duration_s')}s")
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
    result = run(skip_raw=args.skip_raw, only_marts=args.only_marts)
    _print_summary(result)


if __name__ == "__main__":
    main()
