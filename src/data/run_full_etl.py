"""Runs the full ETL pipeline for scheduled jobs.

Usage:
    set -a && source .env && set +a
    .venv/bin/python -m src.data.run_full_etl
"""
from __future__ import annotations

from src.data.startup_etl import run_full_etl_once


def run() -> dict[str, object]:
    # force=True bypasses startup toggle because this module is for scheduled runs.
    return run_full_etl_once(trigger="scheduler", force=True)


if __name__ == "__main__":
    result = run()
    print(result)
