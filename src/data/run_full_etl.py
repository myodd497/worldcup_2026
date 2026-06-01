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
    usage = result.get("api_usage") if isinstance(result, dict) else None
    if isinstance(usage, dict):
        print("=" * 60)
        print("API Usage Summary")
        print(f"Total API calls: {usage.get('total_calls')}")
        print(f"Requests remaining: {usage.get('requests_remaining')}")
        print(f"Requests limit: {usage.get('requests_limit')}")
        print(f"Calls by endpoint: {usage.get('calls_by_endpoint')}")
        quota_headers = usage.get("last_quota_headers")
        if quota_headers:
            print(f"Quota headers: {quota_headers}")
        print("=" * 60)
