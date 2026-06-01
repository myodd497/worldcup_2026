"""Tracks API-Football request usage during ETL runs.

Best-effort quota extraction from response headers. Header names can vary
between plans/providers, so this module scans common variants.
"""
from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "total_calls": 0,
    "calls_by_endpoint": {},
    "requests_remaining": None,
    "requests_limit": None,
    "last_quota_headers": {},
}


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def reset_api_usage() -> None:
    with _LOCK:
        _STATE["total_calls"] = 0
        _STATE["calls_by_endpoint"] = {}
        _STATE["requests_remaining"] = None
        _STATE["requests_limit"] = None
        _STATE["last_quota_headers"] = {}


def record_api_call(endpoint: str, response_headers: dict[str, str] | None = None) -> None:
    headers = {str(k).lower(): str(v) for k, v in (response_headers or {}).items()}

    with _LOCK:
        _STATE["total_calls"] += 1

        calls_by_endpoint: dict[str, int] = _STATE["calls_by_endpoint"]
        calls_by_endpoint[endpoint] = calls_by_endpoint.get(endpoint, 0) + 1

        # Common quota/header variants exposed by API gateways.
        remaining = (
            _as_int(headers.get("x-ratelimit-requests-remaining"))
            or _as_int(headers.get("x-ratelimit-remaining"))
            or _as_int(headers.get("x-requests-remaining"))
            or _as_int(headers.get("x-ratelimit-remaining-day"))
        )
        limit = (
            _as_int(headers.get("x-ratelimit-requests-limit"))
            or _as_int(headers.get("x-ratelimit-limit"))
            or _as_int(headers.get("x-requests-limit"))
            or _as_int(headers.get("x-ratelimit-limit-day"))
        )

        if remaining is not None:
            _STATE["requests_remaining"] = remaining
        if limit is not None:
            _STATE["requests_limit"] = limit

        quota_headers = {
            k: v
            for k, v in headers.items()
            if "ratelimit" in k or "request" in k
        }
        if quota_headers:
            _STATE["last_quota_headers"] = quota_headers


def get_api_usage_snapshot() -> dict[str, Any]:
    with _LOCK:
        return {
            "total_calls": int(_STATE["total_calls"]),
            "calls_by_endpoint": dict(_STATE["calls_by_endpoint"]),
            "requests_remaining": _STATE["requests_remaining"],
            "requests_limit": _STATE["requests_limit"],
            "last_quota_headers": dict(_STATE["last_quota_headers"]),
        }
