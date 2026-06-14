"""Request-scoped LLM token and cost tracker.

This module stores usage in a context variable so each request can capture
its own totals even when tool runners execute in worker threads.
"""
from __future__ import annotations

import json
import os
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

_USAGE_CTX: ContextVar[dict[str, Any]] = ContextVar(
    "llm_usage",
    default={
        "token_input": 0,
        "token_output": 0,
        "token_total": 0,
        "estimated_cost_usd": 0.0,
        "models_used": {},
    },
)


def reset_usage_tracker() -> None:
    _USAGE_CTX.set(
        {
            "token_input": 0,
            "token_output": 0,
            "token_total": 0,
            "estimated_cost_usd": 0.0,
            "models_used": {},
        }
    )


def get_usage_summary() -> dict[str, Any]:
    usage = _USAGE_CTX.get()
    return {
        "token_input": int(usage.get("token_input", 0) or 0),
        "token_output": int(usage.get("token_output", 0) or 0),
        "token_total": int(usage.get("token_total", 0) or 0),
        "estimated_cost_usd": round(float(usage.get("estimated_cost_usd", 0.0) or 0.0), 8),
        "models_used": dict(usage.get("models_used") or {}),
    }


def capture_usage_from_response(response: Any, default_model: str | None = None) -> None:
    if response is None:
        return

    model_name = _extract_model_name(response) or (default_model or "unknown")
    in_tok, out_tok, total_tok = _extract_tokens(response)

    if in_tok == 0 and out_tok == 0 and total_tok == 0:
        return

    if total_tok == 0:
        total_tok = in_tok + out_tok

    usage = dict(_USAGE_CTX.get())
    usage["token_input"] = int(usage.get("token_input", 0) or 0) + in_tok
    usage["token_output"] = int(usage.get("token_output", 0) or 0) + out_tok
    usage["token_total"] = int(usage.get("token_total", 0) or 0) + total_tok

    price_in, price_out = _pricing_for_model(model_name)
    if price_in is not None and price_out is not None:
        delta_cost = (in_tok / 1_000_000.0) * price_in + (out_tok / 1_000_000.0) * price_out
        usage["estimated_cost_usd"] = float(usage.get("estimated_cost_usd", 0.0) or 0.0) + delta_cost

    models_used = dict(usage.get("models_used") or {})
    model_usage = dict(models_used.get(model_name) or {"input": 0, "output": 0, "total": 0})
    model_usage["input"] = int(model_usage.get("input", 0) or 0) + in_tok
    model_usage["output"] = int(model_usage.get("output", 0) or 0) + out_tok
    model_usage["total"] = int(model_usage.get("total", 0) or 0) + total_tok
    models_used[model_name] = model_usage
    usage["models_used"] = models_used

    _USAGE_CTX.set(usage)


def _extract_model_name(response: Any) -> str | None:
    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        name = meta.get("model_name") or meta.get("model")
        if name:
            return str(name)
    return None


def _extract_tokens(response: Any) -> tuple[int, int, int]:
    usage_meta = getattr(response, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        in_tok = int(usage_meta.get("input_tokens") or 0)
        out_tok = int(usage_meta.get("output_tokens") or 0)
        total_tok = int(usage_meta.get("total_tokens") or 0)
        return in_tok, out_tok, total_tok

    response_meta = getattr(response, "response_metadata", None)
    if isinstance(response_meta, dict):
        token_usage = response_meta.get("token_usage")
        if isinstance(token_usage, dict):
            in_tok = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
            out_tok = int(token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0)
            total_tok = int(token_usage.get("total_tokens") or 0)
            return in_tok, out_tok, total_tok

    return 0, 0, 0


@lru_cache(maxsize=1)
def _pricing_map() -> dict[str, dict[str, float]]:
    base: dict[str, dict[str, float]] = {
        "gpt-4o": {"input_per_1m": 5.0, "output_per_1m": 15.0},
        "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.6},
    }

    custom_json = os.getenv("LLM_PRICING_JSON", "").strip()
    if custom_json:
        try:
            parsed = json.loads(custom_json)
            if isinstance(parsed, dict):
                for model_name, prices in parsed.items():
                    if isinstance(prices, dict):
                        in_price = prices.get("input_per_1m")
                        out_price = prices.get("output_per_1m")
                        if in_price is not None and out_price is not None:
                            base[str(model_name)] = {
                                "input_per_1m": float(in_price),
                                "output_per_1m": float(out_price),
                            }
        except Exception:
            pass

    deepseek_in = os.getenv("DEEPSEEK_PRICE_INPUT_PER_1M", "").strip()
    deepseek_out = os.getenv("DEEPSEEK_PRICE_OUTPUT_PER_1M", "").strip()
    if deepseek_in and deepseek_out:
        try:
            deepseek_prices = {
                "input_per_1m": float(deepseek_in),
                "output_per_1m": float(deepseek_out),
            }
            base["deepseek-v4-flash"] = deepseek_prices
            base["deepseek-v4-pro"] = deepseek_prices
        except Exception:
            pass

    return base


def _pricing_for_model(model_name: str) -> tuple[float | None, float | None]:
    price_map = _pricing_map()
    if model_name in price_map:
        p = price_map[model_name]
        return float(p["input_per_1m"]), float(p["output_per_1m"])

    lowered = model_name.lower()
    for key, prices in price_map.items():
        if key.lower() in lowered:
            return float(prices["input_per_1m"]), float(prices["output_per_1m"])

    return None, None
