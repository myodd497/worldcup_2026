"""Central LLM provider/model routing for agent ChatOpenAI clients."""
from __future__ import annotations

import os
from typing import Any, Literal

from langchain_openai import ChatOpenAI

LLMTier = Literal["simple", "complex"]


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "deepseek").strip().lower()


def _model_for(provider: str, tier: LLMTier, *, tools: bool = False) -> str:
    if provider == "openai":
        if tier == "complex":
            return os.getenv("OPENAI_COMPLEX_MODEL", "gpt-4o")
        return os.getenv("OPENAI_SIMPLE_MODEL", "gpt-4o-mini")

    if tools:
        return os.getenv("DEEPSEEK_TOOL_MODEL", os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"))
    if tier == "complex":
        return os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro")
    return os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")


def create_chat_model(
    tier: LLMTier,
    *,
    temperature: float = 0,
    max_retries: int | None = None,
    timeout: int | None = None,
    tools: bool = False,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a ChatOpenAI-compatible client for the configured provider/tier."""
    provider = _provider()
    model = _model_for(provider, tier, tools=tools)

    params: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        **kwargs,
    }
    if max_retries is not None:
        params["max_retries"] = max_retries
    if timeout is not None:
        params["timeout"] = timeout

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        params["api_key"] = api_key
        params["base_url"] = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        # DeepSeek V4 supports thinking mode; enable it for the complex tier.
        if tier == "complex":
            effort = os.getenv("DEEPSEEK_REASONING_EFFORT", "high")
            params["extra_body"] = {
                "thinking": {"type": "enabled"},
                "reasoning_effort": effort,
            }
    elif provider != "openai":
        raise ValueError("LLM_PROVIDER must be either 'deepseek' or 'openai'")

    return ChatOpenAI(**params)
