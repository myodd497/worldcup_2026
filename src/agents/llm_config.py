"""Central LLM provider/model routing for agent ChatOpenAI clients."""
from __future__ import annotations

import os
from typing import Any, Iterator, Literal

from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from src.agents.llm_usage_tracker import capture_usage_from_response

LLMTier = Literal["simple", "complex"]


class _TrackedChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that records token/cost usage after every call."""

    _default_model: str = ""

    def invoke(self, *args: Any, **kwargs: Any) -> BaseMessage:
        response = super().invoke(*args, **kwargs)
        capture_usage_from_response(response, default_model=self._default_model)
        return response

    async def ainvoke(self, *args: Any, **kwargs: Any) -> BaseMessage:
        response = await super().ainvoke(*args, **kwargs)
        capture_usage_from_response(response, default_model=self._default_model)
        return response

    def generate(self, *args: Any, **kwargs: Any) -> LLMResult:
        result = super().generate(*args, **kwargs)
        capture_usage_from_response(result, default_model=self._default_model)
        return result

    async def agenerate(self, *args: Any, **kwargs: Any) -> LLMResult:
        result = await super().agenerate(*args, **kwargs)
        capture_usage_from_response(result, default_model=self._default_model)
        return result


def _provider() -> str:
    value = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    return value or "deepseek"


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
        # DeepSeek V4 defaults to thinking mode. Enable it explicitly on complex,
        # and disable it on simple to keep low-stakes replies short.
        if tier == "complex":
            effort = os.getenv("DEEPSEEK_REASONING_EFFORT", "medium")
            params["extra_body"] = {
                "thinking": {"type": "enabled"},
                "reasoning_effort": effort,
            }
        else:
            params["extra_body"] = {"thinking": {"type": "disabled"}}
    elif provider != "openai":
        raise ValueError(
            f"LLM_PROVIDER must be either 'deepseek' or 'openai' (got {provider!r})"
        )

    client = _TrackedChatOpenAI(**params)
    client._default_model = model
    return client
