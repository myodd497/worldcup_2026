"""Tiny embeddings layer with on-disk cache.

One function: `embed(texts) -> list[list[float]]`. Cached by SHA1(text) on disk
so cold starts are free after the first run and CI without network falls back
gracefully (callers should be ready to handle an empty return).

Used by:
  - schema_retriever (semantic table retrieval)
  - sql_few_shots    (semantic few-shot retrieval)
  - orchestrator     (semantic answer cache)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_MODEL = "text-embedding-3-small"   # 1536-d, $0.02 / 1M tokens
_DIM = 1536
_CACHE_DIR = Path(os.getenv("WC2026_EMB_CACHE_DIR", ".cache/embeddings"))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MEM_CACHE: dict[str, list[float]] = {}
_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            _client = OpenAI()
        except Exception as exc:
            logger.warning("OpenAI client unavailable for embeddings: %s", exc)
            _client = False  # sentinel: don't retry
    return _client or None


def _key(text: str) -> str:
    return hashlib.sha1(f"{_MODEL}::{text}".encode("utf-8")).hexdigest()


def _disk_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def _load(key: str) -> list[float] | None:
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]
    p = _disk_path(key)
    if p.exists():
        try:
            v = json.loads(p.read_text())
            if isinstance(v, list) and len(v) == _DIM:
                _MEM_CACHE[key] = v
                return v
        except Exception:
            pass
    return None


def _store(key: str, vec: list[float]) -> None:
    _MEM_CACHE[key] = vec
    try:
        _disk_path(key).write_text(json.dumps(vec))
    except Exception as exc:
        logger.debug("failed to persist embedding %s: %s", key, exc)


def embed(texts: Sequence[str]) -> list[list[float] | None]:
    """Return one vector per input. None when embedding could not be computed."""
    if not texts:
        return []
    out: list[list[float] | None] = [None] * len(texts)
    misses: list[tuple[int, str, str]] = []  # (index, key, text)
    for i, t in enumerate(texts):
        if not t:
            continue
        k = _key(t)
        cached = _load(k)
        if cached is not None:
            out[i] = cached
        else:
            misses.append((i, k, t))

    if not misses:
        return out

    client = _get_client()
    if client is None:
        return out  # network/key unavailable; caller falls back to keyword scoring

    try:
        resp = client.embeddings.create(model=_MODEL, input=[m[2] for m in misses])
        for (idx, key, _txt), item in zip(misses, resp.data):
            vec = list(item.embedding)
            _store(key, vec)
            out[idx] = vec
    except Exception as exc:
        logger.warning("embeddings.create failed: %s", exc)
    return out


def embed_one(text: str) -> list[float] | None:
    return embed([text])[0]


def cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))
