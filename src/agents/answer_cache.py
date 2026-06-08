"""Semantic answer cache for the orchestrator.

Caches `(question, conversation_context_digest) → final_reply` using cosine
similarity over question embeddings. Avoids burning LLM/BQ calls on
repeated / paraphrased questions.

In-memory only. Cleared on process restart by design — we don't want a
stale cache surviving a data refresh.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from src.agents.embeddings import cosine, embed_one

_SIMILARITY_THRESHOLD = 0.97
_MAX_ENTRIES = 256
_TTL_SECONDS = 30 * 60  # 30 min — short enough to respect ETL refreshes


@dataclass
class _Entry:
    question: str
    context_digest: str
    embedding: list[float]
    reply: str
    created_at: float


_CACHE: list[_Entry] = []


def _digest(context: str | None) -> str:
    return hashlib.sha1((context or "").encode("utf-8")).hexdigest()[:16]


def _purge_expired() -> None:
    now = time.time()
    _CACHE[:] = [e for e in _CACHE if now - e.created_at < _TTL_SECONDS]


def lookup(question: str, context: str | None = None) -> str | None:
    if not question or not question.strip():
        return None
    _purge_expired()
    vec = embed_one(question.strip())
    if vec is None:
        return None
    ctx = _digest(context)
    best: tuple[float, _Entry] | None = None
    for entry in _CACHE:
        if entry.context_digest != ctx:
            continue
        sim = cosine(vec, entry.embedding)
        if sim >= _SIMILARITY_THRESHOLD and (best is None or sim > best[0]):
            best = (sim, entry)
    return best[1].reply if best else None


def store(question: str, reply: str, context: str | None = None) -> None:
    if not question or not reply:
        return
    vec = embed_one(question.strip())
    if vec is None:
        return
    if len(_CACHE) >= _MAX_ENTRIES:
        # Evict oldest.
        _CACHE.sort(key=lambda e: e.created_at)
        del _CACHE[: max(1, _MAX_ENTRIES // 4)]
    _CACHE.append(_Entry(
        question=question.strip(),
        context_digest=_digest(context),
        embedding=vec,
        reply=reply,
        created_at=time.time(),
    ))


def clear() -> None:
    _CACHE.clear()
